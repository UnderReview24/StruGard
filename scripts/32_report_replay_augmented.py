"""Aggregate saved CSVs and render six diagnostic figures. No model training."""
import _bootstrap
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.continual.replay_augmented_runner import METHODS

ROOT=Path('outputs/replay_augmented_rae')
LABELS=dict(zip(METHODS,['A Full + Structural Replay','B Old RAE, no replay','C RAE + Random Replay',
                      'D RAE + Structural Replay','E Structural RAE, no Expand','F Structural RAE, Always Expand']))
COLORS=dict(zip(METHODS,['#333333','#AA6A5D','#D09B45','#386E9D','#73936B','#8B79A6']))


def aggregate():
    assert json.loads((ROOT/'audit.json').read_text())['status']=='passed'
    final=pd.concat([pd.read_csv(ROOT/'runs'/m/'final_results.csv') for m in METHODS],ignore_index=True)
    stages=pd.concat([pd.read_csv(ROOT/'runs'/m/'stage_metrics.csv') for m in METHODS],ignore_index=True)
    router=pd.concat([pd.read_csv(ROOT/'runs'/m/'router_diagnostics.csv') for m in METHODS],ignore_index=True)
    samples=pd.concat([pd.read_csv(ROOT/'runs'/m/'sample_predictions.csv') for m in METHODS],ignore_index=True)
    for name,table in [('final_results',final),('stage_metrics',stages),('router_diagnostics',router),('sample_predictions',samples)]:
        table.to_csv(ROOT/(name+'.csv'),index=False)
    matrix=[]
    for row in stages.to_dict('records'):
        matrix.append(dict(method=row['method'],stage_id=row['stage_id'],
            **{'domain_'+d:mae for d,mae in json.loads(row['per_domain_mae']).items()}))
    pd.DataFrame(matrix).to_csv(ROOT/'continual_mae_matrix.csv',index=False)
    compute=['method','stage_id','optimizer_steps','backward_steps','current_sample_exposures','current_gradient_exposures',
             'replay_sample_exposures','wall_clock_time','peak_gpu_memory_bytes','total_parameters','trainable_parameters',
             'max_trainable_parameters','memory_size','replay_training_memory','retention_probe_size']
    stages[compute].to_csv(ROOT/'compute_metrics.csv',index=False)
    expert=router.merge(stages[['method','stage_id','num_experts','trainable_parameters','total_parameters']],on=['method','stage_id'])
    expert.to_csv(ROOT/'expert_history.csv',index=False)
    retention=router[router.stage_id>1][[c for c in ['method','stage_id','decision','L_hist','L_new_before_adapt','L_new_after_adapt',
        'L_old_before','L_old_after','Delta_old','relative_old_degradation','new_domain_failed','retention_failed','expansion_reason'] if c in router]]
    retention.to_csv(ROOT/'retention_metrics.csv',index=False)
    route_counts=[]
    for m in METHODS:
        frame=stages[(stages.method==m)&(stages.stage_id>1)]
        weights=frame.current_sample_exposures
        route_counts.append(dict(method=m,**{a:float((frame[a+'_fraction']*weights).sum()/weights.sum()) for a in ['reuse','adapt','expand']},
            novel=float((frame.get('novel_fraction',pd.Series(0.,index=frame.index)).fillna(0.)*weights).sum()/weights.sum())))
    pd.DataFrame(route_counts).to_csv(ROOT/'action_fractions.csv',index=False)
    old=[]
    for method in ['structural_replay','rae','no_replay']:
        p=pd.read_csv(Path('outputs/pilot')/method/'final_results.csv').iloc[0]
        old.append(dict(method=method,average_seen_mae=float(p.average_seen_mae),source='preserved historical pilot; different training protocol'))
    pd.DataFrame(old).to_csv(ROOT/'historical_reference.csv',index=False)
    shutil.copy2(ROOT/'runs/D_structural_rae/config.yaml',ROOT/'resolved_config.yaml')
    return final,stages,router


def figures():
    # Every plotted value is loaded from a saved CSV, including derived statistics.
    stages=pd.read_csv(ROOT/'stage_metrics.csv')
    matrix=pd.read_csv(ROOT/'continual_mae_matrix.csv')
    samples=pd.read_csv(ROOT/'sample_predictions.csv')
    retention=pd.read_csv(ROOT/'retention_metrics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'axes.titleweight':'semibold','axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6})
    def save(fig,name):
        fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(ROOT/(name+'.png'),dpi=150,bbox_inches='tight')
        plt.close(fig)
    for column,ylabel,name in [
        ('average_seen_domain_mae','Mean MAE across seen domains (eV/atom)','average_seen_mae_vs_stage'),
        ('forgetting','Mean forgetting on prior domains (eV/atom)','forgetting_vs_stage'),
        ('num_experts','Number of experts','number_of_experts_vs_stage')]:
        fig,ax=plt.subplots(figsize=(8.8,4.1))
        for m in METHODS:
            frame=stages[stages.method==m]
            ax.plot(frame.stage_id,frame[column],marker='o',markersize=5,color=COLORS[m],label=LABELS[m],
                    linewidth=2.6 if m==METHODS[3] else 1.6,linestyle='--' if m==METHODS[1] else '-')
        ax.set(xlabel='Continual stage (domain order: 2, 1, 0)',ylabel=ylabel,xticks=[1,2,3])
        if column=='num_experts':ax.set_yticks(range(1,int(stages.num_experts.max())+1))
        ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=9)
        ax.set_title('Fixed 5k diagnostic benchmark | seed 42',loc='left',pad=12)
        save(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.6),sharey=True)
    for ax,d,title in zip(axes,[2,1,0],['D1: triclinic / monoclinic','D2: orthorhombic / tetragonal','D3: cubic / hexagonal / trigonal']):
        for m in METHODS:
            frame=matrix[matrix.method==m]
            ax.plot(frame.stage_id,frame['domain_'+str(d)],marker='o',color=COLORS[m],label=LABELS[m],linewidth=2.4 if m==METHODS[3] else 1.5)
        ax.set(title=title,xlabel='Continual stage',xticks=[1,2,3])
    axes[0].set_ylabel('Domain test MAE (eV/atom)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.21),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout()
    save(fig,'per_domain_mae_vs_stage')
    frame=samples[(samples.method==METHODS[3])&(samples.stage_id==3)&(samples.split=='test')]
    rho=float(spearmanr(frame.stable_novelty,frame.absolute_error).statistic)
    pd.DataFrame([dict(method=METHODS[3],stage_id=3,split='test',n=len(frame),spearman_rho=rho,
                      interpretation='post-training descriptive association, no tuning')]).to_csv(ROOT/'novelty_error_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.9,4.5))
    for d,color in zip([2,1,0],['#386E9D','#73936B','#D09B45']):
        group=frame[frame.domain_id==d]
        ax.scatter(group.stable_novelty,group.absolute_error,s=22,alpha=.6,color=color,label='Domain '+str(d),edgecolors='none')
    ax.set(xlabel='Stable novelty (nearest-prototype distance)',ylabel='Absolute test error (eV/atom)',
           title=f'D: Structural RAE | final stage | n={len(frame)} | Spearman rho={rho:.3f}')
    ax.legend(frameon=False)
    save(fig,'novelty_vs_absolute_error')
    frame=retention.dropna(subset=['L_old_before','L_old_after']).copy()
    fig,ax=plt.subplots(figsize=(9.1,5.7))
    for i,row in enumerate(frame.itertuples()):
        color=COLORS[row.method]
        ax.plot([row.L_old_before,row.L_old_after],[i,i],color=color,linewidth=2.3)
        ax.scatter(row.L_old_before,i,color='white',edgecolors=color,s=48,zorder=3)
        ax.scatter(row.L_old_after,i,color=color,marker='s',s=38,zorder=3)
    ax.set_yticks(range(len(frame)),[r.method[0]+': stage '+str(r.stage_id)+' / '+r.decision for r in frame.itertuples()])
    ax.invert_yaxis()
    ax.set(xlabel='MAE on fixed historical retention probe (eV/atom)',title='Retention before and after the first half of stage training')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',color='#444444',markerfacecolor='white',linestyle='',label='Before'),
                       Line2D([],[],marker='s',color='#444444',linestyle='',label='After')],loc='best',frameon=False)
    fig.text(.12,-.01,'F measures early forced expansion. REUSE has no updates. B has no replay probe.',fontsize=9,color='#555555')
    fig.tight_layout()
    save(fig,'retention_before_after_adaptation')


def report(final,stages,router):
    indexed=final.set_index('method');a,b,c,d,e,f=[indexed.loc[m] for m in METHODS]
    hist=pd.read_csv(ROOT/'historical_reference.csv').set_index('method')
    old_default=float(hist.loc['rae','average_seen_mae'])
    identical=json.loads((ROOT/'selection_identifiability.json').read_text())['random_and_structural_training_exposures_identical']
    lines=['# SP-Crystal-R：5k 诊断结果','',
        '范围：既有 5000 条数据、3 个域、seed=42、顺序 [2,1,0]、固定 503 个测试结构。A–F 全部完成并停止训练。',
        '检查点复现通过：旧 Structural Replay 的 MAE=0.10206509；旧无回放 R/A/E=0.18195963。这是保存权重的重新评估，不是旧协议独立训练复现。',
        '旧默认 R/A/E=0.16393394 已经包含回放和蒸馏；B 使用你指定的原无回放实现。新 A 使用修正后的联合批次协议，因此不能与旧 0.102065 的训练成本混为一谈。','',
        '| 组别 | 平均 MAE | 遗忘 | 专家数 | 最终记忆 | 优化步数 | 回放曝光量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,row in indexed.iterrows():
        lines.append(f'| {LABELS[m]} | {row.final_avg_mae:.6f} | {row.final_forgetting:.6f} | {int(row.num_experts)} | {int(row.memory_size)} | {int(row.optimizer_steps)} | {int(row.replay_exposures)} |')
    lines += ['','## 逐阶段表现','', '| 组别 | S1 平均 MAE | S2 平均 MAE | S3 平均 MAE | S1 当前域 MAE | S2 当前域 MAE | S3 当前域 MAE |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        t=stages[stages.method==m].sort_values('stage_id')
        lines.append('| '+m+' | '+' | '.join(f'{v:.6f}' for v in list(t.average_seen_domain_mae)+list(t.current_domain_mae))+' |')
    lines += ['','每个已见域的完整 MAE 矩阵保存在 continual_mae_matrix.csv；遗忘按旧项目的 MAE 定义重新计算。','',
              '## 四个问题','',
              f'Q1：新结构回放 R/A/E 相对本次 B 的 MAE 变化为 {(d.final_avg_mae/b.final_avg_mae-1)*100:+.2f}%；相对历史默认 R/A/E 为 {(d.final_avg_mae/old_default-1)*100:+.2f}%。'+
              ('误差降低。' if d.final_avg_mae<b.final_avg_mae else '未改善旧无回放 R/A/E。')+' B 与 D 还存在表征、控制器和训练预算差异，不能把全部差值归因于回放。','',
              f'Q2：结构选择 D={d.final_avg_mae:.6f}，同框架随机选择 C={c.final_avg_mae:.6f}；'+
              ('两组实际训练回放索引、动作、初始参数完全相同，不能据此认定结构选择优于随机选择。D1 的1427条全部装入2000条预算；真正发生选择差异后，D3却进入REUSE，没有梯度更新。微小数值差异不能归因于记忆选择。' if identical else ('本次结构选择数值更好，但单种子不足以建立稳健优势。' if d.final_avg_mae<c.final_avg_mae else '本次没有胜过随机选择。')),' ',
              f'Q3：D={d.final_avg_mae:.6f}，禁用扩展 E={e.final_avg_mae:.6f}；'+('本次完整控制器更好。' if d.final_avg_mae<e.final_avg_mae else '本次扩展没有带来更低 MAE。')+
              f' 但始终扩展 F={f.final_avg_mae:.6f} 优于 E，不能断言扩展本身没有价值；本次更直接暴露的是当前按需控制策略的不足。D/E 的后续路由和有效更新次数随专家数变化，因此同时报告计算量。','',
              f'Q4：D 与新全骨干回放 A={a.final_avg_mae:.6f} 相差 {d.final_avg_mae-a.final_avg_mae:+.6f}，相对差距 {(d.final_avg_mae/a.final_avg_mae-1)*100:+.2f}%；使用 {int(d.num_experts)} 个专家。'+
              ('达到本次全骨干回放水平。' if d.final_avg_mae<=a.final_avg_mae else '仍未达到本次全骨干回放水平。'),' ',
              '上述比较都是固定单种子的诊断结果，不是统计显著性结论。没有根据测试结果调阈值、追加训练或扩大样本。','',
              '## 控制器与记忆','',
              '比例按实际当前样本曝光量统计；B 的原路由另有 NOVEL（暂不更新）状态。S1 为共享初始化，不计入 R/A/E 比例。','']
    fractions=pd.read_csv(ROOT/'action_fractions.csv').set_index('method')
    for m in METHODS:
        r=fractions.loc[m]
        lines.append(f'- {m}：Reuse {r.reuse:.2%} / Adapt {r.adapt:.2%} / Expand {r.expand:.2%} / NOVEL {r.novel:.2%}。')
    for event in router[(router.method==METHODS[3])&(router.stage_id>1)].to_dict('records'):
        lines.append(f"- 主模型 S{event['stage_id']}：{event['decision']}；触发原因 {event['expansion_reason']}；新域验证 MAE {event['L_new_before_adapt']:.6f}→{event['L_new_after_adapt']:.6f}；保留探针 MAE {event['L_old_before']:.6f}→{event['L_old_after']:.6f}。")
    lines += ['','结构记忆复用原来的 D1 冻结向量、域均衡 k-center 和 6000 候选上限。每阶段训练前从历史记忆划出 20% 固定探针，余下 80% 参与联合回放；探针全程不进入本阶段梯度。2000 条为总预算，包含探针。',
        'S1 后记忆 1427；S2 时回放训练/探针=1142/285；S2 后记忆 2000；S3 时=1600/400；S3 后仍为2000。B 记忆始终0。',
        '', '## 配置、计算量与限制','',
        '配置完整保存于 resolved_config.yaml。h_s 固定 D1 坐标，h_p 最后两层 GCN 可更新；残差 h_s−μ，专家输入 [h_p,h_s−μ]。稳定层 lr=0，慢速层1e-5，专家与头1e-3；rank16、dropout0.1、AdamW、weight_decay1e-4、clip5。',
        '新方法每次32当前+32回放（末批按当前数量匹配），Huber_current+Huber_replay，一次前向/反向/优化。无蒸馏。S2/S3 每阶段10遍，前5遍为适应探测，最后状态作为检查点。REUSE 仅推理一遍，零优化更新。γ_new=γ_old=0.10；新域验证使用选定专家，保留探针使用真实最近专家路由。',
        'A/C/D/E/F 采用同一联合批次协议；B 保留原实现的冻结骨干与验证早停。参数、峰值显存、曝光量、逐阶段耗时见 compute_metrics.csv。总参数含实际驻留的冻结表征参数；A 的原 D1 选择向量从缓存读取，无额外驻留的完整选择编码器。',
        '优化步数不包含已经完成并共同复用的 D1 训练。新方法逐阶段耗时包含路由、探针与记忆更新，排除最终测试和检查点写盘；B 的计时为 fit 循环（含验证），因此时间并非严格同口径。CUDA scatter 未强制逐位确定性。',
        '遗留 B 在训练前也读取测试集用于旧诊断图，但这些数值不会输入路由或优化。新 A/C/D/E/F 只在决策与训练完成后测试。此差别已写入 leakage_audit.txt。',
        '本次失败假设：主模型第二域的联合适应同时未解决新域误差与旧知识退化，触发 both；第三域的几何新颖度低于门限，直接 REUSE，即便其当前验证误差仍高。低结构新颖度不能保证预测已经足够准确；冻结旧专家参数也不能阻止共享慢速骨干变化影响旧域预测。这是日志支持的失败现象，没有通过额外训练或测试集阈值搜索修饰。',
        '', '## 文件与审计','',
        '新增文件：src/models/sp_crystal_replay.py；src/continual/{replay_augmented_rae.py,replay_augmented_runner.py,replay_legacy_runner.py}；configs/experiment/replay_augmented.yaml；tests/test_replay_augmented.py；scripts/{30_run_replay_augmented.py,31_audit_replay_augmented.py,32_report_replay_augmented.py}；REPLAY_AUGMENTED_IMPLEMENTATION.md。',
        '验证：51项测试通过；300条烟雾测试通过；全部新方法记忆选择按旧策略独立重建一致；探针/梯度索引隔离、固定测试集、实际优化调用计数与检查点逐样本预测复评通过；739个已有代码、配置、划分与结果文件哈希保持一致。',
        '原始日志、逐步样本索引、记忆状态、检查点及分阶段证据位于远程 outputs/replay_augmented_rae/runs/。下载包包含代码、CSV、图和审计证据；大型检查点留在远程原位。',
        '训练已停止。未启动额外种子、30k、全量或可选回放维护实验。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'final_source_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    manifests={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*')
               if p.is_file() and p.suffix not in {'.pt','.zip'} and p.name!='delivery_manifest.json'}
    (ROOT/'delivery_manifest.json').write_text(json.dumps(manifests,indent=2))


if __name__=='__main__':
    final,stages,router=aggregate()
    figures()
    report(final,stages,router)
    print(final.to_string(index=False))
