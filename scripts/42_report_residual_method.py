"""Export the five-method comparison and diagnostics from saved CSVs only."""
import _bootstrap
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.continual.residual_method import METHODS

ROOT=Path('outputs/residual_method')
LABELS=dict(zip(METHODS,['A Direct / Sequential FT','B Structural Replay','C Static Residual','D Always Expand Residual','E Dynamic Residual RAE']))


def aggregate():
    assert json.loads((ROOT/'audit.json').read_text())['status']=='passed'
    cfg=load_config('configs/experiment/residual_method.yaml');runs=Path(cfg['experiment']['run_root'])
    final=pd.concat([pd.read_csv(runs/m/'final_results.csv') for m in METHODS],ignore_index=True)
    stages=pd.concat([pd.read_csv(runs/m/'stage_metrics.csv') for m in METHODS],ignore_index=True)
    samples=pd.concat([pd.read_csv(runs/m/'sample_predictions.csv') for m in METHODS],ignore_index=True)
    gains=pd.concat([pd.read_csv(runs/m/'residual_gain.csv') for m in METHODS[1:]],ignore_index=True)
    direct=samples[samples.method==METHODS[0]]
    direct_gain=[]
    for (s,d),group in direct.groupby(['stage_id','domain_id']):
        direct_gain.append(dict(method=METHODS[0],stage_id=s,domain_id=d,split='test',expert_id='all',samples=len(group),
            base_mae=group.base_absolute_error.mean(),corrected_mae=group.final_absolute_error.mean(),residual_gain=0.,
            mean_abs_residual=0.,gate_mean=0.,beta=0.,expert_active=False,active_sample_count=0,novelty_mean=np.nan))
    gains=pd.concat([pd.DataFrame(direct_gain),gains],ignore_index=True)
    gradients=pd.concat([pd.read_csv(runs/m/'gradient_audit.csv') for m in METHODS[1:]],ignore_index=True)
    events=pd.concat([pd.read_csv(runs/m/'expert_history.csv') for m in METHODS[1:]],ignore_index=True)
    for name,table in [('final_results',final),('stage_metrics',stages),('sample_predictions',samples),
                       ('residual_gain',gains),('gradient_audit',gradients),('expert_history',events)]:
        table.to_csv(ROOT/(name+'.csv'),index=False)
    stages[['method','stage_id','optimizer_steps','new_optimizer_steps_this_task','current_exposures','replay_exposures',
            'total_parameters','trainable_parameters','wall_clock_seconds']].to_csv(ROOT/'compute_metrics.csv',index=False)
    matrix=[];activation=[]
    for m in METHODS:
        for stage in [1,2,3]:
            table=samples[(samples.method==m)&(samples.stage_id==stage)&(samples.split=='test')&(samples.evaluation_phase=='stage_complete')]
            table=table[table.domain_id.isin([2,1,0][:stage])]
            values=table.groupby('domain_id').final_absolute_error.mean()
            matrix.append(dict(method=m,stage_id=stage,**{'domain_'+str(int(d)):float(v) for d,v in values.items()}))
            active=table[table.expert_active]
            by_expert=[]
            for expert,group in active.groupby('expert_id'):
                by_expert.append(dict(expert_id=int(expert),n=len(group),base_mae=float(group.base_absolute_error.mean()),
                    corrected_mae=float(group.final_absolute_error.mean()),gain=float(group.base_absolute_error.mean()-group.final_absolute_error.mean())))
            activation.append(dict(method=m,stage_id=stage,active_test_samples=len(active),
                active_test_expert_ids=json.dumps(sorted(map(int,active.expert_id.unique()))),
                active_expert_test_metrics=json.dumps(by_expert),
                active_test_base_mae=float(active.base_absolute_error.mean()) if len(active) else None,
                active_test_corrected_mae=float(active.final_absolute_error.mean()) if len(active) else None))
    pd.DataFrame(matrix).to_csv(ROOT/'continual_matrix.csv',index=False)
    pd.DataFrame(activation).to_csv(ROOT/'activation_summary.csv',index=False)
    shutil.copy2(runs/METHODS[4]/'config.yaml',ROOT/'resolved_config.yaml')
    return final,stages,events


def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cross=pd.read_csv(ROOT/'direct_continual_matrix.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(6.6,4.3));image=ax.imshow(cross.to_numpy(),cmap='YlOrRd',vmin=0)
    ax.set(xticks=[0,1,2],xticklabels=['D1','D2','D3'],yticks=[0,1,2],yticklabels=cross.index,
           xlabel='Test structural domain',ylabel='Sequential training state',title='Direct Predictor: structural-domain generalization')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{cross.iloc[i,j]:.4f}',ha='center',va='center',color='black',fontsize=12)
    fig.colorbar(image,ax=ax,label='MAE (eV/atom)');fig.tight_layout();fig.savefig(ROOT/'direct_cross_domain.png',dpi=160);plt.close(fig)
    stages=pd.read_csv(ROOT/'stage_metrics.csv');colors=['#777777','#426F92','#BF994F','#8676A3','#688B68']
    fig,ax=plt.subplots(figsize=(9,4.4))
    for m,color in zip(METHODS,colors):
        group=stages[stages.method==m]
        ax.plot(group.stage_id,group.average_seen_mae,marker='o',color=color,label=LABELS[m],linewidth=2 if m==METHODS[4] else 1.6)
    ax.set(xlabel='Continual stage (domain order: 2, 1, 0)',ylabel='Average seen-domain MAE (eV/atom)',xticks=[1,2,3],title='Fixed 5k comparison | seed 42')
    ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False)
    fig.savefig(ROOT/'average_seen_mae.png',dpi=160,bbox_inches='tight');plt.close(fig)
    final=pd.read_csv(ROOT/'final_results.csv').set_index('method')
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    for i,m in enumerate(METHODS[1:]):
        row=final.loc[m]
        ax.plot([row.base_final_avg_seen_mae,row.final_avg_seen_mae],[i,i],color=colors[i+1],linewidth=2)
        ax.scatter(row.base_final_avg_seen_mae,i,s=65,facecolors='white',edgecolors=colors[i+1],zorder=3)
        ax.scatter(row.final_avg_seen_mae,i,s=45,marker='s',color=colors[i+1],zorder=4)
    ax.set_yticks(range(4),[LABELS[m] for m in METHODS[1:]])
    ax.set(xlabel='Final average seen-domain MAE (eV/atom)',title='Each model: base prediction vs corrected prediction')
    ax.invert_yaxis();ax.grid(alpha=.2)
    fig.text(.12,-.02,'Open circle: its shared base. Filled square: its final corrected output.',fontsize=9)
    fig.savefig(ROOT/'base_vs_corrected.png',dpi=160,bbox_inches='tight');plt.close(fig)


def report(final,stages,events):
    rows=final.set_index('method');a,b,c,d,e=[rows.loc[m] for m in METHODS]
    cross=pd.read_csv(ROOT/'direct_cross_domain.csv').set_index('test_domain')
    own=float(cross.loc['D1','mae']);shift2=float(cross.loc['D2','mae']);shift3=float(cross.loc['D3','mae'])
    activation=pd.read_csv(ROOT/'activation_summary.csv');active_e=activation[(activation.method==METHODS[4])&(activation.stage_id==3)].iloc[0]
    experts=json.loads(active_e.active_expert_test_metrics)
    active_d=activation[(activation.method==METHODS[3])&(activation.stage_id==3)].iloc[0]
    d_experts=json.loads(active_d.active_expert_test_metrics)
    gain_e=pd.read_csv(ROOT/'residual_gain.csv')
    active_groups=gain_e[(gain_e.method==METHODS[4])&(gain_e.stage_id==3)&(gain_e.split=='test')&(gain_e.expert_id!='all')&(gain_e.expert_active)]
    lines=['# 共享预测器 + 动态残差：5k 诊断报告','',
        '五种指定方法已完成，训练已停止。固定5000条结构、3个合并晶系域、seed42、顺序[2,1,0]、原3999/498/503训练/验证/测试划分。没有30k、全量、额外种子或阈值搜索。','',
        '| 方法 | S1 MAE | S2 当前域 | S2 平均 | 最终当前域 | 最终平均 | 遗忘 | 残差数/激活数 | 记忆 | 优化步 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,r in rows.iterrows():
        lines.append(f'| {LABELS[m]} | {r.S1_mae:.6f} | {r.S2_current_mae:.6f} | {r.S2_avg_seen_mae:.6f} | {r.final_current_mae:.6f} | {r.final_avg_seen_mae:.6f} | {r.forgetting:.6f} | {int(r.num_residual_experts)}/{int(r.num_active_experts)} | {int(r.memory_size)} | {int(r.optimizer_steps)} |')
    lines += ['','A复用既有Sequential Fine-Tuning的三个检查点；810是其历史训练步数，本任务新增A训练步数为0。B-E均重新执行810次共享预测器更新，D1使用共同的已训练权重。',
        'A保留原验证集选择检查点；B-E为固定10遍、最终步检查点。B-E最终结果使用统一确定性计算；历史非确定性0.097065基线仅作单独参考。','',
        '## 七个问题','',
        f'Q1：D1训练后的MAE为D1={own:.6f}，未见D2={shift2:.6f}（+{(shift2/own-1)*100:.2f}%），未见D3={shift3:.6f}（+{(shift3/own-1)*100:.2f}%）。这支持结构域分布偏移下的泛化退化；成分与目标分布也可能变化，不能声称结构本身具有纯因果作用。','',
        f'Q2：结构回放B={b.final_avg_seen_mae:.6f}，Direct A={a.final_avg_seen_mae:.6f}；相对MAE变化{(b.final_avg_seen_mae/a.final_avg_seen_mae-1)*100:+.2f}%。'+('本次结构回放更低。' if b.final_avg_seen_mae<a.final_avg_seen_mae else '本次结构回放未改善Direct。'),' ',
        f'Q3：静态残差C={c.final_avg_seen_mae:.6f}，回放B={b.final_avg_seen_mae:.6f}，差值{c.final_avg_seen_mae-b.final_avg_seen_mae:+.6f}。'+('本次数值有所改善。' if c.final_avg_seen_mae<b.final_avg_seen_mae else '本次静态残差未改善回放基线。')+' 单种子不能证明稳健优势。','',
        f'Q4：动态残差E={e.final_avg_seen_mae:.6f}，静态C={c.final_avg_seen_mae:.6f}，差值{e.final_avg_seen_mae-c.final_avg_seen_mae:+.6f}。'+('本次数值更低，动态容量的稳健收益仍未建立。' if e.final_avg_seen_mae<c.final_avg_seen_mae else '本次不支持宣称动态容量优于静态容量。')+(' E最终没有激活残差，其差异来自训练后的共享预测器，不能归因为推理时动态残差容量的收益。' if int(e.num_active_experts)==0 else ''),' ',
        f'Q5：主模型累计{int(e.num_residual_experts)}个残差专家，最终有{int(e.num_active_experts)}个激活开关；实际对最终测试样本提供修正的专家ID为{active_e.active_test_expert_ids}，覆盖{int(active_e.active_test_samples)}个测试样本。','',
        'Q6：有当前验证覆盖的激活专家都通过Gain>0检查；测试集不保证同样获益。'+
        (f'最终活跃“专家-测试域”组合中，{int((active_groups.residual_gain>0).sum())}/{len(active_groups)}组的修正MAE低于自身base。' if len(active_groups) else '最终没有活跃残差测试组，因此该测试收益问题不适用。'),'']
    for x in experts:
        lines.append(f"- 专家{x['expert_id']}：测试样本{x['n']}；base MAE={x['base_mae']:.6f}，corrected MAE={x['corrected_mae']:.6f}，gain={x['gain']:+.6f}。")
    for x in d_experts:
        lines.append(f"补充D始终扩展对照：激活专家{x['expert_id']}覆盖{x['n']}个测试样本；base MAE={x['base_mae']:.9f}，corrected MAE={x['corrected_mae']:.9f}，gain={x['gain']:+.9f}。该微小差异不足以支持实质修正收益。")
    lines += ['',f'Q7：主模型自身共享路径的最终平均MAE={e.base_final_avg_seen_mae:.6f}，校正后={e.final_avg_seen_mae:.6f}；本次同计算协议的回放B={b.final_avg_seen_mae:.6f}。共享路径相对B变化{(e.base_final_avg_seen_mae/b.final_avg_seen_mae-1)*100:+.2f}%，最终输出相对历史0.097065变化{(e.final_avg_seen_mae/0.09706452075366795-1)*100:+.2f}%。关闭残差可以逐位回退到自身base，但这不等于训练权重或性能必然等同历史基线。','',
        '## 基础路径与残差贡献','', '| 方法 | 自身base平均MAE | 最终平均MAE | 输出修正收益 |', '|---|---:|---:|---:|']
    for m,r in rows.iterrows():lines.append(f'| {LABELS[m]} | {r.base_final_avg_seen_mae:.9f} | {r.final_avg_seen_mae:.9f} | {r.base_final_avg_seen_mae-r.final_avg_seen_mae:+.9f} |')
    lines += ['','当前样本的校正损失端到端更新共享路径，因此C/D/E与B的差值不全是推理时加上残差的收益。上表单独给出同一模型的基础预测与校正后预测，避免混淆。','',
        '## 控制器和验证安全性','']
    for m in METHODS[2:]:
        for event in events[(events.method==m)&(events.stage_id>1)].to_dict('records'):
            lines.append(f"- {m} / S{event['stage_id']}：{event['decision']}；新专家={event['new_expert']}；原因={event['expansion_reason']}；最终验证base={event['final_base_val_mae']:.6f}，corrected={event['final_corrected_val_mae']:.6f}。")
    lines += ['','RAE只控制残差容量。包括REUSE在内，共享完整ALIGNN始终进行当前+回放训练；REUSE默认当前优化走base，不训练残差。回放样本按自身最近原型选择已验证历史残差，残差分支不接收回放梯度；共享骨干与共享头都有非零回放梯度。',
        '每阶段末在当前验证集上按真实原型路由分别检查各残差；Gain>0启用，否则关闭。无当前验证覆盖的区域保留过去的验证开关。当前验证安全性不意味着旧域或测试集一定获益；测试结果从不参与激活。',
        '', '## 配置与数值复现问题','',
        '完整配置：resolved_config.yaml。全共享骨干/头lr=1e-3，残差lr=1e-3；rank16、GELU、dropout0.1、稳定编码器冻结；shift=LayerNorm(W(h_s−μ_nearest))；最终预测=base+tanh(β)×gate×scalar_residual。β创建/继承时归零，末层不同时归零。',
        '记忆2000、候选6000、原D1结构向量与原域均衡k-center。保持原80/20记忆划分，20%预留包含在预算内、不进入梯度，也不参与本任务的残差激活。每次32当前+32回放，单次Huber_current+Huber_replay反向与优化；10遍中的前5遍共享适应、后5遍可训练残差。',
        '复现问题已保留：首次非确定性关闭残差B=0.138998，与历史0.097065不一致，尽管初始权重、完整批次索引与优化器配置一致；首步损失约1e-9差异被训练放大。随后两个真实批次的原生/封装对照，在确定性计算下更新权重逐位相同。最终B-E统一使用torch.use_deterministic_algorithms(True)和CUBLAS_WORKSPACE_CONFIG=:4096:8，仅运行一次；未搜索学习率、阈值或测试集参数。',
        '数值协议变更由已观察到的复现偏差触发，因此最终比较属于探索性诊断，不能把新旧数字差异当作算法提升。旧B尝试及中断C日志/检查点保留于outputs/residual_method/runs/；最终有效比较使用outputs/residual_method/verified/runs/。',
        '工程梯度检查包括最初2个一次性更新，以及确定性检查中的3个封装更新+2个原生参照更新；这些权重未用于最终训练。表格的810步只指各最终共享训练运行，不包含工程排障成本。',
        '', '## 交付与审计','',
        '新增源码：src/models/sp_crystal_residual.py；src/continual/residual_method.py、residual_diagnostics.py；configs/experiment/residual_method.yaml；tests/test_residual_method.py；scripts/40_residual_method.py、41_audit_residual_method.py、42_report_residual_method.py；RESIDUAL_METHOD_IMPLEMENTATION.md。旧项目没有重命名或替换。',
        '63项测试通过；真实梯度检查、原生/关闭残差训练等价检查通过；最终检查点重新评估、固定503测试样本、验证激活开关、回放完整索引、稳定编码器和原1101个受保护文件哈希通过。',
        '必需CSV均在本目录：final_results.csv、direct_cross_domain.csv、direct_cross_domain_mae.csv、direct_continual_matrix.csv、residual_gain.csv、sample_predictions.csv、gradient_audit.csv。补充文件包括stage_metrics.csv、continual_matrix.csv、expert_history.csv、activation_summary.csv、compute_metrics.csv、leakage_audit.txt和numerical_reproducibility.json。',
        '远程完整检查点、记忆、逐步索引位于verified/runs/。下载包包含源码、CSV、日志与审计；大型权重保留在远程。所有训练已停止，不追加其他实验。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'final_source_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    print(final.to_string(index=False))


if __name__=='__main__':
    final,stages,events=aggregate();figures();report(final,stages,events)
