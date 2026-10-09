"""CSV-driven diagnostics, six vector PDF figures and the SGCR result report."""
import _bootstrap
import json
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.config import load_config,save_config
from src.utils.logging import save_json
from src.continual.sgcr_trainer import METHODS

ROOT=Path('outputs/sgcr')
LABELS=dict(zip(METHODS,['A Direct','B Structural Replay','C + Shift Retrieval','D + Conflict Retrieval','E Full SGCR']))
COLORS=['#777777','#426F92','#BF994F','#8676A3','#688B68']


def direct(cfg):
    source=Path(cfg['sgcr']['direct_reference']);out=Path(cfg['experiment']['run_root'])/METHODS[0]
    out.mkdir(exist_ok=True)
    archived_final=pd.read_csv(source/'final_results.csv').iloc[0]
    old=pd.read_csv(source/'sample_predictions.csv')
    old=old[(old.split=='test')&(old.evaluation_phase=='stage_complete')].copy()
    old=old[old.apply(lambda r:r.domain_id in [2,1,0][:int(r.stage_id)],axis=1)]
    old=old.rename(columns={'final_prediction':'prediction','final_absolute_error':'absolute_error'})
    old['method']=METHODS[0];old['stable_novelty']=np.nan;old['nearest_prototype']=-1
    samples=old[['method','material_id','stage_id','domain_id','target','prediction','absolute_error','stable_novelty','nearest_prototype','split']]
    stages=pd.read_csv(source/'stage_metrics.csv')
    stages['method']=METHODS[0];stages['peak_gpu_memory_mb']=np.nan;stages['autograd_calls']=0
    stages['per_domain_mae']=[json.dumps({str(int(d)):float(g.absolute_error.mean()) for d,g in samples[samples.stage_id==s].groupby('domain_id')}) for s in stages.stage_id]
    stages=stages[['method','stage_id','domain_id','current_domain_mae','average_seen_mae','per_domain_mae','forgetting','memory_size',
                   'optimizer_steps','new_optimizer_steps_this_task','wall_clock_seconds','peak_gpu_memory_mb','current_exposures','replay_exposures','autograd_calls']]
    final=dict(method=METHODS[0],S1_avg_mae=float(stages.iloc[0].average_seen_mae),S2_avg_mae=float(stages.iloc[1].average_seen_mae),
        final_avg_mae=float(stages.iloc[2].average_seen_mae),forgetting=float(stages.iloc[2].forgetting),memory_size=0,
        optimizer_steps=int(stages.optimizer_steps.sum()),new_optimizer_steps_this_task=0,wall_clock_seconds=float(stages.wall_clock_seconds.sum()),
        peak_gpu_memory_mb=np.nan,extra_inference_samples=0,extra_inference_seconds=0.,
        total_parameters=int(archived_final.total_parameters),trainable_parameters=int(archived_final.trainable_parameters))
    pd.DataFrame([final]).to_csv(out/'final_results.csv',index=False)
    samples.to_csv(out/'sample_predictions.csv',index=False);stages.to_csv(out/'stage_metrics.csv',index=False)
    save_json(out/'provenance.json',dict(source=str(source),reused_checkpoints=True,new_training_steps=0,peak_memory_unavailable=True,
                                      checkpoint_policy='original validation-selected historical Direct checkpoints'))


def aggregate(cfg):
    assert json.loads((ROOT/'audit.json').read_text())['status']=='passed'
    direct(cfg);runs=Path(cfg['experiment']['run_root'])
    for name,methods in [('final_results',METHODS),('stage_metrics',METHODS),('sample_predictions',METHODS),
                         ('batch_diagnostics',METHODS[1:]),('replay_selection',METHODS[2:]),('retrieval_quality_batches',METHODS[3:])]:
        frames=[pd.read_csv(runs/m/(name+'.csv')) for m in methods]
        pd.concat(frames,ignore_index=True).to_csv(ROOT/(name+'.csv'),index=False)
    stages=pd.read_csv(ROOT/'stage_metrics.csv')
    stages[['method','stage_id','forgetting']].to_csv(ROOT/'forgetting.csv',index=False)
    matrix=[]
    for r in stages.itertuples():
        matrix.append(dict(method=r.method,stage_id=r.stage_id,**{'domain_'+k:v for k,v in json.loads(r.per_domain_mae).items()}))
    pd.DataFrame(matrix).to_csv(ROOT/'continual_matrix.csv',index=False)
    save_config(ROOT/'resolved_config.yaml',cfg)


def analyses():
    diagnostics=pd.read_csv(ROOT/'batch_diagnostics.csv');quality=pd.read_csv(ROOT/'retrieval_quality_batches.csv')
    correlations=[]
    for method,group in diagnostics.groupby('method',sort=False):
        for stage,part in [('all',group)]+[(str(s),g) for s,g in group.groupby('stage_id')]:
            for negative_only in [False,True]:
                subset=part[part.gradient_cosine_before<0] if negative_only else part
                subset=subset[['novelty','gradient_cosine_before']].dropna()
                rho=p_value=np.nan
                if len(subset)>=3 and subset.novelty.nunique()>1 and subset.gradient_cosine_before.nunique()>1:
                    rho,p_value=spearmanr(subset.novelty,-subset.gradient_cosine_before)
                correlations.append(dict(method=method,stage_id=stage,negative_only=negative_only,n=len(subset),spearman_rho=rho,
                    nominal_p_value=p_value,interpretation='descriptive correlated training batches; p-value assumes independence and is not confirmatory'))
    pd.DataFrame(correlations).to_csv(ROOT/'novelty_gradient_conflict.csv',index=False)
    proxy=[]
    for method in METHODS[3:]:
        part=diagnostics[diagnostics.method==method]
        for stage,group in [('all',part)]+[(str(s),g) for s,g in part.groupby('stage_id')]:
            group=group[['mean_selected_conflict_score','gradient_cosine_before']].dropna()
            coefficient=np.nan
            if len(group)>=3 and group.mean_selected_conflict_score.nunique()>1 and group.gradient_cosine_before.nunique()>1:
                coefficient=float(spearmanr(group.mean_selected_conflict_score,-group.gradient_cosine_before).statistic)
            proxy.append(dict(method=method,stage_id=stage,n=len(group),proxy_vs_exact_conflict_spearman=coefficient))
    pd.DataFrame(proxy).to_csv(ROOT/'proxy_gradient_conflict.csv',index=False)
    comparisons=[]
    for (method,stage),part in quality.groupby(['method','stage_id'],sort=False):
        values={name:float(np.average(part[name],weights=part['count'])) for name in
                ['selected_conflict','random_conflict','selected_structural_relevance','random_structural_relevance']}
        comparisons.append(dict(method=method,stage_id=stage,batches=len(part),sample_exposures=int(part['count'].sum()),**values,
            conflict_difference=values['selected_conflict']-values['random_conflict'],
            relevance_difference=values['selected_structural_relevance']-values['random_structural_relevance']))
    pd.DataFrame(comparisons).to_csv(ROOT/'retrieval_quality.csv',index=False)
    projection=[]
    for method,group in diagnostics.groupby('method',sort=False):
        for stage,part in [('all',group)]+[(str(s),g) for s,g in group.groupby('stage_id')]:
            active=part[part.projection_applied]
            projection.append(dict(method=method,stage_id=stage,batches=len(part),conflicting_batches=int((part.gradient_cosine_before<0).sum()),
                projection_count=len(active),projection_rate=len(active)/len(part),mean_rho=float(part.rho.mean()),
                mean_rho_when_projected=float(active.rho.mean()) if len(active) else np.nan,
                rho_zero_fraction=float((part.rho==0).mean()),rho_one_fraction=float((part.rho==1).mean()),
                cosine_before_projected=float(active.gradient_cosine_before.mean()) if len(active) else np.nan,
                cosine_after_projected=float(active.gradient_cosine_after.mean()) if len(active) else np.nan))
    pd.DataFrame(projection).to_csv(ROOT/'projection_summary.csv',index=False)
    stages=pd.read_csv(ROOT/'stage_metrics.csv');history=[]
    for method in METHODS:
        group=stages[stages.method==method].set_index('stage_id')
        for s in [2,3]:
            before=json.loads(group.loc[s-1,'per_domain_mae']);after=json.loads(group.loc[s,'per_domain_mae'])
            for domain in before:
                history.append(dict(method=method,stage_id=s,historical_domain_id=int(domain),before_mae=before[domain],after_mae=after[domain],
                                    mae_change=after[domain]-before[domain]))
    pd.DataFrame(history).to_csv(ROOT/'historical_domain_changes.csv',index=False)


def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    stages=pd.read_csv(ROOT/'stage_metrics.csv')
    def save(fig,name):
        fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(ROOT/(name+'.png'),bbox_inches='tight');plt.close(fig)
    for column,name,title,ylabel in [('average_seen_mae','average_seen_mae_vs_stage','Fixed 5k comparison | seed 42','Average seen-domain MAE (eV/atom)'),
        ('forgetting','forgetting_vs_stage','Forgetting relative to best observed historical-domain MAE','Forgetting (eV/atom)')]:
        fig,ax=plt.subplots(figsize=(7.6,4.6))
        for i,method in enumerate(METHODS):
            part=stages[stages.method==method]
            ax.plot(part.stage_id,part[column],marker=['o','s','^','D','o'][i],color=COLORS[i],label=LABELS[method],
                    linewidth=2 if i==4 else 1.6,markersize=5)
        ax.set(xlabel='Continual stage (domain order: 2, 1, 0)',ylabel=ylabel,xticks=[1,2,3],title=title)
        ax.grid(alpha=.2);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False)
        save(fig,name)
    diag=pd.read_csv(ROOT/'batch_diagnostics.csv');full=diag[diag.method==METHODS[4]]
    correlation=pd.read_csv(ROOT/'novelty_gradient_conflict.csv')
    value=correlation[(correlation.method==METHODS[4])&(correlation.stage_id=='all')&(~correlation.negative_only)].iloc[0]
    fig,ax=plt.subplots(figsize=(7,4.7))
    for s,color in [(2,'#426F92'),(3,'#BF994F')]:
        part=full[full.stage_id==s]
        ax.scatter(part.novelty,-part.gradient_cosine_before,s=12,alpha=.48,color=color,label=f'Stage {s} (n={len(part)})',rasterized=False)
    ax.axhline(0,color='#AAAAAA',linewidth=.8);ax.grid(alpha=.15)
    ax.set(xlabel='Stable batch novelty',ylabel='Negative current/replay gradient cosine',
           title=f'Full SGCR: novelty and conflict | Spearman r={value.spearman_rho:.3f}')
    ax.legend(frameon=False,loc='best');save(fig,'novelty_vs_gradient_conflict')
    quality=pd.read_csv(ROOT/'retrieval_quality.csv');quality=quality[quality.method==METHODS[4]].sort_values('stage_id')
    for selected,random,name,title,ylabel in [
        ('selected_conflict','random_conflict','selected_vs_random_conflict','Replay selection: conflict score','Mean raw conflict score, ReLU(-cosine)'),
        ('selected_structural_relevance','random_structural_relevance','replay_structural_relevance','Replay selection: structural relevance','Mean raw relevance, exp(-distance / tau)')]:
        fig,ax=plt.subplots(figsize=(6.6,4.4));x=np.arange(len(quality));w=.34
        bars1=ax.bar(x-w/2,quality[selected],w,label='SGCR selected',color=COLORS[4])
        bars2=ax.bar(x+w/2,quality[random],w,label='Uniform reference',color='#AAB4BE')
        ax.bar_label(bars1,fmt='%.3f',padding=3,fontsize=9);ax.bar_label(bars2,fmt='%.3f',padding=3,fontsize=9)
        ymax=max(float(quality[selected].max()),float(quality[random].max()))
        ax.set(xticks=x,xticklabels=[f'Stage {s}' for s in quality.stage_id],ylabel=ylabel,title=title,ylim=(0,max(.1,ymax*1.3)))
        ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15);save(fig,name)
    fig,ax=plt.subplots(figsize=(6.3,5.0))
    for active,color,label in [(False,'#AAB4BE','No projection'),(True,COLORS[4],'Projection applied')]:
        part=full[full.projection_applied==active]
        ax.scatter(part.gradient_cosine_before,part.gradient_cosine_after,s=15,alpha=.55,color=color,label=f'{label} (n={len(part)})')
    ax.plot([-1,1],[-1,1],ls='--',color='#888888',lw=1);ax.axhline(0,color='#AAAAAA',lw=.7);ax.axvline(0,color='#AAAAAA',lw=.7)
    ax.set(xlabel='Current/replay gradient cosine before correction',ylabel='Current/replay gradient cosine after correction',
           xlim=(-1.03,1.03),ylim=(-1.03,1.03),title='Full SGCR: effect of gradient correction')
    ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.15);save(fig,'gradient_cosine_before_after')


def report():
    final=pd.read_csv(ROOT/'final_results.csv').set_index('method');stages=pd.read_csv(ROOT/'stage_metrics.csv')
    b,c,d,e=[final.loc[m] for m in METHODS[1:]]
    correlations=pd.read_csv(ROOT/'novelty_gradient_conflict.csv')
    corr=correlations[(correlations.method==METHODS[4])&(correlations.stage_id=='all')]
    all_corr=corr[~corr.negative_only].iloc[0];negative=corr[corr.negative_only].iloc[0]
    projections=pd.read_csv(ROOT/'projection_summary.csv')
    full_projection=projections[(projections.method==METHODS[4])&(projections.stage_id=='all')].iloc[0]
    projected_stages=projections[(projections.method==METHODS[4])&(projections.stage_id!='all')].set_index('stage_id')
    quality=pd.read_csv(ROOT/'retrieval_quality.csv');eq=quality[quality.method==METHODS[4]]
    proxy=pd.read_csv(ROOT/'proxy_gradient_conflict.csv')
    ep=proxy[(proxy.method==METHODS[4])&(proxy.stage_id=='all')].iloc[0]
    reproduction=json.loads((ROOT/'baseline_reproduction.json').read_text())
    old_b=pd.read_csv(Path(reproduction['reference'])/'final_results.csv').iloc[0]
    lines=['# SGCR：5k 固定划分诊断报告','',
      '实现、必需A-E实验、审计及六张图已完成，所有训练停止。固定5000条晶体、seed42、域顺序[2,1,0]、原3999/498/503训练/验证/测试划分。没有30k、全量、多种子、阈值搜索或可选F实验。','',
      ('本轮SGCR没有保留强结构回放的预测质量；默认检索与投影组合的性能目标失败。' if e.final_avg_mae>b.final_avg_mae else '本轮SGCR的数值表现达到强回放基线水平，仍属于单种子诊断。'),' ',
      '## 最终结果','', '| 方法 | S1平均MAE | S2平均MAE | 最终平均MAE | 遗忘 | 记忆 | 优化步 | 耗时秒 | 峰值GPU MiB |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,r in final.iterrows():
        peak='未记录' if pd.isna(r.peak_gpu_memory_mb) else f'{r.peak_gpu_memory_mb:.1f}'
        lines.append(f'| {LABELS[m]} | {r.S1_avg_mae:.6f} | {r.S2_avg_mae:.6f} | {r.final_avg_mae:.6f} | {r.forgetting:.6f} | {int(r.memory_size)} | {int(r.optimizer_steps)} | {r.wall_clock_seconds:.1f} | {peak} |')
    lines+=['','MAE单位eV/atom，平均指标对已见结构域等权。A复用原顺序微调检查点，810步和时间均是历史成本，本任务新增A更新为0；其峰值显存未记录，保留为空。B-E均从共同D1权重开始，新执行810次更新，最后一步保存，无测试选择。A保留原验证集选模，因此A与B-E不是完全相同的检查点选择协议。','',
      f'基线复现：B={reproduction["reproduced_mae"]:.9f}，原B={reproduction["reference_mae"]:.9f}；三个阶段预测最大差异={reproduction["prediction_max_differences"]}，非逐位一致的预测器张量数={reproduction["non_bitwise_predictor_tensor_counts"]}。C-E仅在该复现通过后启动。','',
      '## 十个问题','',
      f'Q1：结构检索C相对B的最终MAE变化为{c.final_avg_mae-b.final_avg_mae:+.6f}。'+('本次改善。' if c.final_avg_mae<b.final_avg_mae else '本次未改善。'),' ',
      f'Q2：冲突检索D相对结构检索C变化为{d.final_avg_mae-c.final_avg_mae:+.6f}。'+('本次改善。' if d.final_avg_mae<c.final_avg_mae else '本次未改善。'),' ',
      f'Q3：完整SGCR E相对D变化为{e.final_avg_mae-d.final_avg_mae:+.6f}。'+('加入梯度稳定后本次MAE更低。' if e.final_avg_mae<d.final_avg_mae else '加入梯度稳定后本次MAE没有改善。'),' ',
      f'Q4：最终遗忘B={b.forgetting:.6f}，D={d.forgetting:.6f}，E={e.forgetting:.6f}。'+('E相对D减轻遗忘。' if e.forgetting<d.forgetting else 'E未比D进一步降低该遗忘指标。')+(' B已经为0，无法再在这一非负指标上降低。' if b.forgetting==0 else ''),' ',
      'Q5：以下使用相同数量、相同当前模型下的历史均匀样本作只读对照，分数为可比的原始ReLU(-cosine)，没有分别归一化后再比较。','',
      '| 阶段 | 已选冲突 | 随机冲突 | 差值 | 已选结构相关性 | 随机结构相关性 |', '|---|---:|---:|---:|---:|---:|']
    for r in eq.itertuples():lines.append(f'| S{r.stage_id} | {r.selected_conflict:.6f} | {r.random_conflict:.6f} | {r.conflict_difference:+.6f} | {r.selected_structural_relevance:.6f} | {r.random_structural_relevance:.6f} |')
    lines+=['',f'Q6：完整SGCR中，novelty与负梯度余弦的Spearman相关为{all_corr.spearman_rho:.6f}（n={int(all_corr.n)}）；仅负梯度余弦批次为{negative.spearman_rho:.6f}（n={int(negative.n)}）。'+('合并轨迹呈正相关，但不能直接推出因果或跨域稳健关系。' if all_corr.spearman_rho>0 else '本次合并轨迹不支持更大novelty对应更强冲突这一正相关假设。'),' ',
      f'Q7：梯度修正触发{int(full_projection.projection_count)}/{int(full_projection.batches)}次，比例{full_projection.projection_rate*100:.2f}%。',' ',
      f'Q8：触发时平均rho={full_projection.mean_rho_when_projected:.6f}；所有批次rho=0占{full_projection.rho_zero_fraction*100:.2f}%，rho=1占{full_projection.rho_one_fraction*100:.2f}%。若rho大量饱和，当前诊断接近固定强度投影，不能仅凭名称证明细粒度结构调节有效。',' ',
      f'Q9：SGCR耗时{e.wall_clock_seconds:.1f}秒，同批梯度诊断的B耗时{b.wall_clock_seconds:.1f}秒，{e.wall_clock_seconds/b.wall_clock_seconds:.2f}倍（+{(e.wall_clock_seconds/b.wall_clock_seconds-1)*100:.1f}%）。原未逐批增加精确梯度诊断的B耗时{old_b.wall_clock_seconds:.1f}秒，供单独参考。',' ',
      f'Q10：SGCR最终MAE={e.final_avg_mae:.6f}，强回放B={b.final_avg_mae:.6f}，相对变化{(e.final_avg_mae/b.final_avg_mae-1)*100:+.2f}%。'+('本次超过基线，仍需把它视为单种子诊断而非稳健优势。' if e.final_avg_mae<b.final_avg_mae else '本次未能保留强回放基线的质量。')+' 不追加模块、不搜索参数、不扩大实验。','',
      '## 各阶段遗忘','', '| 方法 | S1 | S2 | S3 |','|---|---:|---:|---:|']
    for m in METHODS:
        group=stages[stages.method==m].sort_values('stage_id')
        lines.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.6f}' for v in group.forgetting)+' |')
    lines+=['','遗忘沿用原定义：历史域当前MAE减去引入该域以来的最小MAE，再对历史域平均；最小值包含当前阶段，因此下界为0。historical_domain_changes.csv另列学习新域前后的原始MAE变化，避免零遗忘掩盖改进或不同退化方向。','',
      '## 分阶段 novelty 与冲突','', '| 方法 | 阶段 | 全部批次相关 | 负余弦批次相关 |', '|---|---|---:|---:|']
    for m in METHODS[1:]:
        for s in ['2','3']:
            group=correlations[(correlations.method==m)&(correlations.stage_id==s)]
            x=group[~group.negative_only].iloc[0];y=group[group.negative_only].iloc[0]
            lines.append(f'| {LABELS[m]} | S{s} | {x.spearman_rho:.6f} | {y.spearman_rho:.6f} |')
    lines+=['','训练批次并不独立：它们共享记忆并沿着同一参数轨迹演化。相关系数仅用于描述；CSV中的名义p值依赖独立性假设，不作为显著性或因果证据。合并系数也可能由阶段差异驱动，应同时看分阶段结果。','',
      '## 分阶段投影行为','', '| 阶段 | 负余弦批次 | 投影批次 | 总批次 | 平均rho | rho=0比例 | rho=1比例 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in projections[(projections.method==METHODS[4])&(projections.stage_id!='all')].itertuples():
        lines.append(f'| S{r.stage_id} | {r.conflicting_batches} | {r.projection_count} | {r.batches} | {r.mean_rho:.6f} | {r.rho_zero_fraction*100:.2f}% | {r.rho_one_fraction*100:.2f}% |')
    gate_text='；'.join(f"S{s}平均rho={r.mean_rho:.6f}，{int(r.conflicting_batches)}个负余弦批次中投影{int(r.projection_count)}次" for s,r in projected_stages.iterrows())+'。'
    if any((projected_stages.rho_zero_fraction>.8)&(projected_stages.conflicting_batches>projected_stages.projection_count)):
        gate_text+='低结构novelty并不代表没有优化冲突；部分阶段的校准饱和使大量冲突没有触发投影。'
    weaker_relevance=eq[eq.relevance_difference<0].stage_id.tolist()
    relevance_text=('；'+','.join('S'+str(s) for s in weaker_relevance)+'的已选平均结构相关性低于均匀对照。') if weaker_relevance else '。'
    lines+=['',gate_text,'',
      '## 机制诊断','',
      f'结构检索造成的MAE变化为{c.final_avg_mae-b.final_avg_mae:+.6f}，冲突优先级随后带来{d.final_avg_mae-c.final_avg_mae:+.6f}，梯度修正再带来{e.final_avg_mae-d.final_avg_mae:+.6f}。这些对照定位组件在本次运行中的贡献，不能自动外推到其他种子或数据规模。',
      f'完整SGCR中，已选样本的平均头部冲突代理与实际子集负梯度余弦的Spearman相关为{ep.proxy_vs_exact_conflict_spearman:.6f}。代理分数较高不等于实际联合梯度必然更冲突，也不保证更好的最终MAE；分阶段结果保存在proxy_gradient_conflict.csv。',
      ('各阶段已选样本的冲突分数均高于均匀对照' if (eq.conflict_difference>0).all() else '已选与均匀对照的冲突分数差异见上表')+relevance_text+' 局部候选中的优先级最高不保证全局平均相关性更高，也不保证更好代表历史测试分布。',
      '若E未改善D，说明本次默认投影策略没有带来MAE收益，但不能仅凭一次消融把原因唯一归为投影过强；应结合rho饱和、实际触发率、代理与精确梯度的一致性，以及分阶段novelty相关性判断。','',
      '## 实现约定与问题','',
      '- 存储保持原2000预算、6000候选、域均衡k-center和80/20梯度/预留划分。原型层仅在已有历史训练记忆上聚类，K=8；预留样本可以参加训练记忆几何统计，但不进入检索候选、签名或优化。',
      '- 当前查询是冻结D1向量的批均值；q50/q90来自256个固定历史记忆批均值。tau来自512对历史记忆样本。所有选择在实验前固定，没有验证/测试阈值调参。',
      '- 检索为最近两个非空原型桶，最多128局部+64全局候选，75%局部优先+25%全局桶均衡；缺额时转移配额，始终与当前批等量且无重复。最后短批沿用32样本的历史校准。',
      '- 冲突签名使用精确Huber导数乘归一化预测特征，属于廉价头部梯度代理，不等同整个骨干的逐样本梯度。评分在当前模型eval/no_grad下计算，训练模式随后恢复。',
      '- 参数修正范围是实际gcn_layers.3与fc；所有B-E逐批测量两项精确梯度。E只在dot<0、rho>0且梯度范数足够时覆写子集梯度。非修正分支保留原联合backward；全局裁剪5在修正后执行，每批只有一个AdamW step。',
      '- 梯度余弦前后指当前梯度与历史梯度的夹角；修正后还要加上回放梯度，不应把该图误读为最终联合更新方向。',
      '- 该约束只作用于所选回放样本和参数子集的原始梯度。AdamW的自适应预条件、其他参数的更新、有限步长，以及训练样本与历史测试分布的差异，都使它无法保证历史测试MAE不增加。',
      '- 耗时包括原型、验证/记忆处理、候选推理、均匀对照推理和逐批精确梯度诊断，不包括最终测试推理和结果序列化。均匀对照不影响训练，但消耗时间，因此这是一套带诊断的开销比较。',
      '- 工程预检包括3次SGCR更新和2次原实现参照更新，权重全部丢弃；最终810步不含预检成本。B-E均使用确定性计算。',
      '- 固定预算数据只支持当前5k单种子的机制诊断。若某组件无收益，优先依据检索、相关性和rho饱和情况解释，不假设消融链必然单调。','',
      '## 文件与验证','',
      '新增文件：src/continual/sgcr_memory.py、sgcr_retrieval.py、sgcr_gradients.py、sgcr_trainer.py；configs/experiment/sgcr.yaml；tests/test_sgcr.py；scripts/50_sgcr.py、51_sgcr_preflight.py、52_audit_sgcr.py、53_report_sgcr.py；SGCR_IMPLEMENTATION.md。旧实现均未修改。',
      '78项测试通过；基线完整复现、最终检查点重评、固定503测试ID、原型训练记忆来源、回放预算/索引、局部优先排序、梯度修正公式、一步优化器和冻结编码器检查通过；1364个历史文件哈希保持不变。',
      '本目录包含final_results.csv、stage_metrics.csv、sample_predictions.csv、batch_diagnostics.csv、replay_selection.csv、novelty_gradient_conflict.csv、retrieval_quality.csv、projection_summary.csv、forgetting.csv、historical_domain_changes.csv以及完整配置与审计。',
      '六张PDF均由这些CSV生成：average_seen_mae_vs_stage.pdf、forgetting_vs_stage.pdf、novelty_vs_gradient_conflict.pdf、selected_vs_random_conflict.pdf、gradient_cosine_before_after.pdf、replay_structural_relevance.pdf。',
      '远程检查点与逐步审计保留在/root/rivermind-data/continual_crystal/outputs/sgcr/runs/。交付包包含源码、CSV、PDF、日志和审计；大型权重保留在远程。所有训练已经停止。']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'delivery_source_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    print(final.to_string())


if __name__=='__main__':
    cfg=load_config('configs/experiment/sgcr.yaml');aggregate(cfg);analyses();figures();report()
