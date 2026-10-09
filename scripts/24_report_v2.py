"""Report the seven requested runs without presuming any scientific advantage."""
import _bootstrap
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.continual.v2_trainer import VARIANTS
from src.utils.config import load_config
from src.utils.logging import save_json


NAMES={'full':'SP-Crystal-V2','no_shift':'w/o shift residual','no_slow':'w/o slow adaptation',
       'novelty_only':'Novelty-only expansion','no_expansion':'w/o expansion',
       'always_expand':'Always expand','structural_replay':'Structural replay'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment/v2.yaml');a=p.parse_args()
    cfg=load_config(a.config);root=Path(cfg['experiment']['run_root']);out=root/'summary'
    out.mkdir(exist_ok=True)
    finals,stages,events,samples,audits=[],[],[],[],[]
    for name in VARIANTS:
        run=root/name
        audit=json.loads((run/'audit.json').read_text())
        if audit['status']!='passed':raise ValueError(f'Unverified method: {name}')
        audits.append(audit)
        finals.append(pd.read_csv(run/'final_results.csv'))
        stages.append(pd.read_csv(run/'stage_metrics.csv'))
        event=pd.read_csv(run/'controller_history.csv');event.insert(0,'method',name);events.append(event)
        for stage in range(1,4):
            table=pd.read_csv(run/f'stage_{stage}/predictions.csv')
            table.insert(0,'stage',stage);table.insert(0,'method',name);samples.append(table)
    final=pd.concat(finals,ignore_index=True);stage=pd.concat(stages,ignore_index=True)
    event=pd.concat(events,ignore_index=True);sample=pd.concat(samples,ignore_index=True)
    counts=pd.DataFrame(audits)[['variant','optimizer_steps','effective_optimizer_steps','current_exposures','replay_exposures']]
    counts=counts.rename(columns={'variant':'method'})
    final=final.merge(counts,on='method',suffixes=('_final_stage','_total'))
    final.to_csv(out/'final_results.csv',index=False)
    stage.to_csv(out/'stage_metrics.csv',index=False)
    event.to_csv(out/'controller_history.csv',index=False)
    sample.to_csv(out/'sample_predictions.csv',index=False)
    counts.to_csv(out/'compute_audit.csv',index=False)
    per_domain=[]
    for r in stage.itertuples():
        for domain,mae in json.loads(r.per_domain_mae).items():
            per_domain.append(dict(method=r.method,stage=r.stage,domain_id=int(domain),mae=mae))
    pd.DataFrame(per_domain).to_csv(out/'per_domain_mae.csv',index=False)
    stage[['method','stage','forgetting']].to_csv(out/'forgetting.csv',index=False)
    stage[['method','stage','number_of_experts','trainable_parameter_count','total_parameter_count','reference_parameter_count']].to_csv(out/'parameter_growth.csv',index=False)
    stage[['method','stage','novelty_mean','novelty_quantiles','reuse_fraction','adapt_fraction','expand_fraction']].to_csv(out/'routing_summary.csv',index=False)
    old=pd.read_csv('outputs/pilot/rae/final_results.csv').iloc[0]
    reference=final.set_index('method');full=float(reference.at['full','average_seen_mae'])
    improvement=100*(float(old.average_seen_mae)-full)/float(old.average_seen_mae)
    gap=100*(full/float(reference.at['always_expand','average_seen_mae'])-1)
    full_path=event[(event.method=='full') & (event.stage>1)].decision.tolist()
    novelty_path=event[(event.method=='novelty_only') & (event.stage>1)].decision.tolist()
    initial_a=torch.load(root/'full/stage_1/checkpoint.pt',map_location='cpu',weights_only=True)['model']
    initial_b=torch.load(root/'novelty_only/stage_1/checkpoint.pt',map_location='cpu',weights_only=True)['model']
    equal_initial_weights=all(torch.equal(v,initial_b[k]) for k,v in initial_a.items())
    trace_a=json.loads((root/'full/stage_2/optimizer_trace.json').read_text())
    trace_b=json.loads((root/'novelty_only/stage_2/optimizer_trace.json').read_text())
    numerical=dict(initial_weights_identical=equal_initial_weights,controller_paths_identical=full_path==novelty_path,
        first_step_loss_difference=abs(trace_a[0]['loss']-trace_b[0]['loss']),
        deterministic_algorithms_enforced=False,
        interpretation='Identical decisions do not isolate need-aware gating. Floating-point training divergence is present; do not attribute the final difference to a different controller path.')
    save_json(out/'numerical_reproducibility.json',numerical)
    equal_fields=['optimizer_steps','current_exposures','replay_exposures']
    checks=dict(runtime_and_output_audits_passed=True,methods=7,samples=5000,domains=3,seed=42,
        step_budgets_and_sample_exposures_matched=all(counts[k].nunique()==1 for k in equal_fields),
        effective_updates_matched=counts.effective_optimizer_steps.nunique()==1,
        full_better_than_old_rae=full<float(old.average_seen_mae),relative_improvement_over_old_rae_percent=improvement,
        slow_adaptation_better=full<float(reference.at['no_slow','average_seen_mae']),
        shift_residual_better=full<float(reference.at['no_shift','average_seen_mae']),
        full_fewer_experts_than_always_expand=int(reference.at['full','number_of_experts'])<int(reference.at['always_expand','number_of_experts']),
        gap_to_always_expand_percent=gap,validation_only_controller=True,
        full_and_novelty_only_decisions_identical=full_path==novelty_path,
        larger_experiments_started=False,statistical_significance_established=False,
        old_rae_comparison='Archived pilot context; old R/A/E is not compute-matched to the V2 suite.')
    save_json(out/'findings.json',checks)
    comparison=final[['method','average_seen_mae','forgetting','number_of_experts','trainable_parameter_count',
                     'optimizer_steps_total','effective_optimizer_steps_total','replay_exposures_total']]
    comparison.to_csv(out/'comparison.csv',index=False)
    compact=comparison[['method','average_seen_mae','forgetting','number_of_experts','effective_optimizer_steps_total']].copy()
    compact['method']=compact.method.map(NAMES)
    compact.columns=['方法','平均 MAE','遗忘指标','专家数','有效更新步数']
    change_text=(f'误差下降 {improvement:.2f}%' if improvement>=0 else f'误差上升 {-improvement:.2f}%')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    colors=['#C44E52','#4C72B0','#8172B3','#CCB974','#64B5CD','#55A868','#333333']
    for name,color in zip(VARIANTS,colors):
        rows=stage[stage.method==name]
        axes[0].plot(rows.stage,rows.average_seen_mae,'o-',color=color,label=NAMES[name],markersize=4)
        axes[1].plot(rows.stage,rows.number_of_experts,'o-',color=color,markersize=4)
    axes[0].set(xlabel='Stream stage',ylabel='Average seen-domain MAE (eV/atom)',xticks=[1,2,3])
    axes[1].set(xlabel='Stream stage',ylabel='Number of experts',xticks=[1,2,3],yticks=range(1,int(stage.number_of_experts.max())+1))
    for ax in axes:ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.05),fontsize=9)
    fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(out/'v2_performance_and_growth.png',dpi=180,bbox_inches='tight');plt.close(fig)
    text=['# SP-Crystal-V2：5k 七组实验结果','',
          '范围固定为 5000 条晶体结构、3 个结构域、seed=42；未启动 3 万条或全量实验。','',
          '按照用户确认，保留严格冻结 REUSE。七组仅在预留步骤与样本暴露方面一致，有效参数更新步数不同；这些结果不能称为严格等计算量训练对比。','',
          compact.to_markdown(index=False,floatfmt='.6f'),'',
          '## 相对于旧模型','',
          f'旧 R/A/E 的平均 MAE 为 {float(old.average_seen_mae):.6f}；V2 为 {full:.6f}，{change_text}。旧结果只用于诊断参考，其计算协议与本轮不同。','',
          '## 控制实验的数值证据','',
          f"- 慢速层：完整 V2 {full:.6f}；冻结慢速层 {float(reference.at['no_slow','average_seen_mae']):.6f}。",
          f"- 位移残差：完整 V2 {full:.6f}；去掉位移输入 {float(reference.at['no_shift','average_seen_mae']):.6f}。",
          f"- 容量：V2 {int(reference.at['full','number_of_experts'])} 个专家；Always Expand {int(reference.at['always_expand','number_of_experts'])} 个；V2 的 MAE 相对差距 {gap:.2f}%。",'',
          '这些是单种子的点估计，不能据此声称统计显著或推广到其他结构域顺序。增长是否有效，需要同时看误差与专家数；满足扩展判据不等于证明了性能收益。','',
          '完整 V2 与 novelty-only 在本次两个后续域上采取了相同的动作序列。因此，这个基准尚未展示“需要性判据”改变容量决策的作用。两者虽然初始权重一致，但训练日志存在从微小浮点差异逐步扩大的轨迹差异；没有强制逐位确定性，不能把两者最终 MAE 差归因于不同的扩展动作。','',
          '## 预算与数据使用','',counts.to_markdown(index=False),'',
          f"预留优化步数和当前/回放样本暴露一致：{checks['step_budgets_and_sample_exposures_matched']}；有效更新步数一致：{checks['effective_updates_matched']}。",'',
          '七组使用相同的固定测试集、逐步当前样本和回放 ID。控制器只使用当前训练结构统计及当前验证集误差。测试集仅在该阶段训练和路由决策完成后评估。阶段 1 共用首域检查点，不重复计算为本轮新增优化步骤。','',
          '当前完整模型在第 3 域选择 REUSE，低结构新颖度并未同时保证低预测误差。这是后续诊断线索；本次没有据此改阈值或追加调参实验。','',
          '## 完整记录','',
          'controller_history.csv 记录每域新颖度、历史验证 MAE、适应前后误差、决策和新专家；sample_predictions.csv 提供各阶段已见测试结构的新颖度、位移范数、门控值、预测及误差。结构回放基线没有门控模块，其 gate_value 留空。','',
          '所有结论均来自保存的 CSV；不修改测试标签、不根据测试误差调整阈值。完整检查点、预算记录和审计结果保留在远程 outputs/v2_5k/。七组完成后停止，没有后续大数据任务。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(comparison.to_string(index=False),flush=True)
    print(json.dumps(checks,indent=2),flush=True)
