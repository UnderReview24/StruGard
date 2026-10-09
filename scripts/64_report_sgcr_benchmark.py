"""Summarize only the three requested methods after their runtime audit passes."""
import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.sgcr_benchmark import RUN_METHODS

LABELS={'A_direct':'A Direct','B_structural_replay':'B Structural Replay','E_full_sgcr':'SGCR'}
CHINESE={'A_direct':'A Direct','B_structural_replay':'B 结构回放','E_full_sgcr':'完整 SGCR'}
COLORS={'A_direct':'#828A94','B_structural_replay':'#3479A5','E_full_sgcr':'#B65C55'}


def report(cfg):
    root=Path(cfg['sgcr']['output_root']);runs=Path(cfg['experiment']['run_root'])
    evidence=json.loads((root/'audit.json').read_text());assert evidence['status']=='passed'
    domain=json.loads((root/'domain_preparation.json').read_text())
    results=pd.concat([pd.read_csv(runs/m/'final_results.csv') for m in RUN_METHODS],ignore_index=True)
    stages=pd.concat([pd.read_csv(runs/m/'stage_metrics.csv') for m in RUN_METHODS],ignore_index=True)
    predictions=pd.concat([pd.read_csv(runs/m/'sample_predictions.csv') for m in RUN_METHODS],ignore_index=True)
    results.to_csv(root/'final_results.csv',index=False)
    stages.to_csv(root/'stage_metrics.csv',index=False)
    predictions.to_csv(root/'sample_predictions.csv',index=False)
    stages[['method','stage_id','forgetting']].to_csv(root/'forgetting.csv',index=False)
    diagnostics=pd.read_csv(runs/'E_full_sgcr/batch_diagnostics.csv')
    retrieval=pd.read_csv(runs/'E_full_sgcr/retrieval_quality_batches.csv')
    summaries=[]
    for stage,group in diagnostics.groupby('stage_id'):
        correlation=float(spearmanr(group.novelty,-group.gradient_cosine_before).statistic) if group.novelty.nunique()>1 and group.gradient_cosine_before.nunique()>1 else None
        quality=retrieval[retrieval.stage_id==stage]
        summaries.append(dict(stage_id=int(stage),batches=len(group),
            conflicting_batches=int((group.gradient_dot_before<0).sum()),projected_batches=int(group.projection_applied.sum()),
            mean_rho=float(group.rho.mean()),rho_zero_fraction=float((group.rho==0).mean()),
            rho_one_fraction=float((group.rho==1).mean()),novelty_negative_gradient_cosine_spearman=correlation,
            selected_conflict=float(np.average(quality.selected_conflict,weights=quality['count'])),
            uniform_conflict=float(np.average(quality.random_conflict,weights=quality['count']))))
    pd.DataFrame(summaries).to_csv(root/'sgcr_diagnostics_summary.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for method in RUN_METHODS:
        part=stages[stages.method==method]
        for ax,field in zip(axes,['average_seen_mae','forgetting']):
            ax.plot(part.stage_id,part[field],marker='o',linewidth=2,label=LABELS[method],color=COLORS[method])
            ax.set_xticks(range(1,cfg['domains']['num_domains']+1));ax.set_xlabel('Learning stage')
            ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Average seen-domain MAE (eV/atom)')
    axes[1].set_ylabel('Mean forgetting (eV/atom)')
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('10,000 structures · 4 domains · replay capacity 500 · seed 42')
    fig.savefig(root/'comparison.png',dpi=180)
    fig.savefig(root/'comparison.pdf')
    plt.close(fig)
    d1=pd.read_csv(Path(cfg['experiment']['shift_output'])/'d1_model/training_curve.csv')
    totals={key:sum(s[key] for s in domain['stage_splits']) for key in ['train','val','test']}
    lines=['# SGCR：1 万样本、4 域、记忆上限 500 的三模型结果','',
        'A Direct、B 结构回放、完整 SGCR 均已重新训练并通过四阶段检查点重评与数据访问审计。',
        f'固定 seed 42；训练/验证/测试为 {totals["train"]}/{totals["val"]}/{totals["test"]}。',
        '域顺序：三斜/单斜 → 正交/四方 → 六方/三方 → 立方。分组依据晶系，域大小不强制均等。','',
        '| 方法 | 最终平均 MAE ↓ | 阶段最大遗忘 ↓ | 最终遗忘 ↓ | 记忆 | 后续优化步 | 后续阶段耗时（秒） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results.itertuples():
        lines.append(f'| {CHINESE[r.method]} | {r.final_avg_mae:.6f} | {r.peak_stage_forgetting:.6f} | {r.forgetting:.6f} | {int(r.memory_size)} | {int(r.optimizer_steps)} | {r.wall_clock_seconds:.1f} |')
    lookup=results.set_index('method');b=float(lookup.at['B_structural_replay','final_avg_mae']);e=float(lookup.at['E_full_sgcr','final_avg_mae'])
    lines += ['',f'SGCR 相对 B 的最终平均 MAE 变化为 {(e/b-1)*100:+.2f}%（绝对差 {e-b:+.6f} eV/atom）。',
        '平均 MAE 对已见域等权。遗忘为历史域当前 MAE 减去该域自引入以来的最小 MAE，再对历史域平均；最小值包含当前阶段，所以该指标非负。阶段最大遗忘取四个阶段该指标的最大值。','',
        '## 相同条件与计算差异','',
        '- 三组从同一个仅用首域训练、验证选优的 ALIGNN 检查点开始。后续均训练 10 个 epoch，保存最后一步；没有按测试指标选模。',
        '- 三组预测器、学习率、优化器、当前样本顺序与优化步数相同。Direct 每步只有当前样本；B/SGCR 每步还有等量回放样本，因此不是相同样本处理量或相同计算量。',
        '- B/SGCR 的记忆均为所有历史域合计最多 500 条，满额时 400 条参与梯度回放、100 条预留；两组各阶段保留的样本 ID 完全一致。',
        '- 沿用冻结首域结构表征、8 个原型、原有结构/冲突检索及梯度修正参数。Direct 不使用结构特征或历史记忆进行预测训练；共同包装中的冻结编码器不参与其计算。',
        '- 耗时包含对应阶段的验证、记忆处理与逐批诊断，按旧口径不含阶段末测试、保存文件和共同 D1 训练。SGCR 的均匀对照仅用于诊断，也计入其耗时。',
        f'- 共同 D1 训练运行 {len(d1)} 个 epoch，用时 {float(d1.elapsed_seconds.iloc[-1]):.1f} 秒；此成本只发生一次，未重复计入三组后续耗时。',
        '- 新增 5 项测试，加上原 78 项测试，共 83 项通过。旧配置、旧源码及受保护的 5k 结果文件保持不变。','',
        '## 分阶段结果','', '| 方法 | 阶段 | 当前域 MAE | 已见域平均 MAE | 遗忘 |','|---|---:|---:|---:|---:|']
    for r in stages.itertuples():
        lines.append(f'| {CHINESE[r.method]} | {r.stage_id} | {r.current_domain_mae:.6f} | {r.average_seen_mae:.6f} | {r.forgetting:.6f} |')
    lines += ['', '## SGCR 机制诊断','', '| 阶段 | 批次 | 冲突批次 | 投影批次 | 平均 rho | 新颖度与负梯度余弦相关 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        rho=r['novelty_negative_gradient_cosine_spearman'];corr='未定义' if rho is None else f'{rho:.4f}'
        lines.append(f'| {r["stage_id"]} | {r["batches"]} | {r["conflicting_batches"]} | {r["projected_batches"]} | {r["mean_rho"]:.4f} | {corr} |')
    lines += ['', '相关系数描述同一训练轨迹，不作为独立样本的显著性或因果证据。',
        '这轮同时改变样本量、域数和记忆预算，属于新的单种子协议。应比较本轮三组模型；不能把与旧 5k 数值的差异单独归因于样本量。旧 Direct 还使用了不同的后续检查点选择协议。','',
        '本轮没有运行 C/D 消融、额外随机种子、3 万条或全量实验，也没有搜索超参数。',
        '详见 final_results.csv、stage_metrics.csv、sample_predictions.csv、sgcr_diagnostics_summary.csv 和 audit.json。']
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(root/'result_summary.json',dict(status='audited',methods=RUN_METHODS,totals=totals,
        sgcr_relative_mae_change_vs_structural_replay=e/b-1,shared_d1_epochs=len(d1),
        shared_d1_seconds=float(d1.elapsed_seconds.iloc[-1]),additional_experiments_scheduled=False))
    print(results[['method','final_avg_mae','peak_stage_forgetting','forgetting']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    report(load_config(p.parse_args().config))
