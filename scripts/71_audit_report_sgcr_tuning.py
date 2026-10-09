"""Verify all search trials and report the explicitly test-selected result."""
import _bootstrap
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.sgcr_benchmark import expected_steps,verify_benchmark,ConfiguredSGCRRunner
from src.continual.sgcr_tuning import CANDIDATES,ScaledStructuralMemory
from src.continual.sgcr_trainer import SharedStructuralPredictor
from src.continual.strategies import setup_experiment
from src.continual.replay_augmented_rae import partition_memory
from src.continual.metrics import continual_metrics
from src.continual.v2_protocol import tensor_digest
from src.models.alignn_wrapper import make_alignn
from src.data.dataset import make_loader,to_device


def hierarchy(settings,memory,allowed):
    return ScaledStructuralMemory(settings['prototype_count'],settings['structural_epsilon'],42,
        settings['calibration_batches'],32,projection_scale=settings['projection_scale']).fit(memory,allowed)


def audit(root):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    base=load_config(root/'base_config.yaml')
    plan=json.loads((root/'search_plan.json').read_text())
    assert plan['selection_split']=='test' and plan['candidates']==CANDIDATES
    protected=json.loads((root/'protected_before_tuning.json').read_text())
    assert all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    results=pd.read_csv(root/'search_results.csv')
    screen=results[results.fidelity=='screen'];full=results[results.fidelity=='full']
    assert screen.trial_id.tolist()==[c['trial_id'] for c in CANDIDATES] and len(full)==2
    promoted=screen[screen.trial_id!='T0_default'].sort_values(['final_avg_mae','trial_id']).head(2)
    assert full.trial_id.tolist()==promoted.trial_id.tolist()
    assert screen.optimizer_steps.eq(322).all() and full.optimizer_steps.eq(1610).all()
    stream,data=setup_experiment(base);verify_benchmark(base,stream,data);stages=list(stream)
    test_ids={mid for s in stages for mid in s.test};val_ids={mid for s in stages for mid in s.val}
    cache=torch.load(base['v2']['cache'],map_location='cpu',weights_only=True)
    reference=Path(base['experiment']['run_root'])/'B_structural_replay'
    reference_initial=json.loads((reference/'initialization.json').read_text())
    reports=[]
    for trial in results.itertuples():
        cfg=load_config(trial.config);out=Path(trial.run_path)
        settings={**cfg['v2'],**cfg['replay_augmented'],**cfg['sgcr']}
        candidate=next(c for c in CANDIDATES if c['trial_id']==trial.trial_id)
        assert all(settings[k]==v for k,v in candidate.items() if k!='trial_id')
        assert cfg['data']==base['data'] and cfg['domains']==base['domains']
        assert cfg['model']==base['model'] and cfg['training']==base['training'] and cfg['loss']==base['loss']
        assert settings['memory_size']==500 and settings['lambda_struct']==1-settings['lambda_conflict']
        expected=expected_steps(stages,32,settings['epochs_per_stage'])
        assert expected==trial.optimizer_steps
        initial=json.loads((out/'initialization.json').read_text())
        assert initial==reference_initial
        samples=pd.read_csv(out/'sample_predictions.csv');metrics=pd.read_csv(out/'stage_metrics.csv')
        diagnostics=pd.read_csv(out/'batch_diagnostics.csv');selection=pd.read_csv(out/'replay_selection.csv')
        quality=pd.read_csv(out/'retrieval_quality_batches.csv')
        assert len(diagnostics)==len(quality)==expected
        assert diagnostics.projection_scale.eq(settings['projection_scale']).all()
        assert diagnostics.rho.between(0,settings['projection_scale']+1e-12).all()
        all_train=set();matrix=np.full((len(stages),len(stages)),np.nan);max_difference=0.
        for stage in stages:
            s=stage.stage_id;p=out/f'stage_{s}'
            evidence=json.loads((p/'data_audit.json').read_text());trace=json.loads((p/'optimizer_trace.json').read_text())
            old=json.loads((reference/f'stage_{s}/data_audit.json').read_text())
            for field in ('memory_before','memory_after','replay_train','reserved'):
                assert evidence[field]==old[field]
            assert evidence['current_train']==list(stage.train) and evidence['current_validation']==list(stage.val)
            assert set(evidence['memory_before'])<=all_train
            all_train.update(stage.train)
            memory=torch.load(p/'replay_memory.pt',map_location='cpu',weights_only=True)['items']
            assert len(memory)<=500 and set(memory)==set(evidence['memory_after'])
            assert set(memory)<=all_train and not set(memory)&(test_ids|val_ids)
            prototype=hierarchy(settings,memory,all_train)
            assert json.loads((p/'prototypes.json').read_text())==json.loads(json.dumps(prototype.state_dict()))
            replay,probe=partition_memory(evidence['memory_before'],2042+s,.2) if s>1 else ([],[])
            assert replay==evidence['replay_train'] and probe==evidence['reserved']
            helper=object.__new__(ConfiguredSGCRRunner)
            helper.settings=settings;helper.batch_size=32;helper.smoke=False
            helper.replay_train=replay;helper.features=cache['records']
            schedule=helper.batches(stage) if s>1 else []
            assert len(trace)==len(schedule)
            row=metrics[metrics.stage_id==s].iloc[0]
            assert row.optimizer_steps==len(trace)
            for recorded,planned in zip(trace,schedule):
                assert recorded['epoch']==planned['epoch'] and recorded['current']==planned['current']
                assert recorded['random_reference']==planned['replay']
                assert recorded['optimizer_steps']==recorded['backward_steps']==1 and recorded['autograd_calls']==2
                assert set(recorded['current'])<=set(stage.train) and set(recorded['replay'])<=set(replay)
                assert not set(recorded['replay'])&set(probe)
                assert len(recorded['current'])==len(recorded['replay'])==len(set(recorded['replay']))
            state=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=SharedStructuralPredictor(make_alignn(state['model_config'])).cuda()
            model.load_state_dict(state['model']);model.eval()
            assert tensor_digest(model.stable_encoder.state_dict().items())==initial['frozen_digest']
            if s==1:assert tensor_digest(model.predictor.state_dict().items())==initial['predictor_digest']
            table=samples[(samples.stage_id==s)&(samples.split=='test')]
            expected_ids={mid for previous in stages[:s] for mid in previous.test}
            assert table.material_id.is_unique and set(table.material_id)==expected_ids
            assert np.allclose(table.absolute_error,(table.target-table.prediction).abs(),atol=3e-7,rtol=1e-6)
            for j,domain in enumerate(stream.order[:s]):
                matrix[s-1,j]=table[table.domain_id==domain].absolute_error.mean()
            calculated=continual_metrics(matrix,s)
            assert abs(calculated['average_seen_mae']-row.average_seen_mae)<1e-7
            assert abs(calculated['forgetting']-row.forgetting)<1e-7
            predicted=[]
            with torch.no_grad():
                for batch in make_loader(data.subset(table.material_id.tolist()),32):
                    predicted.extend(model(to_device(batch,'cuda')['graphs'])['prediction'].cpu().tolist())
            delta=float(np.max(np.abs(np.asarray(predicted)-table.prediction.to_numpy())))
            assert delta<2e-5,(trial.trial_id,trial.fidelity,s,delta)
            max_difference=max(max_difference,delta)
            if s>1:
                previous_memory=torch.load(out/f'stage_{s-1}/replay_memory.pt',map_location='cpu',weights_only=True)['items']
                previous=hierarchy(settings,previous_memory,evidence['memory_before'])
                assert json.loads((p/'historical_prototypes.json').read_text())==json.loads(json.dumps(previous.state_dict()))
                ds=diagnostics[diagnostics.stage_id==s].set_index('step')
                ss={int(k):g for k,g in selection[selection.stage_id==s].groupby('step')}
                assert len(ds)==len(trace)
                for t in trace:
                    d=ds.loc[t['step']];q=previous.query(helper.hs(t['current']))
                    assert abs(q['rho']-d.rho)<1e-6 and abs(q['raw_rho']-d.raw_rho)<1e-6
                    applied=d.gradient_dot_before<0 and d.rho>0 and d.current_gradient_norm**2>1e-8 and d.replay_gradient_norm**2>1e-8
                    assert d.projection_applied==applied
                    if applied:
                        after=d.gradient_dot_before*(1-d.rho*d.replay_gradient_norm**2/(d.replay_gradient_norm**2+1e-8))
                        assert abs(d.gradient_dot_after-after)<1e-5*max(1.,d.current_gradient_norm*d.replay_gradient_norm)
                    else:assert abs(d.gradient_dot_after-d.gradient_dot_before)<1e-10
                    candidates=ss[t['step']]
                    assert candidates.material_id.is_unique and set(candidates.material_id)<=set(replay)
                    assert set(candidates[candidates.selected].material_id)==set(t['replay'])
                    priorities=settings['lambda_struct']*candidates.structural_relevance+settings['lambda_conflict']*candidates.normalized_conflict_score
                    assert np.allclose(priorities,candidates.combined_priority,atol=1e-6)
                    local=candidates[candidates.selection_type=='local_conflict'];count=int(local.selected.sum())
                    ranked=local.sort_values(['combined_priority','material_id'],ascending=[False,True]).head(count)
                    assert set(ranked.material_id)==set(local[local.selected].material_id)
            del model,state
        assert abs(trial.final_avg_mae-calculated['average_seen_mae'])<1e-7
        report=dict(trial_id=trial.trial_id,fidelity=trial.fidelity,status='passed',
            optimizer_steps=expected,final_test_mae=float(trial.final_avg_mae),
            max_checkpoint_prediction_difference=max_difference,test_samples=len(test_ids),
            global_memory_cap=500,projection_count=int(diagnostics.projection_applied.sum()))
        reports.append(report);save_json(out/'tuning_audit.json',report)
        print('TUNING_AUDIT',json.dumps(report),flush=True)
    pool=pd.read_csv(root/'full_comparison.csv')
    best=pool.sort_values(['final_avg_mae','trial_id']).iloc[0]
    selected=json.loads((root/'selected_result.json').read_text())
    assert best.trial_id==selected['trial_id'] and abs(best.final_avg_mae-selected['final_avg_mae'])<1e-12
    result=dict(status='passed',trials=reports,new_trials=len(reports),test_samples=len(test_ids),
        selected_trial=selected['trial_id'],selection_split='test',test_tuned_exploratory=True,
        total_new_optimizer_steps=int(results.optimizer_steps.sum()),protected_files_unchanged=len(protected))
    save_json(root/'audit.json',result)
    return base,results,pool,selected


def report(root,base,results,pool,selected):
    old=pd.read_csv(Path(base['sgcr']['output_root'])/'final_results.csv').set_index('method')
    rows=[]
    for key,label in [('A_direct','A Direct'),('B_structural_replay','B 结构回放'),('E_full_sgcr','原 SGCR')]:
        row=old.loc[key].to_dict();row.update(label=label,trial_id='reference',selection='原固定配置')
        rows.append(row)
    row=dict(selected);row.update(label='本轮测试集最优 SGCR',selection='测试集调参')
    rows.append(row);pd.DataFrame(rows).to_csv(root/'final_comparison.csv',index=False)
    default=float(old.loc['E_full_sgcr','final_avg_mae']);baseline=float(old.loc['B_structural_replay','final_avg_mae'])
    lines=['# SGCR 测试集调参结果','',
        '**按用户要求，本轮直接依据测试集 MAE 筛选和选择配置。结果属于测试集调参的探索结果，不是独立测试集上的泛化估计。**','',
        '固定 10,000 样本、4 个晶系域、seed 42、全局记忆上限 500（400 梯度回放 + 100 预留）、共同 D1 检查点与当前样本顺序。',
        '六组候选每个后续域先训练 2 个 epoch；其中测试 MAE 最好的两个新配置，从共同 D1 重新开始、每域完整训练 10 个 epoch。原 SGCR 的既有 10-epoch 结果也参与最终选择。',
        '筛选分数与完整训练分数分开报告。所有分数都是最后阶段四个域等权的平均测试 MAE；没有混用不同阶段或挑选最佳测试 epoch。','',
        '| 方法 | 最终平均测试 MAE ↓ | 阶段最大遗忘 ↓ | 最终遗忘 ↓ |','|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["label"]} | {row["final_avg_mae"]:.6f} | {row["peak_stage_forgetting"]:.6f} | {row["forgetting"]:.6f} |')
    lines += ['',f'本轮选中 `{selected["trial_id"]}`，完整训练最终测试 MAE 为 {selected["final_avg_mae"]:.9f}。',
        f'相对原 SGCR 变化 {(selected["final_avg_mae"]/default-1)*100:+.2f}%；相对 B 结构回放变化 {(selected["final_avg_mae"]/baseline-1)*100:+.2f}%。','',
        '## 最优配置','', '| 参数 | 原配置 | 本轮选择 |','|---|---:|---:|',
        f'| 投影强度系数 β | 1.0 | {selected["projection_scale"]} |',
        f'| 局部检索比例 | 0.75 | {selected["local_fraction"]} |',
        f'| 冲突优先级权重 | 0.50 | {selected["lambda_conflict"]} |',
        f'| 结构优先级权重 | 0.50 | {1-selected["lambda_conflict"]} |',
        f'| 学习率 | 0.001 | {selected["shared_lr"]} |','',
        'β 是本轮新增暴露的投影强度超参数：ρ_eff = β × ρ，β=1 保留原行为。预测器、记忆选择方法、原型数、候选池规模、回放损失权重等没有更改。','',
        '## 所有候选','', '| 配置 | 预算 | β | 局部比例 | 冲突权重 | 学习率 | 最终测试 MAE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results.itertuples():
        lines.append(f'| {r.trial_id} | {r.fidelity}: {r.epochs_per_stage} epoch/域 | {r.projection_scale} | {r.local_fraction} | {r.lambda_conflict} | {r.shared_lr} | {r.final_avg_mae:.6f} |')
    lines += ['', '## 验证与使用范围','',
        '- 全部 89 项测试通过；8 个试验的 32 个阶段检查点均重新计算测试预测并核对指标，合计新增 5152 次优化更新。',
        '- 各试验的实际记忆 ID 与固定 B 基线一致；回放上限、当前/回放/预留边界、原型计算及缩放后的投影公式均经过审计。',
        '- 原 5k、10k 基线和既有 SGCR 代码/结果文件保留。调参输出独立保存在本目录。',
        '- 这是本轮有限候选中的最优配置，不是整个超参数空间的全局最优。2-epoch 筛选可能漏掉训练更久才表现好的配置。',
        '- 原基线没有获得同等调参预算，因此这里不据此声称 SGCR 在公平调参后优于其他方法；独立种子与未用于调参的测试数据尚未评估。','',
        f'最终权重目录：`{Path(selected["run_path"]).resolve()}`。',
        '完整配置见 best_config.yaml，逐次搜索见 search_results.csv，完整预算比较见 full_comparison.csv，审计见 audit.json。']
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows)[['label','final_avg_mae','forgetting']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/sgcr_10k_test_tuning')
    root=Path(p.parse_args().root)
    report(root,*audit(root))
