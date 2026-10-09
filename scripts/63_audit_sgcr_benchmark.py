"""Audit actual four-stage checkpoints, selections, memory and optimization traces."""
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
from src.continual.sgcr_benchmark import ConfiguredSGCRRunner, RUN_METHODS, verify_benchmark, expected_steps
from src.continual.sgcr_trainer import SharedStructuralPredictor
from src.continual.sgcr_memory import HierarchicalStructuralMemory
from src.continual.sgcr_gradients import stabilization_parameters
from src.continual.replay_augmented_rae import partition_memory
from src.continual.strategies import setup_experiment
from src.continual.v2_protocol import tensor_digest
from src.continual.metrics import continual_metrics
from src.models.alignn_wrapper import make_alignn
from src.data.dataset import make_loader, to_device


def make_hierarchy(settings, memory, allowed):
    return HierarchicalStructuralMemory(settings['prototype_count'],settings['structural_epsilon'],42,
        settings['calibration_batches'],32).fit(memory,allowed)


def audit(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    root=Path(cfg['sgcr']['output_root']);runs=Path(cfg['experiment']['run_root'])
    protected=json.loads((root/'protected_before_run.json').read_text())
    assert all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    assert sorted(p.name for p in runs.iterdir() if p.is_dir())==sorted(RUN_METHODS)
    stream,data=setup_experiment(cfg);verify_benchmark(cfg,stream,data);stages=list(stream)
    settings={**cfg['v2'],**cfg['replay_augmented'],**cfg['sgcr']}
    total_steps=expected_steps(stages,32,settings['epochs_per_stage'])
    tests={mid for s in stages for mid in s.test};vals={mid for s in stages for mid in s.val}
    cached=torch.load(cfg['v2']['cache'],map_location='cpu',weights_only=True)
    features=cached['records']
    reports=[];initializations=[]
    baseline_audits={s.stage_id:json.loads((runs/'B_structural_replay'/f'stage_{s.stage_id}/data_audit.json').read_text()) for s in stages}
    baseline_traces={s.stage_id:json.loads((runs/'B_structural_replay'/f'stage_{s.stage_id}/optimizer_trace.json').read_text()) for s in stages}
    for method in RUN_METHODS:
        direct=method=='A_direct';out=runs/method
        assert json.loads((out/'status.json').read_text())['status']=='passed'
        initial=json.loads((out/'initialization.json').read_text());initializations.append(initial)
        assert initial['optimizer_state_entries']==0
        samples=pd.read_csv(out/'sample_predictions.csv');metrics=pd.read_csv(out/'stage_metrics.csv')
        assert metrics.stage_id.tolist()==list(range(1,len(stages)+1))
        diag=pd.read_csv(out/'batch_diagnostics.csv') if not direct else pd.DataFrame()
        selection=pd.read_csv(out/'replay_selection.csv') if method=='E_full_sgcr' else pd.DataFrame()
        quality=pd.read_csv(out/'retrieval_quality_batches.csv') if method=='E_full_sgcr' else pd.DataFrame()
        if not direct:
            assert len(diag)==total_steps and diag[['novelty','rho','gradient_dot_before','gradient_dot_after']].notna().all().all()
            assert diag.rho.between(0,1).all()
        if method=='E_full_sgcr':
            assert len(quality)==total_steps
        all_train=set();matrix=np.full((len(stages),len(stages)),np.nan);maximum_difference=0.
        for stage in stages:
            s=stage.stage_id;p=out/f'stage_{s}';row=metrics[metrics.stage_id==s].iloc[0]
            evidence=json.loads((p/'data_audit.json').read_text())
            trace=json.loads((p/'optimizer_trace.json').read_text())
            assert evidence['current_train']==list(stage.train) and evidence['current_validation']==list(stage.val)
            assert set(evidence['memory_before'])<=all_train
            all_train.update(stage.train)
            memory=torch.load(p/'replay_memory.pt',map_location='cpu',weights_only=True)['items']
            assert set(memory)==set(evidence['memory_after']) and set(memory)<=all_train
            assert not set(memory)&(tests|vals)
            assert len(memory)==row.memory_size and len(memory)<=(0 if direct else settings['memory_size'])
            if direct:
                assert not evidence['memory_before'] and not evidence['replay_train'] and not evidence['reserved']
            else:
                for field in ['memory_before','memory_after','replay_train','reserved']:
                    assert evidence[field]==baseline_audits[s][field],(method,s,field)
                replay,reserve=partition_memory(evidence['memory_before'],2042+s,.2) if s>1 else ([],[])
                assert replay==evidence['replay_train'] and reserve==evidence['reserved']
                assert len(replay)+len(reserve)<=settings['memory_size']
                hierarchy=make_hierarchy(settings,memory,all_train)
                assert json.loads((p/'prototypes.json').read_text())==json.loads(json.dumps(hierarchy.state_dict()))
                assert set(evidence['prototype_training_ids'])==set(memory)
                for mid,record in memory.items():
                    assert record['prototype_id']==hierarchy.assignment[mid]
                    assert record['target']==float(data.metadata.at[mid,'target'])
                    assert record['domain_id']==data.domains[mid] and 1<=record['insertion_stage']<=s
            helper=object.__new__(ConfiguredSGCRRunner)
            helper.settings=settings;helper.batch_size=32;helper.smoke=False
            helper.replay_train=evidence['replay_train'];helper.features=features
            plan=helper.batches(stage) if s>1 else []
            assert len(trace)==len(plan)==row.optimizer_steps
            assert [t['current'] for t in trace]==[t['current'] for t in baseline_traces[s]]
            for t,planned in zip(trace,plan):
                assert t['current']==planned['current'] and t['epoch']==planned['epoch']
                assert t['optimizer_steps']==t['backward_steps']==1
                assert set(t['current'])<=set(stage.train) and np.isfinite(t['current_loss'])
                assert t['random_reference']==planned['replay']
                assert not set(t['replay'])&set(evidence['reserved'])
                if direct:
                    assert t['replay']==[] and t['autograd_calls']==0
                else:
                    assert t['autograd_calls']==2 and np.isfinite(t['replay_loss'])
                    assert len(t['current'])==len(t['replay'])==len(set(t['replay']))
                    assert set(t['replay'])<=set(evidence['replay_train'])
                    if method=='B_structural_replay':assert t['replay']==planned['replay']
            assert row.current_exposures==sum(len(t['current']) for t in trace)
            assert row.replay_exposures==sum(len(t['replay']) for t in trace)
            state=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=SharedStructuralPredictor(make_alignn(state['model_config'])).cuda()
            model.load_state_dict(state['model']);model.eval()
            assert not any('expert' in n or 'adapter' in n or 'residual' in n for n in state['model'])
            frozen=tensor_digest(model.stable_encoder.state_dict().items())
            assert frozen==initial['frozen_digest']==state['initial_stable_digest']
            if s==1:
                assert tensor_digest(model.predictor.state_dict().items())==initial['predictor_digest']
            if not direct:
                assert evidence['gradient_scope_names']==[n for n,p in stabilization_parameters(model.predictor,settings['gradient_stabilization_scope'])]
            table=samples[(samples.stage_id==s)&(samples.split=='test')]
            expected={mid for previous in stages[:s] for mid in previous.test}
            assert table.material_id.is_unique and set(table.material_id)==expected==set(evidence['test_ids'])
            assert np.allclose(table.absolute_error,(table.target-table.prediction).abs(),atol=3e-7,rtol=1e-6)
            ground_truth=torch.tensor([float(data.metadata.at[mid,'target']) for mid in table.material_id],dtype=torch.float32).numpy()
            assert np.allclose(table.target,ground_truth,atol=1e-8)
            for j,domain in enumerate(stream.order[:s]):
                matrix[s-1,j]=table[table.domain_id==domain].absolute_error.mean()
            computed=continual_metrics(matrix,s)
            assert abs(computed['average_seen_mae']-row.average_seen_mae)<1e-7
            assert abs(computed['forgetting']-row.forgetting)<1e-7
            predicted=[]
            with torch.no_grad():
                for batch in make_loader(data.subset(table.material_id.tolist()),32):
                    predicted.extend(model(to_device(batch,'cuda')['graphs'])['prediction'].cpu().tolist())
            difference=float(np.max(np.abs(np.asarray(predicted)-table.prediction.to_numpy())))
            assert difference<2e-5,(method,s,difference)
            maximum_difference=max(maximum_difference,difference)
            if not direct and s>1:
                previous_memory=torch.load(out/f'stage_{s-1}/replay_memory.pt',map_location='cpu',weights_only=True)['items']
                history=make_hierarchy(settings,previous_memory,evidence['memory_before'])
                assert json.loads((p/'historical_prototypes.json').read_text())==json.loads(json.dumps(history.state_dict()))
                stage_diag=diag[diag.stage_id==s].set_index('step')
                selection_by_step={int(k):g for k,g in selection[selection.stage_id==s].groupby('step')} if len(selection) else {}
                assert len(stage_diag)==len(trace) and stage_diag.index.is_unique
                for t in trace:
                    d=stage_diag.loc[t['step']];query=history.query(helper.hs(t['current']))
                    assert abs(query['novelty']-d.novelty)<1e-6 and abs(query['rho']-d.rho)<1e-6
                    projected=method=='E_full_sgcr' and d.gradient_dot_before<0 and d.rho>0 and d.replay_gradient_norm**2>1e-8 and d.current_gradient_norm**2>1e-8
                    assert d.projection_applied==projected
                    if projected:
                        theoretical=d.gradient_dot_before*(1-d.rho*d.replay_gradient_norm**2/(d.replay_gradient_norm**2+1e-8))
                        assert abs(d.gradient_dot_after-theoretical)<1e-5*max(1.,d.current_gradient_norm*d.replay_gradient_norm)
                    else:
                        assert abs(d.gradient_dot_before-d.gradient_dot_after)<1e-10
                    if method=='E_full_sgcr':
                        selected=selection_by_step[t['step']]
                        assert selected.material_id.is_unique and set(selected.material_id)<=set(evidence['replay_train'])
                        assert set(selected[selected.selected].material_id)==set(t['replay'])
                        distance,relevance=history.structural_scores(selected.material_id.tolist(),query['query'])
                        assert np.allclose(distance,selected.structural_distance,atol=1e-6)
                        assert np.allclose(relevance,selected.raw_structural_relevance,atol=1e-6)
                        expected_priority=settings['lambda_struct']*selected.structural_relevance+settings['lambda_conflict']*selected.normalized_conflict_score
                        assert np.allclose(expected_priority,selected.combined_priority,atol=1e-6)
                        local=selected[selected.selection_type=='local_conflict'];count=int(local.selected.sum())
                        ranked=local.sort_values(['combined_priority','material_id'],ascending=[False,True]).head(count)
                        assert set(ranked.material_id)==set(local[local.selected].material_id)
                if method=='E_full_sgcr':
                    for q in quality[quality.stage_id==s].itertuples():
                        chosen=json.loads(q.selected_ids);random=json.loads(q.random_ids)
                        assert len(chosen)==len(random)==q.count
                        assert set(random)<=set(evidence['replay_train'])
                        assert random==plan[q.step-1]['replay']
                        selected=selection_by_step[q.step];selected=selected[selected.selected]
                        assert abs(selected.conflict_score.mean()-q.selected_conflict)<1e-6
            del model,state
        final=pd.read_csv(out/'final_results.csv').iloc[0]
        assert final.optimizer_steps==metrics.optimizer_steps.sum()==total_steps
        assert abs(final.final_avg_mae-computed['average_seen_mae'])<1e-7
        assert abs(final.peak_stage_forgetting-metrics.forgetting.max())<1e-7
        report=dict(method=method,status='passed',final_mae=float(final.final_avg_mae),test_samples=len(tests),
            optimizer_steps=total_steps,stages=len(stages),max_checkpoint_prediction_difference=maximum_difference,
            projection_count=0 if direct else int(diag.projection_applied.sum()))
        reports.append(report);save_json(out/'audit.json',report)
        print('BENCHMARK_AUDIT',json.dumps(report),flush=True)
    assert all(initial==initializations[0] for initial in initializations)
    result=dict(status='passed',methods=reports,protected_files_unchanged=len(protected),
        identical_initial_predictor=True,identical_current_schedule=True,identical_B_E_memory=True,
        test_samples=len(tests),sample_count=len(data),domain_count=len(stages),global_memory_cap=settings['memory_size'])
    save_json(root/'audit.json',result)
    print('BENCHMARK_AUDIT_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    audit(load_config(p.parse_args().config))
