import _bootstrap
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.sgcr_trainer import METHODS,SharedStructuralPredictor,SGCRRunner,verify_baseline
from src.continual.sgcr_memory import HierarchicalStructuralMemory
from src.continual.sgcr_gradients import stabilization_parameters
from src.continual.strategies import setup_experiment
from src.continual.replay_augmented_runner import verify_fixed_split
from src.continual.metrics import continual_metrics
from src.continual.v2_protocol import tensor_digest
from src.models.alignn_wrapper import make_alignn
from src.data.embeddings import encoder_digest
from src.data.dataset import make_loader,to_device


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    cfg=load_config('configs/experiment/sgcr.yaml');root=Path(cfg['sgcr']['output_root']);runs=Path(cfg['experiment']['run_root'])
    protected=json.loads((root/'protected_files_before.json').read_text())
    changed=[name for name,digest in protected.items() if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    verify_baseline(cfg)
    stream,data=setup_experiment(cfg);verify_fixed_split(cfg,stream,data);stages=list(stream)
    test_ids={mid for s in stages for mid in s.test};val_ids={mid for s in stages for mid in s.val}
    cache=torch.load(cfg['v2']['cache'],map_location='cpu',weights_only=True)
    features={mid:{'stable_embedding':r['stable_embedding']} for mid,r in cache['records'].items()};del cache
    reports=[]
    for method in METHODS[1:]:
        out=runs/method
        assert json.loads((out/'status.json').read_text())['status']=='passed'
        samples=pd.read_csv(out/'sample_predictions.csv');metrics=pd.read_csv(out/'stage_metrics.csv')
        diag=pd.read_csv(out/'batch_diagnostics.csv')
        assert len(diag)==810 and diag[['novelty','rho','gradient_dot_before','gradient_dot_after']].notna().all().all()
        assert diag.rho.between(0,1).all()
        selections=pd.read_csv(out/'replay_selection.csv') if (out/'replay_selection.csv').exists() else pd.DataFrame()
        quality=pd.read_csv(out/'retrieval_quality_batches.csv') if (out/'retrieval_quality_batches.csv').exists() else pd.DataFrame()
        if method in METHODS[3:]:assert len(quality)==810
        all_train=set();matrix=np.full((3,3),np.nan);frozen=None;maximum_difference=0.
        for stage in stages:
            s=stage.stage_id;p=out/f'stage_{s}';row=metrics[metrics.stage_id==s].iloc[0]
            evidence=json.loads((p/'data_audit.json').read_text())
            trace=json.loads((p/'optimizer_trace.json').read_text())
            reference=Path(cfg['sgcr']['baseline_reference'])/f'stage_{s}'
            old=json.loads((reference/'data_audit.json').read_text());old_trace=json.loads((reference/'optimizer_trace.json').read_text())
            for field in ['memory_before','memory_after','replay_train','reserved']:assert evidence[field]==old[field]
            assert evidence['current_train']==list(stage.train) and evidence['current_validation']==list(stage.val)
            assert set(evidence['memory_before'])<=all_train
            all_train.update(stage.train)
            memory=torch.load(p/'replay_memory.pt',map_location='cpu',weights_only=True)['items']
            assert set(memory)==set(evidence['memory_after']) and len(memory)<=2000
            assert set(memory)<=all_train and not set(memory)&(test_ids|val_ids)
            hierarchy=HierarchicalStructuralMemory(cfg['sgcr']['prototype_count'],calibration_batches=cfg['sgcr']['calibration_batches']).fit(memory,all_train)
            saved=json.loads((p/'prototypes.json').read_text())
            assert saved==json.loads(json.dumps(hierarchy.state_dict()))
            for mid,record in memory.items():
                assert record['prototype_id']==hierarchy.assignment[mid]
                assert record['target']==float(data.metadata.at[mid,'target'])
                assert record['domain_id']==data.domains[mid] and 1<=record['insertion_stage']<=s
            state=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=SharedStructuralPredictor(make_alignn(state['model_config'])).cuda();model.load_state_dict(state['model']);model.eval()
            assert not any('expert' in n or 'adapter' in n or 'residual' in n for n in state['model'])
            digest=tensor_digest(model.stable_encoder.state_dict().items())
            if frozen is None:frozen=digest
            assert digest==frozen==state['initial_stable_digest']
            assert evidence['gradient_scope_names']==[n for n,p in stabilization_parameters(model.predictor,cfg['sgcr']['gradient_stabilization_scope'])]
            helper=object.__new__(SGCRRunner);helper.full_stages=stages;helper.settings={**cfg['replay_augmented'],**cfg['sgcr']}
            helper.features=features;helper.data=data;helper.selection_digest=encoder_digest(model.stable_encoder)
            helper.original_selection_features(stage)
            table=samples[(samples.stage_id==s)&(samples.split=='test')]
            expected={mid for st in stages[:s] for mid in st.test}
            assert set(table.material_id)==expected and table.material_id.is_unique
            assert np.allclose(table.absolute_error,(table.target-table.prediction).abs(),atol=3e-7,rtol=1e-6)
            for j,domain in enumerate(stream.order[:s]):matrix[s-1,j]=table[table.domain_id==domain].absolute_error.mean()
            computed=continual_metrics(matrix,s)
            assert abs(computed['average_seen_mae']-row.average_seen_mae)<1e-7
            assert abs(computed['forgetting']-row.forgetting)<1e-7
            predicted=[]
            with torch.no_grad():
                for batch in make_loader(data.subset(table.material_id.tolist()),32):
                    output=model(to_device(batch,'cuda')['graphs'])['prediction'].cpu().tolist()
                    predicted.extend(output)
            difference=float(np.max(np.abs(np.asarray(predicted)-table.prediction.to_numpy())))
            assert difference<2e-5,(method,s,difference);maximum_difference=max(maximum_difference,difference)
            assert len(trace)==row.optimizer_steps
            assert [t['current'] for t in trace]==[t['current'] for t in old_trace]
            if s>1:
                historical=json.loads((p/'historical_prototypes.json').read_text())
                previous_memory=torch.load(out/f'stage_{s-1}/replay_memory.pt',map_location='cpu',weights_only=True)['items']
                hist=HierarchicalStructuralMemory(cfg['sgcr']['prototype_count'],calibration_batches=cfg['sgcr']['calibration_batches']).fit(previous_memory,set(evidence['memory_before']))
                assert json.loads(json.dumps(hist.state_dict()))==historical
                stage_diag=diag[diag.stage_id==s].set_index('step')
                stage_selection={int(k):g for k,g in selections[selections.stage_id==s].groupby('step')} if len(selections) else {}
                for t in trace:
                    assert t['optimizer_steps']==t['backward_steps']==1 and t['autograd_calls']==2
                    assert set(t['current'])<=set(stage.train) and set(t['replay'])<=set(evidence['replay_train'])
                    assert not (set(t['current'])|set(t['replay']))&set(evidence['reserved'])
                    assert len(t['current'])==len(t['replay'])==len(set(t['replay']))
                    assert t['random_reference']==old_trace[t['step']-1]['replay']
                    q=hist.query(helper.hs(t['current']));d=stage_diag.loc[t['step']]
                    assert abs(q['novelty']-d.novelty)<1e-6 and abs(q['rho']-d.rho)<1e-6
                    assert q['nearest_prototype']==d.nearest_prototype
                    expected_projection=method==METHODS[4] and d.gradient_dot_before<0 and d.rho>0 and d.replay_gradient_norm**2>1e-8 and d.current_gradient_norm**2>1e-8
                    assert d.projection_applied==expected_projection
                    if expected_projection:
                        theoretical=d.gradient_dot_before*(1-d.rho*d.replay_gradient_norm**2/(d.replay_gradient_norm**2+1e-8))
                        assert abs(d.gradient_dot_after-theoretical)<1e-5*max(1.,d.current_gradient_norm*d.replay_gradient_norm)
                    else:assert abs(d.gradient_dot_before-d.gradient_dot_after)<1e-10
                    if method in METHODS[2:]:
                        selection=stage_selection[t['step']]
                        assert set(selection.material_id)<=set(evidence['replay_train']) and selection.material_id.is_unique
                        assert set(selection[selection.selected].material_id)==set(t['replay'])
                        assert set(selection.selection_type)<= {'local_conflict','global_coverage'}
                        assert selection.structural_relevance.between(-1e-7,1+1e-7).all()
                        distance,relevance=hist.structural_scores(selection.material_id.tolist(),q['query'])
                        assert np.allclose(distance,selection.structural_distance,atol=1e-6)
                        assert np.allclose(relevance,selection.raw_structural_relevance,atol=1e-6)
                        local=selection[selection.selection_type=='local_conflict']
                        count=int(local.selected.sum())
                        ranked=local.sort_values(['combined_priority','material_id'],ascending=[False,True]).head(count)
                        assert set(ranked.material_id)==set(local[local.selected].material_id)
                        if method in METHODS[3:]:
                            expected_priority=.5*selection.structural_relevance+.5*selection.normalized_conflict_score
                            assert np.allclose(expected_priority,selection.combined_priority,atol=1e-6)
                        n_global=(selection.selection_type=='global_coverage').sum()
                        if n_global and len(t['replay'])>=4:assert d.local_selected>0 and d.global_selected>0
            if len(quality):
                for r in quality[quality.stage_id==s].itertuples():
                    chosen=json.loads(r.selected_ids);random=json.loads(r.random_ids)
                    assert len(chosen)==len(random)==r.count and set(random)<=set(evidence['replay_train'])
                    selected=stage_selection[r.step];selected=selected[selected.selected]
                    assert abs(selected.conflict_score.mean()-r.selected_conflict)<1e-6
                    assert abs(selected.raw_structural_relevance.mean()-r.selected_structural_relevance)<1e-6
            del state,model
        final=pd.read_csv(out/'final_results.csv').iloc[0]
        assert final.optimizer_steps==metrics.optimizer_steps.sum()==810
        assert abs(final.final_avg_mae-computed['average_seen_mae'])<1e-7
        result=dict(method=method,status='passed',final_mae=float(final.final_avg_mae),test_samples=503,
                    optimizer_steps=810,max_checkpoint_prediction_difference=maximum_difference,
                    projection_count=int(diag.projection_applied.sum()))
        reports.append(result);save_json(out/'audit.json',result);print('SGCR_AUDIT',json.dumps(result),flush=True)
    save_json(root/'audit.json',dict(status='passed',protected_files_unchanged=len(protected),methods=reports,fixed_tests=503,
                                   direct_reference_unchanged=True,baseline_reproduced=True))
    print('SGCR_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
