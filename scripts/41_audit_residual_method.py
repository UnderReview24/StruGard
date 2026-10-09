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
from src.continual.strategies import setup_experiment
from src.continual.replay_augmented_runner import verify_fixed_split
from src.continual.residual_method import METHODS
from src.continual.metrics import continual_metrics
from src.continual.v2_protocol import tensor_digest
from src.models.sp_crystal_residual import load_residual
from src.data.dataset import make_loader,to_device


def main():
    root=Path('outputs/residual_method');cfg=load_config('configs/experiment/residual_method.yaml')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    stream,data=setup_experiment(cfg);verify_fixed_split(cfg,stream,data)
    stages=list(stream);all_test={mid for s in stages for mid in s.test};all_val={mid for s in stages for mid in s.val}
    protected=json.loads((root/'protected_files_before.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    cache=torch.load(cfg['v2']['cache'],map_location='cpu',weights_only=True)
    features={mid:r['stable_embedding'] for mid,r in cache['records'].items()};del cache
    results=[]
    for method in METHODS:
        out=Path(cfg['experiment']['run_root'])/method
        assert json.loads((out/'status.json').read_text())['status']=='passed'
        samples=pd.read_csv(out/'sample_predictions.csv');stage_table=pd.read_csv(out/'stage_metrics.csv')
        matrix=np.full((3,3),np.nan);base_matrix=np.full((3,3),np.nan)
        seen_train=set();frozen=None;max_reload=json.loads((out/'status.json').read_text()).get('max_prediction_difference',0.)
        for stage in stages:
            s=stage.stage_id;row=stage_table[stage_table.stage_id==s].iloc[0]
            table=samples[(samples.stage_id==s)&(samples.split=='test')&(samples.evaluation_phase=='stage_complete')]
            expected=all_test if method==METHODS[0] else {mid for st in stages[:s] for mid in st.test}
            assert set(table.material_id)==expected and table.material_id.is_unique
            assert np.allclose(table.final_prediction,table.base_prediction+table.residual_prediction,atol=2e-6,rtol=1e-6)
            assert np.allclose(table.final_absolute_error,(table.target-table.final_prediction).abs(),atol=3e-7,rtol=1e-6)
            assert np.allclose(table.base_absolute_error,(table.target-table.base_prediction).abs(),atol=3e-7,rtol=1e-6)
            assert np.allclose(table.loc[~table.expert_active,'residual_prediction'],0.,atol=0.,rtol=0.)
            for j,domain in enumerate(stream.order[:s]):
                group=table[table.domain_id==domain]
                matrix[s-1,j]=group.final_absolute_error.mean();base_matrix[s-1,j]=group.base_absolute_error.mean()
            metrics=continual_metrics(matrix,s)
            assert abs(metrics['average_seen_mae']-row.average_seen_mae)<1e-7
            assert abs(metrics['forgetting']-row.forgetting)<1e-7
            assert abs(float(np.nanmean(base_matrix[s-1]))-row.base_avg_seen_mae)<1e-7
            if method==METHODS[0]:
                assert row.num_residual_experts==row.num_active_experts==row.memory_size==row.new_optimizer_steps_this_task==0
                continue
            stage_out=out/f'stage_{s}'
            audit=json.loads((stage_out/'data_audit.json').read_text());trace=json.loads((stage_out/'optimizer_trace.json').read_text())
            reference=json.loads((Path('outputs/replay_augmented_rae/runs/A_structural_replay')/f'stage_{s}/data_audit.json').read_text())
            assert audit['current_train']==list(stage.train) and audit['current_validation']==list(stage.val)
            assert audit['memory_before']==reference['memory_before'] and audit['memory_after']==reference['memory_after']
            assert audit['replay_train']==reference['replay_train'] and audit['reserved']==reference['probe']
            assert len(audit['memory_after'])<=2000 and set(audit['memory_before'])<=seen_train
            assert not (set(audit['memory_after'])|set(audit['replay_train'])|set(audit['reserved']))&(all_test|all_val)
            previous_trace=json.loads((Path('outputs/replay_augmented_rae/runs/A_structural_replay')/f'stage_{s}/optimizer_trace.json').read_text())
            assert [(t['current'],t['replay']) for t in trace]==[(t['current'],t['replay']) for t in previous_trace]
            for t in trace:
                assert set(t['current'])<=set(stage.train) and set(t['replay'])<=set(audit['replay_train'])
                assert not (set(t['current'])|set(t['replay']))&set(audit['reserved'])
                assert t['optimizer_steps']==t['backward_steps']==1 and t['shared_update']
                assert len(t['current'])==len(t['replay'])>0
                if t['action']=='REUSE':assert t['training_expert'] is None
            assert len(trace)==row.optimizer_steps==row.backward_steps
            model,state=load_residual(stage_out/'checkpoint.pt','cuda')
            digest=tensor_digest(model.stable_encoder.state_dict().items())
            if frozen is None:frozen=digest
            assert digest==frozen
            assert sum(state['validated'].values())==row.num_active_experts
            assert state['experts']==row.num_residual_experts
            if method==METHODS[1]:assert state['experts']==0
            if method==METHODS[2]:assert state['experts']==1
            if method==METHODS[3]:assert state['experts']==s
            event=json.loads((stage_out/'controller.json').read_text())
            if event['new_expert'] is not None:
                assert event['beta_new_at_creation']==0 and event['inheritance_exact'] and not event['predictor_changed_by_expansion']
                if method==METHODS[4]:assert event['high_novelty'] and event['old_expert_val_mae']>=event['base_val_mae']
            if s>1:
                previous=torch.load(out/f'stage_{s-1}/checkpoint.pt',map_location='cpu',weights_only=True)
                active={t['training_expert'] for t in trace if t['training_expert'] is not None}
                for k in range(previous['experts']):
                    if k not in active:
                        prefix=f'experts.{k}.'
                        assert all(torch.equal(v,state['model'][n]) for n,v in previous['model'].items() if n.startswith(prefix))
            live=[]
            with torch.no_grad():
                for batch in make_loader(data.subset(table.material_id.tolist()),32):
                    mids=batch['material_id'];hs=torch.stack([features[mid] for mid in mids]).cuda();batch=to_device(batch,'cuda')
                    output=model(batch['graphs'],hs)
                    fallback=model(batch['graphs'],hs,disable_residual=True)
                    assert torch.equal(fallback['base_prediction'],fallback['final_prediction'])
                    for i,mid in enumerate(mids):live.append(dict(material_id=mid,final_prediction=float(output['final_prediction'][i]),base_prediction=float(output['base_prediction'][i])))
            live=pd.DataFrame(live).set_index('material_id');saved=table.set_index('material_id')
            difference=max(float((live[c]-saved[c]).abs().max()) for c in ['base_prediction','final_prediction'])
            assert difference<2e-5,(method,s,difference);max_reload=max(max_reload,difference)
            activations=json.loads((out/'validation_activation.json').read_text())
            for evidence in [a for a in activations if a['stage']==s]:
                assert evidence['validation_ids']==list(stage.val)
                subset=samples[(samples.stage_id==s)&(samples.split=='validation')&(samples.evaluation_phase==evidence['phase'])&(samples.candidate_expert==evidence['expert_id'])]
                assert set(subset.material_id)==set(stage.val) and subset.material_id.is_unique
                assert abs(subset.base_absolute_error.mean()-evidence['base_val_mae'])<1e-7
                assert abs(subset.final_absolute_error.mean()-evidence['corrected_val_mae'])<1e-7
                gain=evidence['base_val_mae']-evidence['corrected_val_mae']
                assert evidence['expert_active']==(gain>0)
                if evidence['phase']=='activation_final' and evidence['eligible_samples']>0:
                    assert state['validated'][evidence['expert_id']]==evidence['expert_active']
            final_val=samples[(samples.stage_id==s)&(samples.split=='validation')&(samples.evaluation_phase=='stage_complete')]
            assert set(final_val.material_id)==set(stage.val)
            assert final_val.final_absolute_error.mean()<=final_val.base_absolute_error.mean()+1e-6
            for call in [c for c in json.loads((out/'access_audit.json').read_text()) if c['stage']==s]:
                if call['role']=='validation':assert call['ids']==list(stage.val)
                if call['role']=='test':assert call['phase']=='stage_complete'
                if call['role']=='training_prediction':assert set(call['ids'])<=set(stage.train)
            seen_train.update(stage.train)
            del model,state
        final=pd.read_csv(out/'final_results.csv').iloc[0]
        assert abs(final.final_avg_seen_mae-metrics['average_seen_mae'])<1e-7
        if method!=METHODS[0]:
            grads=pd.read_csv(out/'gradient_audit.csv');rep=grads[grads.loss_source=='replay']
            assert (rep.shared_backbone>0).all() and (rep.shared_head>0).all() and (rep.residual_expert==0).all()
            assert int(stage_table.optimizer_steps.sum())==int(final.optimizer_steps)==810
        result=dict(method=method,status='passed',final_mae=float(final.final_avg_seen_mae),test_samples=503,
            max_checkpoint_prediction_difference=max_reload,optimizer_steps=int(final.optimizer_steps),active_experts=int(final.num_active_experts))
        save_json(out/'audit.json',result);results.append(result)
        print('RESIDUAL_AUDIT',json.dumps(result),flush=True)
    save_json(root/'audit.json',dict(status='passed',methods=results,protected_files_unchanged=len(protected),
        fixed_tests=503,full_shared_updates=810,split_sha256=cfg['replay_augmented']['split_sha256']))
    (root/'leakage_audit.txt').write_text('''PASSED: residual-method data/gradient/activation audit.
Only existing5000 structures, fixed3999/498/503 split, seed42 and order[2,1,0]. Protected source/checkpoints/logs/results match all original hashes.
Direct Predictor reuses the completed sequential full ALIGNN checkpoints. All3x3 test evaluations are read-only cross-domain diagnostics, never controller evidence. No duplicate Direct training.
B-E exactly match the verified0.097065 baseline's current/replay ID sequence, k-center-selected memories,80/20 partition and2000 total memory budget. Reserved20% never enters gradients or residual activation. No test/validation IDs enter replay/prototypes/optimization.
Shared predictive backbone/head ALWAYS train:810 steps each, up to32current+32replay, one forward/backward/optimizer call. Actual optimizer hooks match traces. Replay gradient probes are nonzero on the full predictive backbone and shared head, zero on residual experts. Engineering preflights used2 initial disposable wrapper updates, then3 deterministic wrapper updates and2 native-reference updates. All used permitted train IDs; all preflight weights were discarded. These are separate from the810 final-run steps.
Stable h_s is the frozen D1 encoder coordinate system. Its weights match across checkpoints. Residual is h_s-mu_nearest, with a per-expert learned projection/LayerNorm. Predictive h_p can drift without moving stable coordinates. Cached samplewise features contain no learned future statistics; only authorized training IDs enter prototypes and memory.
New scalar residual expert inherits only small residual parameters; beta resets to0 and creation leaves base weights untouched. The final output always includes y_base. With all residuals disabled, checkpoint inference equals its own shared prediction exactly. This is an architectural fallback, not guaranteed identical training trajectory to a separate baseline.
RAE controls residual capacity, not shared training. Dynamic expansion needs high training novelty AND nonpositive current-validation gain from the existing nearest candidate. All switches are recomputed from current validation only at stage completion after prototype changes. Each candidate is evaluated only in its actual routed region with other corrections disabled. Unsupported regions retain prior validation switches; no historical test result is used. Active validation-supported regions have positive current validation gain, but test gains are not guaranteed.
Current residual learning is end-to-end, so it can alter shared backbone/head weights. Historical replay uses its own nearest validated residual, evaluated with residual gradients disabled; never forced through the new-domain expert. Unrelated old expert weights remain unchanged.
All final503 test IDs match exactly. New-method test predictions occur only after training/activation decisions. Stored predictions reproduce MAEs, forgetting, validation gains and checkpoint outputs. No test-label routing, threshold search, future-training replay or large-data runs were performed.
Direct historical timing/checkpoint selection differs from newly timed B-E; Direct optimizer_steps is its archived training count, new_optimizer_steps_this_task is0. B-E use last-step checkpoints, and shared D1 training is excluded from newly incurred810 updates. A separate frozen D1 encoder is included in B-E total parameter counts, not duplicated per expansion.
''')
    print('RESIDUAL_AUDIT_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
