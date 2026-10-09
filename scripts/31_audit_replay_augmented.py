"""Independent saved-ID/metric/weight/checkpoint audit; never retrains."""
import _bootstrap
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.continual.replay_augmented_runner import METHODS,verify_fixed_split
from src.continual.strategies import setup_experiment
from src.continual.metrics import continual_metrics
from src.continual.v2_protocol import collate_prefix,stable_digest
from src.continual.replay_buffer import StructuralReplayBuffer
from src.continual.static_training import evaluate
from src.models.alignn_wrapper import make_alignn
from src.models.v2 import load_v2
from src.models.factory import load_sp_checkpoint
from src.continual.diagnostics import evaluate_sp
from src.data.embeddings import extract_embeddings
from src.utils.config import load_config
from src.utils.logging import save_json


def main():
    root=Path('outputs/replay_augmented_rae')
    cfg=load_config('configs/experiment/replay_augmented.yaml')
    stream,data=setup_experiment(cfg);verify_fixed_split(cfg,stream,data)
    stages=list(stream)
    test_ids={mid for s in stages for mid in s.test}
    validation_ids={mid for s in stages for mid in s.val}
    features=torch.load(cfg['v2']['cache'],map_location='cpu',weights_only=True)['records']
    protected=json.loads((root/'protected_files_before.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    results=[]
    for method in METHODS:
        out=Path(cfg['experiment']['run_root'])/method
        assert json.loads((out/'status.json').read_text())['status']=='passed'
        table=pd.read_csv(out/'stage_metrics.csv')
        predictions=pd.read_csv(out/'sample_predictions.csv')
        matrix=np.full((3,3),np.nan)
        seen_train=set();previous_weights=None;frozen_reference=None
        original_memory=StructuralReplayBuffer(2000,6000,42,'random' if method==METHODS[2] else 'kcenter')
        legacy_router=[]
        max_reload=0.;audited_steps=0
        for stage in stages:
            s=stage.stage_id;stage_out=out/f'stage_{s}'
            row=table[table.stage_id==s].iloc[0]
            saved=predictions[(predictions.stage_id==s)&(predictions.split=='test')]
            expected={mid for st in stages[:s] for mid in st.test}
            assert set(saved.material_id)==expected and saved.material_id.is_unique
            assert np.allclose(saved.absolute_error,(saved.target-saved.prediction).abs(),atol=3e-7,rtol=1e-6)
            for j,domain in enumerate(stream.order[:s]):
                matrix[s-1,j]=saved[saved.domain_id==domain].absolute_error.mean()
            metrics=continual_metrics(matrix,s)
            assert abs(metrics['average_seen_mae']-row.average_seen_domain_mae)<1e-7
            assert abs(metrics['forgetting']-row.forgetting)<1e-7
            ids=saved.material_id.tolist()
            if method==METHODS[1]:
                trace=json.loads((stage_out/'compute_trace.json').read_text())
                assert trace['optimizer_calls']==row.optimizer_steps==trace['backward_calls']
                assert all(set(b)<=set(stage.train) for b in trace['current_batches'])
                assert sum(map(len,trace['current_batches']))==row.current_sample_exposures
                if s>1:
                    curve=pd.read_csv(stage_out/'training_curve.csv')
                    assert int(curve.updated_batches.sum())==row.optimizer_steps
                model,state=load_sp_checkpoint(stage_out/'best.pt','cuda')
                _,live=evaluate_sp(model,data,ids,32)
                assert state['replay']['memory_size']==0
                before,_=load_sp_checkpoint(out/f'stage_{max(1,s-1)}/best.pt','cuda')
                h=extract_embeddings(before.stable_encoder,data,stage.train,32,cache_dir='data/embeddings/rae_cache')
                route=before.prototype_memory.nearest(h)
                score=route['nearest_distance']
                legacy_router.append(dict(method=method,stage_id=s,decision='LEGACY_SAMPLE_ROUTER',
                    expansion_reason='none',parent_expert=None,new_expert_id=None,
                    mean_novelty=float(score.mean()),novelty_q50=float(torch.quantile(score,.5)),
                    novelty_q90=float(torch.quantile(score,.9)),novelty_q99=float(torch.quantile(score,.99)),
                    nearest_expert_distribution=json.dumps({str(k):int((route['nearest_expert']==k).sum()) for k in route['nearest_expert'].unique().tolist()}),
                    novelty_statistics_source='current training arrival geometry reconstructed from preceding checkpoint'))
                assert int(row.num_experts)==1
                del before
            else:
                audit=json.loads((stage_out/'data_audit.json').read_text())
                trace=json.loads((stage_out/'optimizer_trace.json').read_text())
                event=json.loads((stage_out/'controller.json').read_text())
                mem=torch.load(stage_out/'replay_memory.pt',map_location='cpu',weights_only=True)
                assert audit['current_train']==list(stage.train) and audit['current_validation']==list(stage.val)
                assert set(audit['memory_before'])<=seen_train
                train,probe=set(audit['replay_train']),set(audit['probe'])
                assert train.isdisjoint(probe) and train|probe==set(audit['memory_before'])
                assert len(train)+len(probe)<=2000 and len(mem['items'])==row.memory_size<=2000
                assert not (set(mem['items']) | train | probe) & (test_ids | validation_ids)
                assert sum(t['optimizer_steps'] for t in trace)==row.optimizer_steps
                assert sum(t['backward_steps'] for t in trace)==row.backward_steps==row.optimizer_steps
                assert sum(t['current_exposures'] for t in trace)==row.current_sample_exposures
                assert sum(t['replay_exposures'] for t in trace)==row.replay_sample_exposures
                for t in trace:
                    assert set(t['current'])<=set(stage.train) and set(t['replay'])<=train
                    assert not (set(t['current'])|set(t['replay']))&probe
                    if t['action']=='REUSE':assert t['optimizer_steps']==t['backward_steps']==t['replay_exposures']==0 and not t['replay']
                    else:assert t['optimizer_steps']==t['backward_steps']==1 and len(t['replay'])==len(t['current'])>0
                # Independently rerun the unchanged structural/random memory policy.
                # Source vectors are read only for the current training partition.
                source={}
                for file in Path(cfg['replay_augmented']['selection_embedding_cache']).glob('*.npz'):
                    z=np.load(file)
                    if z['material_id'].tolist()==list(stage.train):
                        source={mid:torch.from_numpy(h.copy()) for mid,h in zip(z['material_id'].tolist(),z['embedding'])};break
                assert set(source)==set(stage.train)
                original_memory.update([dict(material_id=mid,graph_reference=mid,target=float(data.metadata.at[mid,'target']),
                    expert_id=0,domain_id=stage.domain_id,stable_embedding=source[mid],stored_prediction=0.) for mid in stage.train])
                assert original_memory.ids==sorted(mem['items'])==audit['memory_after']
                if s>1:
                    sample=predictions[predictions.stage_id==s]
                    evidence={}
                    for role,prefix,expected_ids in [('validation','L_new',set(stage.val)),('retention_probe','L_old',probe)]:
                        for phase,suffix in [('before_adapt','before'),('after_adapt','after')]:
                            p=sample[(sample.split==role)&(sample.evaluation_phase==phase)]
                            assert set(p.material_id)==expected_ids and p.material_id.is_unique
                            value=float(p.absolute_error.mean())
                            key=prefix+'_'+suffix+('_adapt' if role=='validation' else '')
                            assert abs(value-event[key])<1e-7
                            evidence[key]=value
                    assert abs(event['Delta_old']-(event['L_old_after']-event['L_old_before']))<1e-10
                    if method in [METHODS[2],METHODS[3]] and event['decision']=='EXPAND':
                        assert event['high_novelty'] and (event['new_domain_failed'] or event['retention_failed'])
                        assert len([t for t in trace if t['action']=='ADAPT'])>0
                if method==METHODS[0]:
                    state=torch.load(stage_out/'checkpoint.pt',map_location='cpu',weights_only=True)
                    model=make_alignn(state['model_config']).cuda();model.load_state_dict(state['model'])
                    _,live=evaluate(model,data.subset(ids),32)
                else:
                    model,optimizer,state=load_v2(stage_out/'checkpoint.pt','cuda')
                    digest=stable_digest(model.encoder)
                    if frozen_reference is None:frozen_reference=digest
                    assert frozen_reference==digest
                    if event['decision']=='REUSE':
                        assert row.optimizer_steps==0
                        assert previous_weights is not None and all(torch.equal(v,previous_weights[k]) for k,v in state['model'].items())
                    values=[]
                    with torch.no_grad():
                        for start in range(0,len(ids),32):
                            mids=ids[start:start+32]
                            pred=model(collate_prefix(features,mids))['prediction'].cpu().tolist()
                            values.extend(dict(material_id=mid,prediction=p) for mid,p in zip(mids,pred))
                    live=pd.DataFrame(values)
                previous_weights=state['model']
                accesses=json.loads((out/'access_audit.json').read_text())
                for call in [c for c in accesses if c['stage']==s]:
                    if call['controller_input']:
                        assert call['role'] in ['validation','retention_probe']
                        assert set(call['ids'])<=set(stage.val)|probe
                    if call['role']=='test':assert call['phase']=='stage_complete'
            difference=float((live.set_index('material_id').prediction-saved.set_index('material_id').prediction).abs().max())
            assert difference<2e-5,(method,s,difference)
            max_reload=max(max_reload,difference)
            audited_steps+=int(row.optimizer_steps)
            seen_train.update(stage.train)
            del model,state
        final=pd.read_csv(out/'final_results.csv').iloc[0]
        if legacy_router:pd.DataFrame(legacy_router).to_csv(out/'router_diagnostics.csv',index=False)
        assert abs(final.final_avg_mae-metrics['average_seen_mae'])<1e-7
        assert int(final.optimizer_steps)==audited_steps
        result=dict(method=method,status='passed',fixed_final_tests=503,optimizer_steps=audited_steps,
                    max_checkpoint_prediction_difference=max_reload,final_avg_mae=float(final.final_avg_mae))
        save_json(out/'replay_audit.json',result);results.append(result)
        print('AUDIT_METHOD',json.dumps(result),flush=True)
    cdir=root/'runs'/METHODS[2];ddir=root/'runs'/METHODS[3]
    c0=torch.load(cdir/'stage_1/checkpoint.pt',map_location='cpu',weights_only=True)['model']
    d0=torch.load(ddir/'stage_1/checkpoint.pt',map_location='cpu',weights_only=True)['model']
    initial_equal=all(torch.equal(v,d0[k]) for k,v in c0.items())
    exposure_equal=True
    for s in [2,3]:
        ctrace=json.loads((cdir/f'stage_{s}/optimizer_trace.json').read_text())
        dtrace=json.loads((ddir/f'stage_{s}/optimizer_trace.json').read_text())
        keys=['action','active_expert','current','replay','optimizer_steps']
        exposure_equal &= [{k:r[k] for k in keys} for r in ctrace]==[{k:r[k] for k in keys} for r in dtrace]
    save_json(root/'selection_identifiability.json',dict(initial_weights_identical=initial_equal,
        random_and_structural_training_exposures_identical=exposure_equal,
        interpretation='When exposures/actions are identical, numerical MAE differences do not identify a memory-selection benefit. The first memory fits all D1 samples, and later selection only differs after the final trained stage if D3 is REUSE.'))
    save_json(root/'audit.json',dict(status='passed',methods=results,protected_files_unchanged=len(protected),
        split_hash=cfg['replay_augmented']['split_sha256'],assignment_hash=cfg['replay_augmented']['assignment_sha256']))
    text='''LEAKAGE AND IMPLEMENTATION AUDIT: PASSED
Fixed seed=42, order=[2,1,0], 5000 structures, 3999 train, 498 validation, 503 test. Original split/assignment SHA256 and all protected files match.
Routing: per-stage current training h_s and previously stored prototypes only. Frozen D1 coordinates are reused. Per-sample immutable prefix caches contain no targets/statistical fitting; only authorized IDs are selected at each call.
Memory: EXACT existing StructuralReplayBuffer/RandomReplayBuffer policies independently replayed with original D1 selection embeddings, expert_id=0/domain groups, budget2000, candidate6000. All saved memory IDs match reconstruction.
Probe: deterministic 80/20 historical memory partition before each stage. Both parts count in the same budget. Probe IDs are absent from every current-stage ADAPT/EXPAND gradient batch; historical training before entry to a probe is permitted. No old structure outside retained memory is reopened for training.
ADAPT/EXPAND: one combined forward, Huber(current)+1.0*Huber(replay), one backward and optimizer update, no distillation. Runtime optimizer hooks agree with saved traces. Frozen stable parameters and unrelated expert invariance are asserted during training and stable parameters rechecked from checkpoints.
Decision evidence: exact current validation IDs with forced selected expert for plasticity; exact fixed historical probe IDs using actual nearest-expert routing for retention. Stored before/after sample MAEs independently reproduce controller scalars. Gamma_new=gamma_old=0.10; no threshold search. High novelty plus failure of either criterion is required by the main method; no-expand and always-expand are declared controls.
REUSE: one inference pass, no current target use in the training operation, zero replay optimization, zero optimizer/backward calls. Entire model state identical to preceding checkpoint. Current train labels may subsequently enter the memory for future stages, as requested.
Tests: new A/C/D/E/F read tests only after decisions and training. All 503 final IDs exactly match existing tests; tests never enter gradients, prototypes, replay or controller inputs. Legacy B is unchanged: it also computes pre-arrival test diagnostics, but neither returned test labels nor metrics feed its training-only geometry router. This inherited read-only diagnostic is explicitly separated from controller inputs.
Historical reproduction: saved-checkpoint re-evaluation reproduced 0.10206509 and 0.18195963, not an independent old-protocol training replication. New A uses the requested one-step joint budget and 80/20 partition and is a distinct experiment.
Known comparison limits: B uses the old frozen-backbone architecture and validation-selected early stopping. New A/C/D/E/F use 10 passes and last-step checkpoints; new RAE permits slow-tail adaptation. B-to-D therefore does not isolate replay alone. Random-to-structural and D-to-E do use the same new framework. CUDA scatter is not forced bitwise deterministic; one seed is diagnostic evidence only.
Compute excludes the common already-trained D1 initialization and reports actual subsequent calls. New training-stage wall time includes routing/probe/memory work but excludes final tests/checkpoint serialization. Legacy timing covers its fit loop including validation. Frozen D1 selection is represented by an existing cache; baseline auxiliary selection encoder is not kept resident in this new runner.
All A-F runs complete. No larger datasets, new seeds, threshold tuning or optional maintenance run was launched.
'''
    (root/'leakage_audit.txt').write_text(text)
    print('AUDIT_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
