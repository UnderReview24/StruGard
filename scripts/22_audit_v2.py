"""Independently recompute metrics, replay exposures, controller gates and reloads."""
import _bootstrap
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.continual.metrics import continual_metrics
from src.continual.strategies import setup_experiment
from src.continual.v2_protocol import collate_prefix,schedule_digest
from src.continual.v2_controller import NeedAwareController
from src.models.v2 import load_v2,V2PrototypeMemory
from src.models.alignn_wrapper import make_alignn
from src.data.dataset import make_loader
from src.utils.config import load_config
from src.utils.logging import save_json


def audit(path,protocol,features,data,split_info):
    cfg=load_config(path/'config.yaml')
    assert cfg==protocol['config']
    status=json.loads((path/'status.json').read_text())
    assert status['status']=='passed'
    variant=status['variant']
    access=json.loads((path/'access_audit.json').read_text())
    records=pd.read_csv(path/'stage_metrics.csv')
    matrix=np.full((3,3),np.nan)
    train_seen=set();test_seen=[];frozen_reference=None;max_reload_error=0.
    for i,(plan,entry) in enumerate(zip(protocol['stages'],access)):
        stage=i+1;out=path/f'stage_{stage}'
        assert entry['training_ids']==plan['train'] and entry['validation_ids']==plan['validation']
        assert entry['replay_ids']==plan['replay_before']
        assert set(entry['replay_ids'])<=train_seen
        assert set(plan['train']).isdisjoint(plan['validation']+plan['test'])
        train_seen.update(plan['train']);test_seen+=plan['test']
        assert set(entry['memory_ids'])<=train_seen and len(entry['memory_ids'])<=cfg['replay']['size']
        assert set(entry['test_ids'])==set(test_seen)
        assert entry['test_labels_used_for_controller'] is False
        if stage>1:
            trace=json.loads((out/'optimizer_trace.json').read_text())
            actual=[{key:t[key] for key in ['epoch','current','replay']} for t in trace]
            assert actual==plan['batches'] and schedule_digest(actual)==plan['exposure_digest']
            assert len(trace)==plan['optimizer_slots']==int(records.iloc[i].optimizer_steps)
            assert sum(len(t['current']) for t in trace)==int(records.iloc[i].current_exposures)
            assert sum(len(t['replay']) for t in trace)==int(records.iloc[i].replay_exposures)
            assert sum(t['effective_update'] for t in trace)==int(records.iloc[i].effective_optimizer_steps)
            event=json.loads((out/'controller.json').read_text())
            assert event['controller_validation_ids']==plan['validation'] and not event['test_labels_used']
            prior=torch.load(path/f'stage_{stage-1}/checkpoint.pt',map_location='cpu',weights_only=True)
            prototypes=V2PrototypeMemory();prototypes.load_state_dict(prior['prototypes'])
            h=torch.stack([features[mid]['stable_embedding'] for mid in plan['train']])
            inspected=NeedAwareController(cfg['v2'],variant).inspect(h,prototypes)
            assert inspected['nearest_expert']==event['nearest_expert']
            assert abs(inspected['novelty_mean']-event['novelty'])<1e-6
            historical=prototypes.historical_mae(event['nearest_expert'])
            assert abs(historical-event['L_hist'])<1e-9
            expected=NeedAwareController(cfg['v2'],variant).after_adaptation(inspected,historical,
                event['L_after_adapt'],plan['validation'],plan['validation'])
            assert event['decision']==('ADAPT' if variant=='structural_replay' else expected['decision'])
            if variant in {'full','no_shift','no_slow'} and event['new_expert_id'] is not None:
                assert event['high_novelty'] and event['L_after_adapt']>event['failure_limit']
        table=pd.read_csv(out/'predictions.csv')
        assert table.material_id.is_unique and set(table.material_id)==set(test_seen)
        targets=data.metadata.loc[table.material_id,'target'].to_numpy()
        np.testing.assert_allclose(table.target,targets,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(table.absolute_error,np.abs(table.prediction-table.target),atol=2e-6)
        for j,domain in enumerate(protocol['order'][:stage]):
            matrix[i,j]=float(table[table.domain_id==domain].absolute_error.mean())
        metric=continual_metrics(matrix,stage)
        for key in ['current_mae','average_seen_mae','forgetting']:
            assert abs(metric[key]-float(records.iloc[i][key]))<1e-8
        parsed=json.loads(records.iloc[i].per_domain_mae)
        for domain,value in parsed.items():
            assert abs(float(table[table.domain_id==int(domain)].absolute_error.mean())-value)<1e-8
        # Reload actual dynamically sized checkpoints. Cover each routed expert.
        selected=table.groupby('expert_id',sort=True).head(4).material_id.tolist()
        saved=table.set_index('material_id').loc[selected,'prediction'].to_numpy()
        if variant=='structural_replay':
            state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
            model=make_alignn(state['model_config']).eval();model.load_state_dict(state['model'])
            with torch.no_grad():
                batch=next(iter(make_loader(data.subset(selected),len(selected))))
                predicted=model(batch['graphs'])['out'].numpy()
        else:
            model,optimizer,state=load_v2(out/'checkpoint.pt')
            names=['encoder.encoder.'+n for n in split_info['stable']['names']]
            names+=['encoder.reference_tail.'+n for n in split_info['reference']['names']]
            frozen={n:state['model'][n] for n in names}
            if frozen_reference is None:frozen_reference=frozen
            else:
                for n,v in frozen.items():assert torch.equal(v,frozen_reference[n])
            assert len(model.experts)==int(records.iloc[i].number_of_experts)
            with torch.no_grad():predicted=model(collate_prefix(features,selected,'cpu'))['prediction'].numpy()
        np.testing.assert_allclose(predicted,saved,atol=2e-5,rtol=2e-5)
        max_reload_error=max(max_reload_error,float(np.max(np.abs(predicted-saved))))
    for call in json.loads((path/'validation_access.json').read_text()):
        assert call['ids']==protocol['stages'][call['stage']-1]['validation']
    saved_matrix=pd.read_csv(path/'continual_matrix.csv').drop(columns='stage').to_numpy()
    np.testing.assert_allclose(saved_matrix,matrix,equal_nan=True,atol=1e-8)
    final=pd.read_csv(path/'final_results.csv').iloc[0]
    assert abs(float(final.average_seen_mae)-float(records.iloc[-1].average_seen_mae))<1e-9
    report=dict(status='passed',variant=variant,stages=3,fixed_test_samples=len(test_seen),
        max_checkpoint_prediction_error=max_reload_error,optimizer_steps=int(records.optimizer_steps.sum()),
        effective_optimizer_steps=int(records.effective_optimizer_steps.sum()),
        current_exposures=int(records.current_exposures.sum()),replay_exposures=int(records.replay_exposures.sum()),
        checks=['fixed test IDs','current/past-only optimization','identical per-step exposure schedule',
                'validation-only need decisions','prototype geometry recomputed','saved MAE/forgetting recomputed',
                'real checkpoint inference','frozen shared prefix and reference tail invariance'])
    save_json(path/'audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--config',default='configs/experiment/v2.yaml');a=p.parse_args()
    cfg=load_config(a.config);stream,data=setup_experiment(cfg)
    protocol=torch.load(cfg['v2']['protocol'],map_location='cpu',weights_only=True)
    features=torch.load(cfg['v2']['cache'],map_location='cpu',weights_only=True)['records']
    split_info=json.loads((Path(cfg['experiment']['run_root'])/'parameter_split.json').read_text())
    for run in a.runs:audit(Path(run),protocol,features,data,split_info)
