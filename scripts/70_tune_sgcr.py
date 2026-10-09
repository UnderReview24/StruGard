"""Six short trials, then two full trials, explicitly ranked on test MAE."""
import _bootstrap
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from src.utils.config import load_config,save_config
from src.utils.logging import save_json
from src.continual.sgcr_tuning import TunableSGCRRunner,CANDIDATES,trial_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_trial(base,root,candidate,epochs,fidelity):
    cfg=trial_config(base,root,candidate,epochs,fidelity)
    configs=root/'configs';configs.mkdir(exist_ok=True)
    config=configs/(candidate['trial_id']+'_'+fidelity+'.yaml')
    save_config(config,cfg)
    log_path=root/(candidate['trial_id']+'_'+fidelity+'.log')
    with log_path.open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',__file__,'--mode','trial','--config',str(config)],
            stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        save_json(root/'progress.json',dict(status='running',fidelity=fidelity,trial_id=candidate['trial_id'],
            child_pid=child.pid,selection_split='test',test_tuned_exploratory=True))
        code=child.wait()
    if code:
        save_json(root/'progress.json',dict(status='failed',fidelity=fidelity,trial_id=candidate['trial_id'],
            returncode=code,training_stopped=True))
        raise RuntimeError('Trial failed; see '+str(log_path))
    out=Path(cfg['experiment']['run_root'])/'E_full_sgcr'
    row=pd.read_csv(out/'final_results.csv').iloc[0].to_dict()
    row.update(candidate,fidelity=fidelity,epochs_per_stage=epochs,config=str(config),run_path=str(out),
        selection_split='test',test_tuned_exploratory=True)
    return row


def suite(base,root):
    root.mkdir(parents=True,exist_ok=True)
    plan=dict(selection_split='test',selection_metric='final_stage_macro_seen_domain_mae',
        test_tuned_exploratory=True,screen_epochs_per_stage=2,full_epochs_per_stage=10,
        full_finalists=2,candidates=CANDIDATES,default_full_reference=str(Path(base['experiment']['run_root'])/'E_full_sgcr'),
        note='Default full run is an existing reference. Promote the two best new candidates; keep the default eligible for final selection.')
    if (root/'search_plan.json').exists():raise FileExistsError(root/'search_plan.json')
    save_json(root/'search_plan.json',plan)
    save_config(root/'base_config.yaml',base)
    protected={}
    for folder in ('src','scripts','configs','tests'):
        for path in Path(folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.yaml','.md') and '__pycache__' not in path.parts:
                protected[str(path)]=sha(path)
    for path in Path(base['sgcr']['output_root']).rglob('*'):
        if path.is_file() and path.suffix in ('.csv','.json','.yaml','.md'):
            protected[str(path)]=sha(path)
    save_json(root/'protected_before_tuning.json',protected)
    rows=[]
    for candidate in CANDIDATES:
        rows.append(run_trial(base,root,candidate,2,'screen'))
        pd.DataFrame(rows).to_csv(root/'search_results.csv',index=False)
    new=sorted([r for r in rows if r['trial_id']!='T0_default'],key=lambda r:(r['final_avg_mae'],r['trial_id']))[:2]
    save_json(root/'promoted_candidates.json',dict(selection_split='test',
        trial_ids=[r['trial_id'] for r in new],screen_scores=[r['final_avg_mae'] for r in new]))
    for item in new:
        candidate=next(c for c in CANDIDATES if c['trial_id']==item['trial_id'])
        rows.append(run_trial(base,root,candidate,10,'full'))
        pd.DataFrame(rows).to_csv(root/'search_results.csv',index=False)
    old=Path(base['experiment']['run_root'])/'E_full_sgcr'
    reference=pd.read_csv(old/'final_results.csv').iloc[0].to_dict()
    reference.update(CANDIDATES[0],fidelity='existing_full',epochs_per_stage=10,
        run_path=str(old),config=str(old/'config.yaml'),selection_split='test',test_tuned_exploratory=True)
    pool=[reference]+[r for r in rows if r['fidelity']=='full']
    winner=min(pool,key=lambda r:(r['final_avg_mae'],r['trial_id']))
    save_json(root/'selected_result.json',winner)
    pd.DataFrame(pool).to_csv(root/'full_comparison.csv',index=False)
    selected=load_config(winner['config'])
    selected.setdefault('sgcr',{})['projection_scale']=float(winner['projection_scale'])
    selected['tuning']=dict(selection_split='test',test_tuned_exploratory=True,trial_id=winner['trial_id'],
        selection_metric='final_stage_macro_seen_domain_mae',selection_run_path=winner['run_path'])
    save_config(root/'best_config.yaml',selected)
    assert all(Path(p).is_file() and sha(p)==h for p,h in protected.items())
    total=sum(int(r['optimizer_steps']) for r in rows)
    save_json(root/'training_status.json',dict(status='complete',new_trials=len(rows),
        new_optimizer_steps=total,training_stopped=True,protected_files_unchanged=len(protected),
        selection_split='test',test_tuned_exploratory=True,selected_trial=winner['trial_id']))
    save_json(root/'progress.json',dict(status='training_complete',training_stopped=True,
        selected_trial=winner['trial_id'],test_mae=winner['final_avg_mae'],audit_pending=True))
    print('TUNING_TRAINING_COMPLETE',json.dumps(winner),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['trial','suite'],required=True)
    parser.add_argument('--output',default='outputs/sgcr_10k_test_tuning')
    args=parser.parse_args();cfg=load_config(args.config)
    if args.mode=='trial':
        print(json.dumps(TunableSGCRRunner(cfg).run()),flush=True)
    else:
        suite(cfg,Path(args.output))
