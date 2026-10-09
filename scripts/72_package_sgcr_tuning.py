"""Deliver audited test-selected tuning results; keep all weights on the server."""
import _bootstrap
import argparse
from datetime import datetime,timezone
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from src.utils.logging import save_json


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main(root):
    audit=json.loads((root/'audit.json').read_text());assert audit['status']=='passed'
    selected=json.loads((root/'selected_result.json').read_text())
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
    active=[p for p in processes if 'scripts/70_tune_sgcr.py' in p and 'grep' not in p]
    assert not active,active
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    protected=json.loads((root/'protected_before_tuning.json').read_text())
    assert all(Path(p).is_file() and sha(p)==h for p,h in protected.items())
    stop=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),training_stopped=True,
        training_processes=active,gpu_processes=gpu,new_optimizer_steps=audit['total_new_optimizer_steps'],
        additional_trials_scheduled=False,selection_split='test',test_tuned_exploratory=True)
    save_json(root/'training_stop_verification.json',stop)
    save_json(root/'progress.json',dict(status='complete',training_stopped=True,
        selected_trial=selected['trial_id'],best_test_mae=selected['final_avg_mae'],test_tuned_exploratory=True))
    best=root/'selected_artifacts';best.mkdir(exist_ok=True)
    for name in ('config.yaml','final_results.csv','stage_metrics.csv','sample_predictions.csv',
                 'initialization.json','status.json','tuning_audit.json'):
        source=Path(selected['run_path'])/name
        if source.is_file():
            (best/name).write_bytes(source.read_bytes())
    weights=set(root.rglob('*.pt'))|set(Path(selected['run_path']).rglob('*.pt'))
    save_json(root/'remote_weights_manifest.json',{str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(weights)})
    files={p.relative_to(root).as_posix():p for p in root.rglob('*')
        if p.is_file() and p.suffix not in ('.pt','.npz','.pid','.zip') and '__pycache__' not in p.parts
        and p.name not in ('packaging.log','package_verification.json')}
    for folder in ('src','scripts','configs','tests'):
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml','.md') and '__pycache__' not in p.parts:
                files['source/'+p.as_posix()]=p
    for name in ('requirements.txt','README.md','SGCR_10K_IMPLEMENTATION.md','third_party/ALIGNN_COMMIT.txt'):
        p=Path(name)
        if p.is_file():files['source/'+name]=p
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    destination=root.parent/'SGCR_10k_test_tuning_results.zip'
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,p in sorted(files.items()):archive.write(p,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(name)).hexdigest()==item['sha256'] for name,item in manifest.items())
    summary=dict(archive=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),
        verified_files=len(manifest),weights_included=False,selected_trial=selected['trial_id'])
    save_json(root/'package_verification.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/sgcr_10k_test_tuning')
    main(Path(p.parse_args().root))
