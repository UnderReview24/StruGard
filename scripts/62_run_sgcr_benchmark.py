"""Serial authorized three-method run, ending after audit and report."""
import _bootstrap
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import save_json

METHODS=['A_direct','B_structural_replay','E_full_sgcr']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=load_config(args.config);root=Path(cfg['sgcr']['output_root'])
    assert json.loads((Path(cfg['experiment']['shift_output'])/'gate.json').read_text())['status']=='passed'
    sources={}
    training_scripts={'01_prepare_dataset.py','05_validate_structural_shift.py',
        '60_prepare_sgcr_benchmark.py','61_sgcr_benchmark.py','62_run_sgcr_benchmark.py'}
    for folder in ('src','scripts','configs'):
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml') and '__pycache__' not in p.parts:
                if folder=='scripts' and p.name not in training_scripts:
                    continue
                sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    save_json(root/'training_source_manifest.json',sources)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    phases=[('features','60_prepare_sgcr_benchmark.py',['--phase','features']),
            ('training','61_sgcr_benchmark.py',['--suite']),
            ('audit','63_audit_sgcr_benchmark.py',[]),
            ('report','64_report_sgcr_benchmark.py',[])]
    completed=[]
    for phase,script,extra in phases:
        log_path=root/(phase+'.log')
        with log_path.open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path('scripts')/script),'--config',args.config,*extra],
                stdout=log,stderr=subprocess.STDOUT,env=env)
            save_json(root/'progress.json',dict(status='running',phase=phase,pid=child.pid,
                completed_phases=completed,required_methods=METHODS))
            result=child.wait()
        if result:
            save_json(root/'progress.json',dict(status='failed',phase=phase,returncode=result,
                completed_phases=completed,training_stopped=True))
            raise RuntimeError(phase+' failed; see '+str(log_path))
        completed.append(phase)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    save_json(root/'training_stop_verification.json',dict(training_stopped=True,all_pipeline_children_exited=True,
        required_methods=METHODS,new_continual_optimizer_steps=sum(json.loads((root/'runs'/m/'status.json').read_text())['optimizer_calls'] for m in METHODS),
        gpu_processes=gpu,training_source_unchanged=True,additional_experiments_scheduled=False))
    save_json(root/'queue_status.json',dict(status='complete',required=METHODS,completed=METHODS,
        training_stopped=True,audit_pending=False))
    save_json(root/'progress.json',dict(status='complete',completed_phases=completed,
        required_methods=METHODS,training_stopped=True))
    print('THREE_METHOD_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    main()
