import _bootstrap
import argparse
import json
import subprocess
import sys
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.sgcr_benchmark import ConfiguredSGCRRunner, RUN_METHODS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--method',choices=RUN_METHODS)
    parser.add_argument('--suite',action='store_true')
    args=parser.parse_args()
    if args.suite == bool(args.method):
        parser.error('Choose exactly one of --method or --suite')
    cfg=load_config(args.config)
    if args.method:
        print(json.dumps(ConfiguredSGCRRunner(cfg,args.method).run()),flush=True)
        return
    root=Path(cfg['sgcr']['output_root']); completed=[]
    for method in RUN_METHODS:
        save_json(root/'queue_status.json',dict(status='running',required=RUN_METHODS,
            completed=completed,active=method,training_stopped=False))
        with (root/(method+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-u',__file__,'--config',args.config,'--method',method],
                stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save_json(root/'queue_status.json',dict(status='failed',required=RUN_METHODS,
                completed=completed,active=method,training_stopped=True,returncode=result.returncode))
            raise RuntimeError(method+' failed; see preserved log')
        completed.append(method)
    save_json(root/'queue_status.json',dict(status='training_complete',required=RUN_METHODS,
        completed=completed,training_stopped=True,audit_pending=True))


if __name__=='__main__':
    main()
