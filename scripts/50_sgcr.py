import _bootstrap
import argparse
import json
import subprocess
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.sgcr_trainer import SGCRRunner,METHODS,verify_baseline


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['baseline','run','suite'],required=True)
    parser.add_argument('--method',choices=METHODS[1:]);args=parser.parse_args()
    cfg=load_config('configs/experiment/sgcr.yaml');root=Path(cfg['sgcr']['output_root'])
    if args.mode=='baseline':
        SGCRRunner(cfg,METHODS[1]).run();verify_baseline(cfg)
    elif args.mode=='run':
        assert json.loads((root/'baseline_reproduction.json').read_text())['status']=='passed'
        assert args.method in METHODS[2:]
        SGCRRunner(cfg,args.method).run()
    else:
        assert json.loads((root/'baseline_reproduction.json').read_text())['status']=='passed'
        completed=[METHODS[1]]
        for method in METHODS[2:]:
            save_json(root/'queue_status.json',dict(status='running',completed=completed,active=method))
            with (root/(method+'.log')).open('x') as log:
                result=subprocess.run(['.venv/bin/python',__file__,'--mode','run','--method',method],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                save_json(root/'queue_status.json',dict(status='failed',completed=completed,active=method,training_stopped=True))
                raise RuntimeError(method+' failed; see its preserved log')
            completed.append(method)
        save_json(root/'queue_status.json',dict(status='complete',completed=completed,training_stopped=True))


if __name__=='__main__':main()
