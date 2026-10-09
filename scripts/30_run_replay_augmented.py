import _bootstrap
import argparse
import json
import subprocess
from pathlib import Path
from src.continual.replay_augmented_runner import METHODS,ReplayAugmentedRunner
from src.continual.replay_legacy_runner import run_legacy
from src.utils.config import load_config
from src.utils.logging import save_json


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--suite',action='store_true')
    a=p.parse_args()
    cfg=load_config('configs/experiment/replay_augmented.yaml')
    root=Path('outputs/replay_augmented_rae')
    assert (root/'code_audit.txt').exists()
    assert json.loads((root/'reproduction/status.json').read_text())['status']=='passed'
    if a.suite:
        assert not a.smoke and not a.method
        assert json.loads((root/'smoke/D_structural_rae/status.json').read_text())['status']=='passed'
        assert (root/'preflight_tests.log').exists()
        completed=[]
        for method in METHODS:
            save_json(root/'queue_status.json',dict(status='running',active=method,completed=completed,allowed_methods=METHODS))
            if (Path(cfg['experiment']['run_root'])/method).exists():raise FileExistsError(method)
            with (root/(method+'.log')).open('w') as log:
                subprocess.run(['.venv/bin/python',__file__,'--method',method],stdout=log,stderr=subprocess.STDOUT,check=True)
            completed.append(method)
        save_json(root/'queue_status.json',dict(status='complete',completed=completed,training_stopped=True))
    else:
        assert a.method
        result=run_legacy(cfg) if a.method=='B_old_no_replay' else ReplayAugmentedRunner(cfg,a.method,a.smoke).run()
        print('RUN_COMPLETE',json.dumps(result),flush=True)
