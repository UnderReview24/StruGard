import _bootstrap
import argparse
import json
import subprocess
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.residual_method import METHODS,ResidualRunner
from src.continual.residual_diagnostics import direct_diagnostic,gradient_preflight


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['direct','preflight','suite','train'],required=True)
    p.add_argument('--method',choices=METHODS[1:]);a=p.parse_args()
    cfg=load_config('configs/experiment/residual_method.yaml');root=Path(cfg['residual']['output_root'])
    assert Path('outputs/residual_method/code_audit.txt').exists()
    if a.mode=='direct':direct_diagnostic(cfg)
    elif a.mode=='preflight':gradient_preflight(cfg)
    elif a.mode=='train':
        assert json.loads((root/'gradient_preflight_status.json').read_text())['status']=='passed'
        assert a.method;ResidualRunner(cfg,a.method).run()
    else:
        assert json.loads((root/'gradient_preflight_status.json').read_text())['status']=='passed'
        assert json.loads((root/'runs/A_direct_predictor/status.json').read_text())['status']=='passed'
        assert '63 passed' in (root/'unit_tests.log').read_text()
        done=[METHODS[0]]
        for method in METHODS[1:]:
            save_json(root/'queue_status.json',dict(status='running',active=method,completed=done,allowed_methods=METHODS))
            with (root/(method+'.log')).open('w') as log:
                subprocess.run(['.venv/bin/python',__file__,'--mode','train','--method',method],stdout=log,stderr=subprocess.STDOUT,check=True)
            done.append(method)
        save_json(root/'queue_status.json',dict(status='complete',completed=done,training_stopped=True))
