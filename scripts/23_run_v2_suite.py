"""Run exactly the seven requested 5k variants serially, then stop."""
import _bootstrap
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from src.continual.v2_trainer import VARIANTS
from src.utils.config import load_config
from src.utils.logging import save_json


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/experiment/v2.yaml')
    p.add_argument('--only',nargs='+',choices=VARIANTS)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    cfg=load_config(a.config);root=Path(cfg['experiment']['run_root'])
    names=a.only or VARIANTS
    if a.dry_run:
        print(json.dumps(dict(variants=names,config=a.config,larger_experiments=False),indent=2))
        raise SystemExit(0)
    root.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),variants=names,completed=[],larger_experiments=False,
                started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for name in names:
            status.update(active_variant=name,phase='training')
            save_json(root/'queue_status.json',status)
            done=root/name/'status.json'
            if not done.exists() or json.loads(done.read_text()).get('status')!='passed':
                subprocess.run([sys.executable,'-u','scripts/21_train_v2.py','--config',a.config,'--variant',name],check=True)
            status['phase']='audit';save_json(root/'queue_status.json',status)
            subprocess.run([sys.executable,'-u','scripts/22_audit_v2.py','--config',a.config,str(root/name)],check=True)
            status['completed'].append(name)
            print('V2_VARIANT_AUDITED',name,flush=True)
        status.update(status='complete',phase='stopped',active_variant=None,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        status.update(status='failed',error=str(error))
        raise
    finally:save_json(root/'queue_status.json',status)
