"""Serial, bounded three-method full-data comparison."""
import _bootstrap
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from src.continual.full7_protocol import METHODS, sha
from src.continual.full7_runner import Full7Runner
from src.utils.config import load_config
from src.utils.logging import save_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--method', choices=METHODS); p.add_argument('--suite', action='store_true')
    args = p.parse_args()
    assert args.suite != bool(args.method)
    cfg = load_config(args.config)
    if args.method:
        print('FULL_METHOD_COMPLETE', json.dumps(Full7Runner(cfg, args.method).run()), flush=True)
    else:
        root = Path(cfg['sgcr']['output_root']); complete = []
        protected = json.loads((root / 'protected_before_run.json').read_text())
        assert all(Path(path).exists() and sha(path) == digest for path, digest in protected.items())
        for method in METHODS:
            with (root / (method + '.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', __file__, '--config', args.config, '--method', method],
                    stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                save_json(root / 'queue_status.json', dict(status='running', required=METHODS, completed=complete,
                    active=method, child_pid=child.pid, seed=42, training_stopped=False))
                code = child.wait()
            if code:
                save_json(root / 'queue_status.json', dict(status='failed', completed=complete, active=method, returncode=code))
                raise RuntimeError(method + ' failed; inspect preserved log')
            complete.append(method)
        assert all(Path(path).exists() and sha(path) == digest for path, digest in protected.items())
        save_json(root / 'queue_status.json', dict(status='training_complete', completed=complete,
            training_stopped=True, audit_pending=True, seed=42))
