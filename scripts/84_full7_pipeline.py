"""Run the authorized seed-42 pipeline once, preserving logs and failures."""
import _bootstrap
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from src.continual.full7_protocol import sha
from src.utils.config import load_config
from src.utils.logging import save_json


def launch(root, phase, argv):
    with (root / (phase + '.log')).open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', *argv], stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        save_json(root / 'pipeline_status.json', dict(status='running', phase=phase, child_pid=child.pid,
            requested_methods=['A_direct', 'B_online_ewc', 'E_full_sgcr'], requested_seeds=[42]))
        code = child.wait()
    if code:
        raise RuntimeError(phase + ' failed with exit code ' + str(code))


def main(config, domains_pid):
    cfg = load_config(config); root = Path(cfg['sgcr']['output_root'])
    resolved = root / 'resolved_config.yaml'
    save_json(root / 'pipeline_status.json', dict(status='running', phase='structure_groups_and_splits', child_pid=domains_pid))
    while not resolved.exists():
        try:
            cmd = Path(f'/proc/{domains_pid}/cmdline').read_bytes()
        except FileNotFoundError:
            raise RuntimeError('Domain preparation stopped before writing the resolved configuration')
        if b'80_prepare_full7.py' not in cmd:
            raise RuntimeError('Domain preparation process is no longer active')
        time.sleep(5)
    tests = (root / 'tests.log').read_text()
    match = re.search(r'(\d+) passed', tests)
    assert match and int(match.group(1)) >= 97 and ' failed' not in tests
    test_count = int(match.group(1))
    sources = []
    for folder in ('src', 'configs', 'tests'):
        sources.extend(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts)
    sources.extend(Path(p) for p in ('scripts/80_prepare_full7.py', 'scripts/81_run_full7.py',
        'scripts/84_full7_pipeline.py', 'scripts/86_precache_full_graphs.py', 'third_party/ALIGNN_COMMIT.txt'))
    manifest = {str(p): sha(p) for p in sources}
    save_json(root / 'training_source_manifest.json', manifest)
    save_json(root / 'tests_summary.json', dict(status='passed', passed=test_count,
        source_manifest='training_source_manifest.json'))
    launch(root, 'shared_d1', ['scripts/80_prepare_full7.py', '--config', str(resolved), '--phase', 'stage1'])
    launch(root, 'features', ['scripts/80_prepare_full7.py', '--config', str(resolved), '--phase', 'features'])
    launch(root, 'training_suite', ['scripts/81_run_full7.py', '--config', str(resolved), '--suite'])
    assert all(sha(path) == digest for path, digest in manifest.items())
    launch(root, 'audit', ['scripts/82_audit_full7.py', '--config', str(resolved)])
    launch(root, 'packaging', ['scripts/83_report_package_full7.py', '--config', str(resolved)])
    save_json(root / 'pipeline_status.json', dict(status='complete', phase='complete', seed=42,
        completed_methods=['A_direct', 'B_online_ewc', 'E_full_sgcr'], training_stopped=True,
        additional_experiments_scheduled=False))
    print('FULL7_PIPELINE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--domains-pid', type=int, required=True)
    args = p.parse_args()
    try:
        main(args.config, args.domains_pid)
    except BaseException as exc:
        root = Path(load_config(args.config)['sgcr']['output_root'])
        save_json(root / 'pipeline_status.json', dict(status='failed', error=str(exc), automatic_retry=False))
        raise
