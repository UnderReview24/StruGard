"""Prepare fixed crystal graphs on CPU without fitting to any target labels."""
import _bootstrap
import argparse
import json
import time
from pathlib import Path
import torch
from src.data.dataset import CrystalPropertyDataset, make_loader
from src.continual.full7_protocol import BoundedGraphCache
from src.utils.config import load_config
from src.utils.logging import save_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    cfg = load_config(p.parse_args().config); torch.set_num_threads(1)
    data = CrystalPropertyDataset(cfg['data']['processed'], cache=BoundedGraphCache(cfg['data']['graph_cache'], cfg['data']['graph']))
    start = time.monotonic(); count = 0
    for batch in make_loader(data, 32, num_workers=4):
        count += len(batch['material_id'])
        if count % 2048 == 0: print('GRAPH_PRECACHE', count, len(data), round(time.monotonic() - start, 1), flush=True)
    assert count == len(data) == 132752
    isolated = [json.loads(p.read_text()) for p in (data.cache.isolated_root / 'metadata').glob('*.json')]
    report = dict(status='passed', samples=count, elapsed_seconds=time.monotonic() - start,
        graph_namespace=data.cache.namespace, graph_config=data.cache.config, device='cpu', workers=4,
        labels_used_for_fitting=False, isolated_graphs=isolated)
    save_json(Path(cfg['sgcr']['output_root']) / 'graph_preparation.json', report)
    print('GRAPH_PRECACHE_COMPLETE', json.dumps(report), flush=True)
