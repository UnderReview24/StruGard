"""Prepare fresh group-held-out seven-domain splits and a shared D1 model."""
import _bootstrap
import argparse
import gzip
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data.full_structure_groups import structural_groups
from src.data.convert_structure import to_pymatgen
from src.data.domain_builder import crystal_system_domains, domain_summary
from src.data.embeddings import extract_embeddings, encoder_digest
from src.continual.full7_protocol import ORDER, sha, grouped_splits, setup_full, read_full_metadata
from src.continual.static_training import fit
from src.continual.strategies import shared_stage1
from src.continual.v2_protocol import tensor_digest
from src.models.alignn_wrapper import make_alignn, ALIGNNEncoder
from src.utils.config import load_config, save_config
from src.utils.logging import save_json


def domains(cfg):
    root = Path(cfg['sgcr']['output_root']); processed = Path(cfg['data']['processed'])
    manifest = json.loads((processed / 'manifest.json').read_text())
    metadata = read_full_metadata(processed / 'metadata.csv').sort_values('material_id')
    assert len(metadata) == manifest['prepared_samples'] == manifest['total_source_samples'] == 132752
    assert sha(processed / 'metadata.csv') == manifest['metadata_sha256']
    assert sha(processed / 'structures.json.gz') == manifest['structures_sha256']
    assignments = crystal_system_domains(metadata, {n: [n] for n in ORDER}, min_domain_size=100)
    groups_path = root / 'duplicate_groups.csv'
    existing = pd.read_csv(groups_path) if groups_path.exists() else pd.DataFrame(columns=['material_id', 'group_id', 'group_size'])
    assert existing.material_id.is_unique and set(existing.material_id) <= set(metadata.material_id)
    missing = set(metadata.material_id) - set(existing.material_id)
    if missing:
        formulas = set(metadata.loc[metadata.material_id.isin(missing), 'reduced_formula'])
        pending = metadata[metadata.reduced_formula.isin(formulas)]
        records = existing[~existing.material_id.isin(pending.material_id)].to_dict('records')
        print('GROUPING_PENDING', len(pending), 'REUSING', len(records), flush=True)
        with gzip.open(processed / 'structures.json.gz', 'rt') as f:
            structures = json.load(f)
        checked = len(records); start = time.monotonic()
        for formula, part in pending.groupby('reduced_formula', sort=True):
            ids = part.material_id.tolist()
            if len(ids) == 1:
                records.append(dict(material_id=ids[0], group_id=ids[0], group_size=1))
            else:
                values = [to_pymatgen(structures[mid]) for mid in ids]
                if len(ids) >= 32: print('LARGE_FORMULA', formula, len(ids), flush=True)
                def progress(done, total):
                    if total >= 32: print('PAIR_ROWS', formula, done, total, flush=True)
                for mids in structural_groups(ids, values, workers=12, progress=progress):
                    for mid in mids:
                        records.append(dict(material_id=mid, group_id=mids[0], group_size=len(mids)))
            checked += len(ids)
            if checked // 2000 > (checked - len(ids)) // 2000:
                print('STRUCTURE_GROUPING', checked, len(metadata), round(time.monotonic() - start, 1), flush=True)
        pd.DataFrame(records).sort_values('material_id').to_csv(groups_path, index=False)
        if len(existing):
            save_json(root / 'grouping_resume.json', dict(status='passed', previous_samples=len(existing),
                recomputed_formulas=sorted(formulas), recomputed_ids=sorted(pending.material_id),
                retained_samples=len(records), structure_matching_tolerances_unchanged=True))
        del structures
    groups = pd.read_csv(groups_path)
    assert len(groups) == len(metadata) and set(groups.material_id) == set(metadata.material_id)
    legacy_ids = set(pd.read_csv('data/processed/n10000_seed42/metadata.csv').material_id)
    old = json.loads(Path('data/domains/crystal4_n10000_splits_seed42.json').read_text())
    legacy_val = {mid for split in old['splits'].values() for key in ('val', 'test') for mid in split[key]}
    saved, table = grouped_splits(assignments, groups, legacy_ids, legacy_val)
    cfg['experiment']['sample_limit'] = len(metadata)
    for path in (cfg['domains']['assignments'], cfg['domains']['splits'], cfg['domains']['split_audit']):
        assert not Path(path).exists(), path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(cfg['domains']['assignments'], index=False)
    table.to_csv(cfg['domains']['split_audit'], index=False)
    save_json(cfg['domains']['splits'], saved)
    cfg['replay_augmented']['assignment_sha256'] = sha(cfg['domains']['assignments'])
    cfg['replay_augmented']['split_sha256'] = sha(cfg['domains']['splits'])
    summary = domain_summary(metadata, assignments)
    counts = table.groupby(['domain_name', 'split']).size().unstack(fill_value=0)
    summary = summary.merge(counts, left_on='domain_name', right_index=True)
    summary['stage_id'] = summary.domain_name.map({n: i + 1 for i, n in enumerate(ORDER)})
    summary.sort_values('stage_id').to_csv(root / 'domain_summary.csv', index=False)
    save_json(root / 'split_protocol.json', dict(status='passed', original_samples=len(metadata),
        retained_samples=len(table), legacy_pilot_samples=len(legacy_ids), legacy_pilot_test_excluded=True,
        unique_structure_groups=int(groups.group_id.nunique()), repeated_structure_groups=int(groups[groups.group_size > 1].group_id.nunique()),
        grouping=dict(method='pairwise pymatgen StructureMatcher and connected components', ltol=.001, stol=.001, angle_tol=.1,
            scale=False, primitive_cell=True, symmetric=True, note='Tolerance-based structure groups; not a chemical-family holdout'),
        order_names=ORDER, split_counts=table.split.value_counts().to_dict(),
        assignment_sha256=cfg['replay_augmented']['assignment_sha256'], split_sha256=cfg['replay_augmented']['split_sha256']))
    save_config(root / 'resolved_config.yaml', cfg)
    print(summary.sort_values('stage_id').to_string(index=False), flush=True)


def stage1(cfg):
    stream, data = setup_full(cfg, require_d1=False); first = list(stream)[0]
    root = Path(cfg['experiment']['shift_output']); root.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    model = make_alignn(cfg['model'])
    save_json(root / 'access_audit.json', dict(training_ids=list(first.train),
        validation_ids=list(first.val), test_ids_used=[], initialization='random seed 42'))
    result = fit(model, data.subset(first.train), data.subset(first.val), root / 'd1_model',
        cfg['training'], 42, epochs=cfg['training']['epochs_stage1'])
    save_json(root / 'stage1_status.json', dict(status='passed', **result))
    print('SHARED_D1_COMPLETE', json.dumps(result), flush=True)


def features(cfg):
    stream, data = setup_full(cfg); stages = list(stream)
    plain = make_alignn(cfg['model'])
    shared_stage1(plain, stages[0], Path(cfg['sgcr']['output_root']) / 'feature_initialization', cfg)
    encoder = ALIGNNEncoder(plain).requires_grad_(False).eval().cuda()
    manifest = json.loads((data.processed / 'manifest.json').read_text())
    signature = dict(structures_sha256=manifest['structures_sha256'], graph_namespace=data.cache.namespace,
        encoder=encoder_digest(encoder), frozen_state_digest=tensor_digest(encoder.state_dict().items()),
        domain_signature=stream.signature, d1_training_ids=list(stages[0].train),
        d1_checkpoint_sha256=sha(Path(cfg['experiment']['shift_output']) / 'd1_model/best.pt'))
    records = {}
    for stage in stages:
        h = extract_embeddings(encoder, data, stage.train, 32,
            cache_dir=cfg['replay_augmented']['selection_embedding_cache'])
        records.update({mid: dict(stable_embedding=v) for mid, v in zip(stage.train, h)})
        print('FULL_TRAIN_FEATURES', stage.stage_id, len(stage.train), flush=True)
    other = sorted(set(data.ids) - set(records))
    h = extract_embeddings(encoder, data, other, 32)
    records.update({mid: dict(stable_embedding=v) for mid, v in zip(other, h)})
    assert set(records) == set(data.ids)
    destination = Path(cfg['v2']['cache']); destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    torch.save(dict(signature=signature, records=records), destination)
    save_json(Path(cfg['sgcr']['output_root']) / 'feature_preparation.json',
        dict(status='passed', samples=len(records), labels_used=False, signature=signature, cache_sha256=sha(destination)))
    print('FULL_FEATURES_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--phase', choices=['domains', 'stage1', 'features'], required=True)
    args = p.parse_args(); {'domains': domains, 'stage1': stage1, 'features': features}[args.phase](load_config(args.config))
