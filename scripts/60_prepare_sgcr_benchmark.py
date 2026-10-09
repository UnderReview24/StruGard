"""Prepare the requested benchmark without training an extra all-domain model."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from src.utils.config import load_config, save_config
from src.utils.logging import save_json
from src.data.domain_builder import crystal_system_domains, domain_summary
from src.data.continual_stream import ContinualDomainStream
from src.data.embeddings import extract_embeddings, encoder_digest
from src.models.alignn_wrapper import make_alignn, ALIGNNEncoder
from src.continual.strategies import setup_experiment, shared_stage1
from src.continual.v2_protocol import tensor_digest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_domains(cfg):
    processed = Path(cfg['data']['processed'])
    manifest = json.loads((processed/'manifest.json').read_text())
    metadata = pd.read_csv(processed/'metadata.csv')
    assert len(metadata) == manifest['prepared_samples'] == cfg['experiment']['sample_limit']
    assert manifest['seed'] == cfg['experiment']['seed']
    assert metadata.material_id.is_unique
    old = pd.read_csv('data/processed/metadata.csv')
    assert set(old.material_id) <= set(metadata.material_id)
    settings = cfg['domains']
    assignments = crystal_system_domains(metadata, groups=settings['groups'],
        min_domain_size=settings['min_domain_size'],
        max_samples_per_domain=settings['max_samples_per_domain'], seed=cfg['experiment']['seed'])
    assert len(assignments) == len(metadata) and assignments.domain_id.nunique() == settings['num_domains']
    path = Path(settings['assignments']); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), assignments, check_dtype=False)
    else:
        assignments.to_csv(path, index=False)
    stream = ContinualDomainStream(assignments, settings['splits'], cfg['data']['split_seed'], settings['order_seed'])
    summary = domain_summary(metadata, assignments)
    summary.to_csv(path.with_suffix('.summary.csv'), index=False)
    cfg['replay_augmented']['assignment_sha256'] = digest(path)
    cfg['replay_augmented']['split_sha256'] = digest(settings['splits'])
    cfg['experiment']['status'] = 'requested_three_method_benchmark'
    cfg['sgcr']['required_methods'] = ['A_direct', 'B_structural_replay', 'E_full_sgcr']
    cfg['sgcr']['direct_reference'] = str(Path(cfg['experiment']['run_root'])/'A_direct')
    cfg['sgcr']['baseline_reference'] = None
    root = Path(cfg['sgcr']['output_root'])
    report = dict(status='passed', retained_samples=len(assignments), excluded_samples=0,
        order=stream.order, signature=stream.signature, old_5k_ids_retained=True,
        partition_rule='per-domain 80/10/10', baseline_training_required_for_domain_construction=False,
        stage_splits=[dict(stage=s.stage_id, domain_id=s.domain_id,
            train=len(s.train), val=len(s.val), test=len(s.test)) for s in stream])
    save_json(root/'domain_preparation.json', report)
    save_config(root/'resolved_config.yaml', cfg)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


def prepare_features(cfg):
    torch.set_num_threads(4)
    stream, data = setup_experiment(cfg)
    stages = list(stream)
    plain = make_alignn(cfg['model'])
    shared_stage1(plain, stages[0], Path(cfg['sgcr']['output_root'])/'feature_initialization', cfg)
    encoder = ALIGNNEncoder(plain).requires_grad_(False).eval().cuda()
    manifest = json.loads((data.processed/'manifest.json').read_text())
    signature = dict(structures_sha256=manifest['structures_sha256'], graph_namespace=data.cache.namespace,
        encoder=encoder_digest(encoder), frozen_state_digest=tensor_digest(encoder.state_dict().items()),
        domain_signature=stream.signature, d1_training_ids=list(stages[0].train),
        d1_checkpoint_sha256=digest(Path(cfg['experiment']['shift_output'])/'d1_model/best.pt'))
    records = {}
    for stage in stages:
        h = extract_embeddings(encoder, data, stage.train, batch_size=32,
            cache_dir=cfg['replay_augmented']['selection_embedding_cache'])
        records.update({mid:dict(stable_embedding=v) for mid,v in zip(stage.train,h)})
        print('TRAINING_FEATURES', stage.stage_id, len(stage.train), flush=True)
    other_ids = sorted(set(data.ids)-set(records))
    h = extract_embeddings(encoder, data, other_ids, batch_size=32)
    records.update({mid:dict(stable_embedding=v) for mid,v in zip(other_ids,h)})
    assert set(records) == set(data.ids)
    assert encoder_digest(encoder) == signature['encoder']
    destination = Path(cfg['v2']['cache']); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = torch.load(destination, map_location='cpu', weights_only=True)
        assert old['signature'] == signature
        assert all(torch.equal(old['records'][mid]['stable_embedding'],r['stable_embedding']) for mid,r in records.items())
    else:
        torch.save(dict(signature=signature, records=records), destination)
    save_json(Path(cfg['sgcr']['output_root'])/'feature_preparation.json', dict(status='passed',
        samples=len(records), feature_dimension=encoder.output_dim, signature=signature,
        cache_sha256=digest(destination), labels_used_for_feature_extraction=False))
    print('FEATURE_PREPARATION_PASSED', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['domains','features'], required=True)
    args = parser.parse_args()
    {'domains':prepare_domains, 'features':prepare_features}[args.phase](load_config(args.config))
