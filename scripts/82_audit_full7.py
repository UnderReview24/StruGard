"""Audit all 21 saved stages, fresh splits, EWC consolidation and SGCR updates."""
import _bootstrap
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.continual.full7_protocol import METHODS, setup_full, sha
from src.continual.full7_runner import Full7Runner
from src.continual.sgcr_benchmark import expected_steps
from src.continual.sgcr_trainer import SharedStructuralPredictor
from src.continual.sgcr_tuning import ScaledStructuralMemory
from src.continual.replay_buffer import StructuralReplayBuffer
from src.continual.replay_augmented_rae import partition_memory
from src.continual.metrics import continual_metrics
from src.continual.v2_protocol import tensor_digest
from src.continual.ewc import OnlineEWC
from src.models.alignn_wrapper import make_alignn
from src.data.dataset import make_loader, to_device
from src.utils.config import load_config
from src.utils.logging import save_json


def hierarchy(settings, items, allowed):
    return ScaledStructuralMemory(settings['prototype_count'], settings['structural_epsilon'], 42,
        settings['calibration_batches'], 32, projection_scale=settings['projection_scale']).fit(items, allowed)


def audit(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); torch.use_deterministic_algorithms(True)
    root = Path(cfg['sgcr']['output_root'])
    for name in ('protected_before_run.json', 'training_source_manifest.json'):
        manifest = json.loads((root / name).read_text())
        assert all(Path(p).is_file() and sha(p) == h for p, h in manifest.items()), name
    assert json.loads((root / 'tests_summary.json').read_text())['passed'] >= 97
    stream, data = setup_full(cfg); stages = list(stream)
    tests = {mid for s in stages for mid in s.test}; vals = {mid for s in stages for mid in s.val}
    legacy = set(pd.read_csv('data/processed/n10000_seed42/metadata.csv').material_id)
    split = pd.read_csv(cfg['domains']['split_audit'])
    assert split.groupby('group_id').split.nunique().eq(1).all()
    legacy_groups = set(split[split.material_id.isin(legacy)].group_id)
    assert not set(split[split.split == 'test'].group_id) & legacy_groups
    assert not tests & legacy and len(data) == 132752
    settings = {**cfg['v2'], **cfg['replay_augmented'], **cfg['sgcr']}
    count = expected_steps(stages, 32, 10)
    features = torch.load(cfg['v2']['cache'], map_location='cpu', weights_only=True)['records']
    reference_initial = None; reports = []; final_rows = []; schedules = {}
    for method in METHODS:
        out = Path(cfg['experiment']['run_root']) / method
        status = json.loads((out / 'status.json').read_text()); assert status['status'] == 'passed'
        run_cfg = load_config(out / 'config.yaml'); assert run_cfg == cfg
        initial = json.loads((out / 'initialization.json').read_text())
        if reference_initial is None: reference_initial = initial
        assert initial == reference_initial and initial['optimizer_state_entries'] == 0
        samples = pd.read_csv(out / 'sample_predictions.csv'); metrics = pd.read_csv(out / 'stage_metrics.csv')
        final = pd.read_csv(out / 'final_results.csv').iloc[0]
        assert len(metrics) == 7 and final.optimizer_steps == count and metrics.optimizer_steps.sum() == count
        is_sgcr = method == 'E_full_sgcr'; is_ewc = method == 'B_online_ewc'
        diagnostics = pd.read_csv(out / 'batch_diagnostics.csv') if is_sgcr else None
        selection = pd.read_csv(out / 'replay_selection.csv') if is_sgcr else None
        if is_sgcr:
            quality = pd.read_csv(out / 'retrieval_quality_batches.csv')
            assert len(diagnostics) == len(quality) == count
            assert diagnostics.projection_scale.eq(.1).all() and diagnostics.rho.between(0, .1 + 1e-12).all()
        seen = set(); previous_memory = {}; matrix = np.full((7, 7), np.nan)
        maximum_difference = 0.; fisher_difference = 0.; checked_steps = 0; maximum_memory = 0
        ewc_positive_penalty_steps = 0
        ewc = OnlineEWC(**{k: cfg['ewc'][k] for k in ('strength', 'decay')}) if is_ewc else None
        reconstructed_buffer = StructuralReplayBuffer(500, settings['candidate_pool_size'], 42, 'kcenter') if is_sgcr else None
        for stage in stages:
            s = stage.stage_id; p = out / f'stage_{s}'
            assert json.loads((p / 'completed_stage.json').read_text())['status'] == 'passed'
            evidence = json.loads((p / 'data_audit.json').read_text())
            trace = json.loads((p / 'optimizer_trace.json').read_text())
            assert evidence['current_train'] == list(stage.train) and evidence['current_validation'] == list(stage.val)
            assert set(evidence['memory_before']) <= seen
            assert set(evidence['memory_before']) == set(previous_memory)
            replay, reserved = partition_memory(evidence['memory_before'], 2042 + s, .2) if evidence['memory_before'] else ([], [])
            assert evidence['replay_train'] == replay and evidence['reserved'] == reserved
            helper = object.__new__(Full7Runner); helper.settings = settings; helper.batch_size = 32
            helper.smoke = False; helper.replay_train = replay; helper.features = features
            planned = helper.batches(stage) if s > 1 else []
            assert len(trace) == len(planned)
            current_schedule = [(r['epoch'], r['current']) for r in trace]
            if method == METHODS[0]: schedules[s] = current_schedule
            assert current_schedule == schedules[s]
            row = metrics[metrics.stage_id == s].iloc[0]
            assert row.optimizer_steps == len(trace)
            assert row.current_exposures == sum(len(t['current']) for t in trace)
            for recorded, intended in zip(trace, planned):
                assert recorded['epoch'] == intended['epoch'] and recorded['current'] == intended['current']
                assert recorded['random_reference'] == intended['replay']
                assert recorded['optimizer_steps'] == recorded['backward_steps'] == 1
                assert set(recorded['current']) <= set(stage.train)
                assert set(recorded['replay']) <= set(replay) and not set(recorded['replay']) & set(reserved)
                if is_sgcr:
                    assert len(recorded['replay']) == len(set(recorded['replay'])) == len(recorded['current'])
                    assert recorded['autograd_calls'] == 2
                else:
                    assert recorded['replay'] == [] and recorded['autograd_calls'] == 0
                if is_ewc: assert np.isfinite(recorded['ewc_penalty']) and recorded['ewc_penalty'] >= 0
            checked_steps += len(trace)
            if is_ewc and s > 1:
                positive = sum(t['ewc_penalty'] > 0 for t in trace)
                assert positive > 0, ('Online EWC regularization was inactive', s)
                ewc_positive_penalty_steps += positive
            checkpoint = torch.load(p / 'checkpoint.pt', map_location='cpu', weights_only=True)
            model = SharedStructuralPredictor(make_alignn(checkpoint['model_config'])).cuda()
            model.load_state_dict(checkpoint['model']); model.eval()
            assert tensor_digest(model.stable_encoder.state_dict().items()) == initial['frozen_digest']
            if s == 1: assert tensor_digest(model.predictor.state_dict().items()) == initial['predictor_digest']
            table = samples[(samples.stage_id == s) & (samples.split == 'test')]
            expected_test_ids = {mid for prior in stages[:s] for mid in prior.test}
            assert table.material_id.is_unique and set(table.material_id) == expected_test_ids
            assert np.allclose(table.absolute_error, (table.target - table.prediction).abs(), atol=1e-6, rtol=1e-6)
            for j, prior in enumerate(stages[:s]):
                matrix[s - 1, j] = table[table.domain_id == prior.domain_id].absolute_error.mean()
            calculated = continual_metrics(matrix, s)
            assert abs(calculated['average_seen_mae'] - row.average_seen_mae) < 1e-7
            assert abs(calculated['forgetting'] - row.forgetting) < 1e-7
            predictions = []
            with torch.no_grad():
                for batch in make_loader(data.subset(table.material_id.tolist()), 32):
                    predictions.extend(model(to_device(batch, 'cuda')['graphs'])['prediction'].cpu().tolist())
            difference = float(np.max(np.abs(np.asarray(predictions) - table.prediction.to_numpy())))
            assert difference < 2e-5, (method, s, difference)
            maximum_difference = max(maximum_difference, difference)
            if is_ewc:
                info = json.loads((p / 'ewc_audit.json').read_text())
                before = tensor_digest(model.predictor.state_dict().items())
                ids = ewc.consolidate(model.predictor, data, stage.train, cfg['ewc']['sample_size'], 42 + s)
                assert ids == info['fisher_ids'] and set(ids) <= set(stage.train)
                assert tensor_digest(model.predictor.state_dict().items()) == before
                saved = torch.load(p / 'ewc_state.pt', map_location='cpu', weights_only=True)
                assert saved['strength'] == ewc.strength and saved['decay'] == ewc.decay
                for name in saved['mean']:
                    assert torch.equal(saved['mean'][name], ewc.mean[name].cpu())
                    a = saved['fisher'][name]; b = ewc.fisher[name].cpu()
                    torch.testing.assert_close(a, b, atol=1e-7, rtol=2e-5)
                    fisher_difference = max(fisher_difference, float((a - b).abs().max()))
                assert info['fisher_backward_calls'] == len(ids)
            seen.update(stage.train)
            memory = torch.load(p / 'replay_memory.pt', map_location='cpu', weights_only=True)['items']
            assert set(memory) == set(evidence['memory_after']) and set(memory) <= seen
            assert not set(memory) & (tests | vals) and len(memory) <= (500 if is_sgcr else 0)
            maximum_memory = max(maximum_memory, len(memory))
            if is_sgcr:
                rebuilt = reconstructed_buffer
                rebuilt.update([dict(material_id=mid, graph_reference=mid, target=float(data.metadata.at[mid, 'target']),
                    expert_id=0, domain_id=stage.domain_id, stable_embedding=features[mid]['stable_embedding'],
                    stored_prediction=0., insertion_stage=s) for mid in stage.train])
                assert set(rebuilt.ids) == set(memory)
                current_h = hierarchy(settings, memory, seen)
                assert json.loads((p / 'prototypes.json').read_text()) == json.loads(json.dumps(current_h.state_dict()))
                if s > 1:
                    assert len(replay) == 400 and len(reserved) == 100
                    old_h = hierarchy(settings, previous_memory, evidence['memory_before'])
                    assert json.loads((p / 'historical_prototypes.json').read_text()) == json.loads(json.dumps(old_h.state_dict()))
                    ds = diagnostics[diagnostics.stage_id == s].set_index('step')
                    selections = {int(k): group for k, group in selection[selection.stage_id == s].groupby('step')}
                    assert len(ds) == len(trace)
                    for t in trace:
                        d = ds.loc[t['step']]; query = old_h.query(helper.hs(t['current']))
                        assert abs(query['rho'] - d.rho) < 1e-6 and abs(query['raw_rho'] - d.raw_rho) < 1e-6
                        applied = d.gradient_dot_before < 0 and d.rho > 0 and d.current_gradient_norm**2 > 1e-8 and d.replay_gradient_norm**2 > 1e-8
                        assert bool(d.projection_applied) == applied
                        expected_after = d.gradient_dot_before
                        if applied:
                            expected_after *= 1 - d.rho * d.replay_gradient_norm**2 / (d.replay_gradient_norm**2 + 1e-8)
                        assert abs(d.gradient_dot_after - expected_after) < 1e-5 * max(1., d.current_gradient_norm * d.replay_gradient_norm)
                        candidates = selections[t['step']]
                        assert candidates.material_id.is_unique and set(candidates.material_id) <= set(replay)
                        assert set(candidates[candidates.selected].material_id) == set(t['replay'])
                        priority = settings['lambda_struct'] * candidates.structural_relevance + settings['lambda_conflict'] * candidates.normalized_conflict_score
                        assert np.allclose(priority, candidates.combined_priority, atol=1e-6)
                        local = candidates[candidates.selection_type == 'local_conflict']; k = int(local.selected.sum())
                        best = local.sort_values(['combined_priority', 'material_id'], ascending=[False, True]).head(k)
                        assert set(best.material_id) == set(local[local.selected].material_id)
            previous_memory = memory
            print('FULL7_STAGE_AUDIT', method, s, 'passed', difference, flush=True)
            del model, checkpoint
        assert checked_steps == count and abs(final.final_avg_mae - calculated['average_seen_mae']) < 1e-7
        assert abs(final.forgetting - calculated['forgetting']) < 1e-7
        assert abs(final.peak_stage_forgetting - metrics.forgetting.max()) < 1e-7
        report = dict(status='passed', method=method, stages=7, optimizer_steps=count, test_samples=len(tests),
            maximum_memory=maximum_memory, max_checkpoint_prediction_difference=maximum_difference,
            max_fisher_recomputation_difference=fisher_difference, final_avg_mae=float(final.final_avg_mae),
            ewc_positive_penalty_steps=ewc_positive_penalty_steps)
        reports.append(report); save_json(out / 'full7_audit.json', report); final_rows.append(final.to_dict())
    pd.DataFrame(final_rows).to_csv(root / 'final_results.csv', index=False)
    d1_status = json.loads((Path(cfg['experiment']['shift_output']) / 'stage1_status.json').read_text())
    d1_steps = ((len(stages[0].train) + 31) // 32) * d1_status['epochs_run']
    outcome = dict(status='passed', methods=reports, stages_per_method=7, checked_checkpoints=21,
        optimizer_steps_per_method=count, continuation_optimizer_steps=3 * count,
        shared_d1_optimizer_steps=d1_steps, total_new_optimizer_steps=3 * count + d1_steps, test_samples=len(tests),
        fresh_test_excludes_pilot_and_duplicate_groups=True, seed=42, full_test_used_for_selection=False)
    save_json(root / 'audit.json', outcome)
    print('FULL7_AUDIT_COMPLETE', json.dumps(outcome), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    audit(load_config(p.parse_args().config))
