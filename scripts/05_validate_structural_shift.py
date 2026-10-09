"""Gate E. Train on D1 only; diagnose ID versus fixed OOD test sets."""
import _bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

from src.continual.static_training import fit, evaluate
from src.data.dataset import CrystalPropertyDataset
from src.data.graph_cache import GraphCache
from src.data.continual_stream import ContinualDomainStream
from src.data.structural_features import geometry_features, FEATURE_NAMES
from src.models.alignn_wrapper import make_alignn
from src.utils.config import load_config, save_config
from src.utils.logging import save_json
from src.utils.seed import seed_everything


def bootstrap_difference(id_error, ood_error, rng, samples, alpha):
    changes = np.empty(samples)
    for i in range(samples):
        changes[i] = rng.choice(ood_error, len(ood_error), replace=True).mean() - rng.choice(id_error, len(id_error), replace=True).mean()
    return tuple(map(float, np.quantile(changes, [alpha/2, 1-alpha/2])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment/pilot.yaml")
    p.add_argument("--output")
    a = p.parse_args()
    cfg = load_config(a.config)
    seed = cfg["experiment"]["seed"]
    seed_everything(seed)
    torch.set_num_threads(4)
    stream = ContinualDomainStream(cfg["domains"]["assignments"], cfg["domains"]["splits"],
                                  cfg["data"].get("split_seed",seed), cfg["domains"]["order_seed"])
    dataset = CrystalPropertyDataset(cfg["data"]["processed"], domains=stream.domain_map,
        cache=GraphCache(cfg["data"]["graph_cache"], cfg["data"]["graph"]))
    stages = list(stream)
    first = stages[0]
    out = Path(a.output or cfg["experiment"].get("shift_output","outputs/shift_validation"))
    out.mkdir(parents=True, exist_ok=True)
    save_config(out / "config.yaml", cfg)
    save_json(out / "access_audit.json", dict(training_ids=list(first.train),
        validation_ids=list(first.val), test_ids={str(s.domain_id): list(s.test) for s in stages},
        future_domains_used_in_training=False, initialization="seeded random weights; no static baseline checkpoint"))
    model = make_alignn(cfg["model"])
    fit(model, dataset.subset(first.train), dataset.subset(first.val), out / "d1_model",
        cfg["training"], seed, epochs=cfg["training"]["epochs_stage1"])
    raw = geometry_features(dataset.structures, first.train)
    mean, scale = raw.mean(0), np.maximum(raw.std(0), 1e-6)
    centers = np.stack([((geometry_features(dataset.structures, s.train)-mean)/scale).mean(0) for s in stages])
    center_distance = cdist(centers, centers)
    pd.DataFrame(center_distance, index=[s.domain_id for s in stages], columns=[s.domain_id for s in stages]).to_csv(out / "structural_domain_distance.csv", index_label="domain_id")
    save_json(out / "geometry_normalizer.json", dict(features=FEATURE_NAMES, mean=mean.tolist(), scale=scale.tolist(),
        fit_ids=list(first.train), note="cell-geometry diagnostic; not a SOAP feature and not model routing"))
    errors, records, predictions = [], [], []
    for i, stage in enumerate(stages):
        metric, table = evaluate(model, dataset.subset(stage.test), cfg["training"]["batch_size"])
        records.append(dict(domain_id=stage.domain_id, stream_position=stage.stage_id,
            in_distribution=i==0, structural_distance_from_d1=float(center_distance[0, i]), **metric))
        errors.append(table.absolute_error.to_numpy())
        predictions.append(table)
    settings = cfg["shift_gate"]
    rng = np.random.default_rng(seed+1)
    alpha = settings["familywise_alpha"] / max(len(stages)-1, 1)
    meaningful = False
    for i, record in enumerate(records):
        delta = record["mae"] - records[0]["mae"]
        ratio = record["mae"] / max(records[0]["mae"], 1e-12) - 1
        low, high = (0., 0.) if i == 0 else bootstrap_difference(
            errors[0], errors[i], rng, settings["bootstrap_samples"], alpha)
        passed = i > 0 and low > 0 and delta >= settings["min_absolute_increase"] and ratio >= settings["min_relative_increase"]
        record.update(mae_increase=delta, relative_increase=ratio,
                      difference_ci_low=low, difference_ci_high=high, meaningful_degradation=passed)
        meaningful |= passed
    pd.DataFrame(records).to_csv(out / "cross_domain_mae.csv", index=False)
    pd.concat(predictions).to_csv(out / "predictions.csv", index=False)
    warning = None if meaningful else "WARNING: current domain construction does not produce a meaningful structural distribution shift."
    save_json(out / "gate.json", dict(status="passed" if meaningful else "failed", warning=warning,
        criteria=settings, order=stream.order, domain_signature=stream.signature,
        caution="Small pilot. Crystal-system groups also differ in composition and target distribution; this test does not isolate a causal structural effect."))
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print("STRUCTURAL_SHIFT_GATE_PASSED" if meaningful else warning, flush=True)
    if not meaningful:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
