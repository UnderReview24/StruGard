"""Check sample access restrictions and recompute published continual metrics."""
import _bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.continual.metrics import continual_metrics
from src.data.continual_stream import ContinualDomainStream
from src.utils.config import load_config
from src.utils.logging import save_json


def audit_run(output):
    out = Path(output)
    cfg = load_config(out / "config.yaml")
    stream = ContinualDomainStream(cfg["domains"]["assignments"], cfg["domains"]["splits"],
                                  cfg["data"].get("split_seed",cfg["experiment"]["seed"]), cfg["domains"]["order_seed"])
    access = json.loads((out / "access_audit.json").read_text())
    assert len(access) == len(stream)
    all_tests = {mid for s in stream for mid in s.test}
    all_vals = {mid for s in stream for mid in s.val}
    all_train = {mid for s in stream for mid in s.train}
    past = set()
    for stage, entry in zip(stream, access):
        current = set(entry["training_ids"])
        replay = set(entry["replay_ids"])
        oracle = entry.get("oracle_access",False)
        assert current == (all_train if oracle else set(stage.train)), f"Stage {stage.stage_id}: unexpected current training access"
        assert set(entry["validation_ids"]) == (all_vals if oracle else set(stage.val))
        if "fisher_ids" in entry:
            assert set(entry["fisher_ids"]).issubset(stage.train), "EWC curvature used forbidden examples"
        assert replay.issubset(past) and len(replay) <= cfg.get("replay", {}).get("size", 2000)
        assert not (current | replay) & (all_tests | all_vals), "Train/validation/test contamination"
        past |= current
    r = pd.read_csv(out / "continual_matrix.csv").drop(columns="stage").to_numpy()
    reported = pd.read_csv(out / "stage_metrics.csv")
    for stage_id in range(1, len(stream)+1):
        metric = continual_metrics(r, stage_id)
        for key in ["current_mae", "average_seen_mae", "forgetting"]:
            assert np.isclose(metric[key], reported.iloc[stage_id-1][key], rtol=1e-9, atol=1e-10)
        preds = pd.read_csv(out / f"stage_{stage_id}/predictions.csv")
        expected = {mid for ids in stream.seen_tests(stage_id).values() for mid in ids}
        assert preds.material_id.is_unique and set(preds.material_id) == expected
        for col, domain in enumerate(stream.order[:stage_id]):
            part = preds[preds.domain_id == domain]
            assert np.isclose(np.abs(part.prediction-part.target).mean(), r[stage_id-1, col], atol=1e-7)
    report = dict(status="passed", stages=len(stream), fixed_test_samples=len(all_tests),
                  oracle_exception=any(entry.get("oracle_access",False) for entry in access),
                  checks=["current-domain-only training", "bounded past-only replay", "no validation/test training access",
                          "fixed seen-domain prediction IDs", "MAE matrix and forgetting recomputed from saved predictions"])
    save_json(out / "audit.json", report)
    print(out, json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("outputs", nargs="+")
    args = p.parse_args()
    for output in args.outputs:
        audit_run(output)
