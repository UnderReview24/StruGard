"""Independently reproduce a static checkpoint's held-out predictions."""
import _bootstrap

import argparse
import json
from pathlib import Path
import torch

from src.data.dataset import CrystalPropertyDataset
from src.data.graph_cache import GraphCache
from src.models.alignn_wrapper import make_alignn
from src.continual.static_training import evaluate
from src.utils.config import load_config
from src.utils.logging import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/baseline_alignn/best.pt")
    p.add_argument("--config", default="outputs/baseline_alignn/config.yaml")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--output", default="outputs/evaluation/baseline")
    p.add_argument("--stage",type=int)
    a = p.parse_args()
    torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if state.get("kind") == "SPCrystal":
        from src.models.factory import load_sp_checkpoint
        from src.continual.strategies import setup_experiment
        from src.continual.diagnostics import evaluate_sp
        cfg = state["config"]
        model,_ = load_sp_checkpoint(a.checkpoint,"cuda")
        stream,data = setup_experiment(cfg)
        stage = state["stage"] if a.stage is None else a.stage
        if not 1 <= stage <= state["stage"]:
            raise ValueError("Requested stage exceeds checkpoint's trained stream")
        if a.split == "test":
            ids = [mid for values in stream.seen_tests(stage).values() for mid in values]
        else:
            ids = list(getattr(list(stream)[stage-1],a.split))
        metric,predictions = evaluate_sp(model,data,ids,cfg["training"]["batch_size"])
        out = Path(a.output)
        out.mkdir(parents=True,exist_ok=True)
        predictions.to_csv(out/"predictions.csv",index=False)
        save_json(out/"metrics.json",metric)
        print(json.dumps(metric),flush=True)
        return
    cfg = load_config(a.config)
    model = make_alignn(state["model_config"]).cuda()
    model.load_state_dict(state["model"])
    seed = cfg["experiment"]["seed"]
    processed = Path(cfg["data"]["processed"])
    splits = json.loads((processed / f"static_split_seed{seed}.json").read_text())
    dataset = CrystalPropertyDataset(processed, ids=splits[a.split],
        cache=GraphCache(cfg["data"]["graph_cache"], cfg["data"]["graph"]))
    metric, predictions = evaluate(model, dataset, cfg["training"]["batch_size"])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    save_json(out / "metrics.json", metric)
    print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
