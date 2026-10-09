import _bootstrap

import argparse
import json
from pathlib import Path
import torch

from src.data.dataset import CrystalPropertyDataset
from src.data.graph_cache import GraphCache
from src.models.alignn_wrapper import make_alignn
from src.continual.static_training import static_split, fit, evaluate
from src.utils.config import load_config, save_config
from src.utils.logging import save_json
from src.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment/pilot.yaml")
    p.add_argument("--output")
    a = p.parse_args()
    assert json.loads(Path("outputs/dataset_check_1000.json").read_text())["status"] == "passed"
    cfg = load_config(a.config)
    seed = cfg["experiment"]["seed"]
    seed_everything(seed)
    torch.set_num_threads(4)
    cache = GraphCache(cfg["data"]["graph_cache"], cfg["data"]["graph"])
    dataset = CrystalPropertyDataset(cfg["data"]["processed"], cache=cache)
    out = Path(a.output or cfg["experiment"].get("baseline_output","outputs/baseline_alignn"))
    out.mkdir(parents=True, exist_ok=True)
    splits = static_split(dataset.ids, seed, Path(cfg["data"]["processed"]) / f"static_split_seed{seed}.json")
    save_config(out / "config.yaml", cfg)
    model = make_alignn(cfg["model"])
    info = fit(model, dataset.subset(splits["train"]), dataset.subset(splits["val"]),
               out, cfg["training"], seed)
    metric, predictions = evaluate(model, dataset.subset(splits["test"]), cfg["training"]["batch_size"])
    predictions.to_csv(out / "predictions.csv", index=False)
    save_json(out / "metrics.json", dict(status="passed", test=metric, **info,
               samples=len(dataset), splits={k: len(splits[k]) for k in ("train", "val", "test")},
               target_units="eV/atom", scope=f"{len(dataset)}-sample engineering pilot, not MatBench leaderboard reproduction"))
    print("STATIC_BASELINE_PASSED", metric, flush=True)


if __name__ == "__main__":
    main()
