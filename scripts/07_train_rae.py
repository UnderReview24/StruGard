import _bootstrap
import argparse
from pathlib import Path
from src.continual.trainer import ContinualTrainer
from src.utils.config import load_config


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",default="configs/experiment/pilot.yaml")
    p.add_argument("--output")
    p.add_argument("--name",default="rae")
    a = p.parse_args()
    cfg = load_config(a.config)
    output = a.output or str(Path(cfg["experiment"].get("run_root","outputs/pilot"))/a.name)
    ContinualTrainer(cfg,output,a.name).run()
