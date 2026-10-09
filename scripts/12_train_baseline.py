import _bootstrap
import argparse
from src.utils.config import load_config
from src.continual.extra_baselines import run_baseline


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",default="configs/experiment/pilot.yaml")
    p.add_argument("--method",required=True,choices=["joint","structural_replay","ewc"])
    p.add_argument("--output",required=True)
    a = p.parse_args()
    run_baseline(load_config(a.config),a.output,a.method)
