"""Materialize the plan's ablations without altering the default router."""
import _bootstrap
from pathlib import Path
from src.utils.config import save_config


VARIANTS = {
    "no_reuse": {"router":{"mode":"no_reuse"}},
    "no_adapt": {"router":{"mode":"no_adapt"}},
    "no_expand": {"router":{"mode":"no_expand","max_experts":1}},
    "rae_random_replay": {"replay":{"method":"random"}},
    "no_replay": {"replay":{"size":0}},
    "always_reuse": {"router":{"mode":"always_reuse"}},
    "always_expand": {"router":{"mode":"always_expand"}},
    "fixed_2": {"router":{"mode":"no_expand","initial_experts":2,"max_experts":2}},
    "fixed_3": {"router":{"mode":"no_expand","initial_experts":3,"max_experts":3}},
    "fixed_5": {"router":{"mode":"no_expand","initial_experts":5,"max_experts":5}},
}


if __name__ == "__main__":
    for name,override in VARIANTS.items():
        output = Path("configs/experiment/ablations") / f"{name}.yaml"
        save_config(output,dict(inherits="../pilot.yaml",experiment={"name":name},**override))
        print(name,output)
