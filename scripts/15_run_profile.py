"""Serial, independently gated profile execution. Dry-run is the default."""
import _bootstrap
import argparse
import json
from pathlib import Path
import subprocess
import sys

from src.utils.config import load_config


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--allow-negative-pilot",action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    exp = cfg["experiment"]
    root = Path(exp.get("run_root","outputs/pilot"))
    commands = [
        ["scripts/01_prepare_dataset.py","--limit",str(exp["sample_limit"]),"--seed","42","--output",cfg["data"]["processed"]],
        ["scripts/01_check_dataset.py","--processed",cfg["data"]["processed"],"--output",str(root/"dataset_check.json")],
        ["scripts/03_train_alignn_baseline.py","--config",a.config],
        ["scripts/02_build_domains.py","--config",a.config],
        ["scripts/05_validate_structural_shift.py","--config",a.config],
        ["scripts/06_train_sequential.py","--config",a.config],
        ["scripts/06_train_sequential.py","--config",a.config,"--method","random_replay"],
        ["scripts/07_train_rae.py","--config",a.config],
        ["scripts/09_audit_outputs.py",str(root/"sequential"),str(root/"random_replay"),str(root/"rae")],
    ]
    for command in commands:
        print(sys.executable," ".join(command),flush=True)
    if a.execute:
        gate = Path("outputs/pilot/summary/scientific_status.json")
        if exp["sample_limit"]>5000 and (not gate.exists() or not json.loads(gate.read_text())["success_criteria_met"]) and not a.allow_negative_pilot:
            raise SystemExit("The pilot has not passed model-superiority criteria. Scaling is stopped; --allow-negative-pilot explicitly overrides this research gate.")
        for command in commands:
            subprocess.run([sys.executable,"-u",*command],check=True)
