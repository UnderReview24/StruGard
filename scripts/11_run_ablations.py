"""Serial pilot ablations; skip only completed, audited runs."""
import _bootstrap
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--only",nargs="*")
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("experiment_variants",Path(__file__).with_name("10_make_experiments.py"))
    variants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(variants)
    subprocess.run([sys.executable,"scripts/10_make_experiments.py"],check=True)
    names = a.only or list(variants.VARIANTS)
    for name in names:
        if name not in variants.VARIANTS:
            raise ValueError(f"Unknown ablation: {name}")
        output = Path("outputs/pilot") / name
        status = output/"status.json"
        if status.exists() and json.loads(status.read_text())["status"]=="passed":
            print("SKIP_COMPLETED",name,flush=True)
            if not a.dry_run:
                subprocess.run([sys.executable,"scripts/09_audit_outputs.py",str(output)],check=True)
            continue
        command = [sys.executable,"-u","scripts/07_train_rae.py","--name",name,
                   "--config",f"configs/experiment/ablations/{name}.yaml","--output",str(output)]
        print("RUN", " ".join(command),flush=True)
        if not a.dry_run:
            subprocess.run(command,check=True)
            subprocess.run([sys.executable,"scripts/09_audit_outputs.py",str(output)],check=True)
