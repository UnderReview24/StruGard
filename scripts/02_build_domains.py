import _bootstrap

import argparse
import gzip
import json
from pathlib import Path
import pandas as pd

from src.data.domain_builder import crystal_system_domains, soap_domains, domain_summary
from src.data.continual_stream import ContinualDomainStream
from src.utils.config import load_config
from src.utils.logging import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment/pilot.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    baseline = Path(cfg["experiment"].get("baseline_output","outputs/baseline_alignn"))
    assert json.loads((baseline/"metrics.json").read_text())["status"] == "passed"
    settings = cfg["domains"]
    metadata = pd.read_csv(Path(cfg["data"]["processed"]) / "metadata.csv")
    report = dict(method=settings["method"], seed=cfg["experiment"]["seed"])
    if settings["method"] == "crystal_system":
        assignments = crystal_system_domains(metadata, groups=settings.get("groups"),
            min_domain_size=settings.get("min_domain_size", 100),
            max_samples_per_domain=settings.get("max_samples_per_domain"),
            seed=cfg["experiment"]["seed"], merge_small=settings.get("merge_small", False))
    elif settings["method"] == "soap":
        with gzip.open(Path(cfg["data"]["processed"]) / "structures.json.gz", "rt") as f:
            structures = json.load(f)
        assignments, extra = soap_domains(metadata, structures,
            settings.get("feature_path", "data/embeddings/domain_features.npy"),
            settings.get("num_domains", 5), settings.get("cluster_seed", 0), settings.get("soap"))
        report.update(extra)
    else:
        raise ValueError("Unknown domain method")
    assert assignments.domain_id.nunique() == settings["num_domains"]
    out = Path(settings["assignments"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        pd.testing.assert_frame_equal(pd.read_csv(out), assignments, check_dtype=False)
    else:
        assignments.to_csv(out, index=False)
    summary = domain_summary(metadata, assignments)
    summary.to_csv(out.with_suffix(".summary.csv"), index=False)
    stream = ContinualDomainStream(assignments, settings["splits"],
        cfg["data"].get("split_seed",cfg["experiment"]["seed"]), settings["order_seed"])
    report.update(status="passed", retained_samples=len(assignments), excluded_samples=len(metadata)-len(assignments),
                  order=stream.order, signature=stream.signature,
                  stage_splits=[dict(stage=s.stage_id, domain_id=s.domain_id,
                    train=len(s.train), val=len(s.val), test=len(s.test)) for s in stream])
    save_json(out.with_suffix(".json"), report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
