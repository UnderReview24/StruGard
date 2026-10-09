"""Use local MatBench, preserve row IDs, and serialize selected structures."""
import _bootstrap

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from src.data.convert_structure import to_jarvis, to_pymatgen
from src.utils.logging import save_json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_source(source, raw, download=False):
    source, raw = Path(source), Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    if source.exists() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            names = [n for n in z.namelist() if Path(n).name == "matbench_mp_e_form.pkl"]
            if len(names) != 1:
                raise ValueError(f"Expected one formation-energy dataset, found {names}")
            destination = raw / "matbench_mp_e_form.pkl"
            if not destination.exists():
                tmp = destination.with_suffix(".partial")
                with z.open(names[0]) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp.replace(destination)
            if destination.stat().st_size != z.getinfo(names[0]).file_size:
                raise ValueError("Extracted size mismatch; refusing stale/corrupt local data.")
            for n in z.namelist():
                if Path(n).name == "README.txt":
                    (raw / "SOURCE_README.txt").write_bytes(z.read(n))
            return destination
    if source.is_file():
        return source
    local = raw / "matbench_mp_e_form.pkl"
    if local.exists():
        return local
    if not download:
        raise FileNotFoundError(f"Local dataset absent: {source}; downloading requires --download.")
    from matminer.datasets import load_dataset
    load_dataset("matbench_mp_e_form").to_pickle(local)
    return local


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="../MoE_pretraining_data-20260907T081101Z-1-001.zip")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symprec", type=float, default=0.01)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    if a.limit < 0:
        raise ValueError("limit must be positive, or zero for the full dataset")
    gate = json.loads((_bootstrap.ROOT / "outputs/environment.json").read_text())
    assert gate["status"] == "passed", "Gate A must pass first"
    source = resolve_source(a.source, a.raw, a.download)
    fingerprint = sha256(source)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(source=str(source), source_sha256=fingerprint,
                    dataset="matbench_mp_e_form", target="e_form", units="eV/atom",
                    limit=a.limit, seed=a.seed, symprec=a.symprec, angle_tolerance=5,
                    selection="sorted prefix of seed permutation; IDs are original row positions")
    old = out / "manifest.json"
    if old.exists():
        previous = json.loads(old.read_text())
        if any(previous.get(k) != v for k, v in manifest.items()):
            raise ValueError("Output belongs to different preparation settings; use another --output.")
        if (out / "metadata.csv").exists() and (out / "structures.json.gz").exists():
            print("Already prepared; reusing", out, flush=True)
            return
    print("Loading", source, flush=True)
    # Reading the historical 475 MB pickle constructs every Python object.
    # Cache a selected raw subset once so 100/1000/5000 gates do not repeatedly
    # unpickle all 132k structures. This does not build any graphs or metadata.
    capacity = max(5000, a.limit)
    subset_path = Path(a.raw) / f"subset_{fingerprint[:12]}_seed{a.seed}_n{capacity}.pkl"
    if a.limit and subset_path.exists():
        selected = pd.read_pickle(subset_path)
        n = selected.attrs["total_source_samples"]
    else:
        df = pd.read_pickle(source)
        assert {"structure", "e_form"}.issubset(df.columns)
        n = len(df)
        positions = np.random.default_rng(a.seed).permutation(n)[:capacity if a.limit else n]
        selected = df.iloc[positions].copy()
        selected["source_row"] = positions
        selected.attrs["total_source_samples"] = n
        if a.limit:
            selected.to_pickle(subset_path)
        del df
    selected = selected.iloc[:a.limit or n].sort_values("source_row")
    metadata, structures = [], {}
    for j, row in enumerate(selected.itertuples()):
        i = int(row.source_row)
        material_id = f"mat_{i:06d}"
        s = to_pymatgen(row.structure)
        target = float(row.e_form)
        if not np.isfinite(target):
            raise ValueError(f"Nonfinite target for {material_id}")
        analyzer = SpacegroupAnalyzer(s, symprec=a.symprec, angle_tolerance=5)
        metadata.append(dict(material_id=material_id, source_row=int(i), target=target,
                             num_atoms=len(s), reduced_formula=s.composition.reduced_formula,
                             crystal_system=analyzer.get_crystal_system(),
                             space_group_number=analyzer.get_space_group_number()))
        structures[material_id] = to_jarvis(s).to_dict()
        if (j + 1) % 100 == 0:
            print(f"Converted {j+1}/{len(selected)}", flush=True)
    table = pd.DataFrame(metadata)
    table.to_csv(out / "metadata.csv", index=False)
    tmp = out / "structures.json.gz.tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(structures, f, allow_nan=False)
    tmp.replace(out / "structures.json.gz")
    manifest.update(total_source_samples=n, prepared_samples=len(table),
                    metadata_sha256=sha256(out / "metadata.csv"),
                    structures_sha256=sha256(out / "structures.json.gz"))
    save_json(old, manifest)
    print(table.groupby("crystal_system").size().to_string(), flush=True)
    print("PREPARATION_PASSED", len(table), "of", n, flush=True)


if __name__ == "__main__":
    main()
