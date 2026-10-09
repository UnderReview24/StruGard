"""Gate A: unmodified upstream scalar recipe plus an explicit CUDA backward."""
import _bootstrap

import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import runpy
import subprocess
import sys

import torch

from src.utils.logging import save_json


def main():
    root = _bootstrap.ROOT
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    report = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "alignn_commit": subprocess.check_output(
            ["git", "-C", str(root / "third_party/alignn"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "jarvis_tools": md.version("jarvis-tools"),
        "pymatgen": md.version("pymatgen"),
        "dgl_installed": importlib.util.find_spec("dgl") is not None,
        "status": "running",
    }
    print(json.dumps(report, indent=2), flush=True)
    save_json(out / "environment.json", report)
    (out / "environment.txt").write_text(json.dumps(report, indent=2) + "\n")
    assert torch.cuda.is_available(), "The required GPU gate cannot pass on CPU."
    assert not report["dgl_installed"], "Run this gate in the DGL-free venv."

    from alignn.models.alignn_atomwise_pure import ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig
    from alignn.torch_graph_builder import build_pure_torch_graph, batch_torch_graph_pairs
    from jarvis.core.atoms import Atoms
    import jarvis.db.figshare

    recipe = root / "third_party/alignn/alignn/examples/recipes/knn"
    toy = out / "official_toy_data"
    toy.mkdir(exist_ok=True)
    config = json.loads((recipe / "config_example.json").read_text())
    (toy / "config.json").write_text(json.dumps(config, indent=2))

    # The official generator downloads the whole JARVIS collection just to get
    # a silicon cell. Supply a local primitive Si cell to that one function;
    # execute the original generator, labels, rattling and training unchanged.
    si = Atoms(
        lattice_mat=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        elements=["Si", "Si"], cartesian=False,
    )
    original = jarvis.db.figshare.get_jid_data
    original_cwd = Path.cwd()
    try:
        jarvis.db.figshare.get_jid_data = lambda **kwargs: {"atoms": si.to_dict()}
        os.chdir(toy)
        runpy.run_path(str(recipe / "make_toy_dataset.py"), run_name="__main__")
    finally:
        os.chdir(original_cwd)
        jarvis.db.figshare.get_jid_data = original

    pairs = [build_pure_torch_graph(
        atoms=si, two_body_cutoff=8, three_body_cutoff=3.5,
        max_neighbors=12, atom_features="cgcnn", use_matscipy_topology=False,
    ) for _ in range(2)]
    g, lg = batch_torch_graph_pairs(pairs)
    net = ALIGNNAtomWisePure(ALIGNNAtomWisePureConfig(**config["model"])).cuda()
    net.train()
    pred = net((g.to("cuda"), lg.to("cuda"), torch.tensor(si.lattice_mat).repeat(2, 1, 1).cuda()))["out"]
    assert pred.shape == (2,) and torch.isfinite(pred).all()
    loss = torch.nn.functional.huber_loss(pred, torch.tensor([-1.0, -2.0], device="cuda"))
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    torch.optim.AdamW(net.parameters(), lr=1e-3).step()
    report["cuda_forward_backward"] = "passed"
    report["toy_input"] = "upstream generator with local primitive Si cell"
    del net, pred, loss, grads
    torch.cuda.empty_cache()

    command = [sys.executable, str(root / "third_party/alignn/alignn/train_alignn.py"),
               "--root_dir", str(toy), "--config_name", str(toy / "config.json"),
               "--output_dir", str(out / "official_toy"),
               "--target_key", "target", "--id_key", "jid"]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MPLBACKEND="Agg")
    with (out / "official_toy.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    report.update(status="passed", official_recipe_epochs=config["epochs"])
    save_json(out / "environment.json", report)
    (out / "environment.txt").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: official scalar training and explicit CUDA forward/backward", flush=True)


if __name__ == "__main__":
    main()
