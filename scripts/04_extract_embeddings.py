import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from src.continual.strategies import setup_experiment
from src.data.embeddings import extract_embeddings
from src.models.alignn_wrapper import ALIGNNEncoder, make_alignn
from src.utils.config import load_config


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment/pilot.yaml")
    p.add_argument("--checkpoint", default="outputs/shift_validation/d1_model/best.pt")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--output", default="data/embeddings/stage1_train.npz")
    a = p.parse_args()
    assert json.loads(Path("outputs/pilot/random_replay/status.json").read_text())["status"] == "passed"
    cfg = load_config(a.config)
    stream, data = setup_experiment(cfg)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = make_alignn(state["model_config"])
    model.load_state_dict(state["model"])
    encoder = ALIGNNEncoder(model).cuda()
    ids = list(getattr(list(stream)[a.stage-1], a.split))
    h = extract_embeddings(encoder, data, ids, cfg["training"]["batch_size"])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, material_id=np.asarray(ids), embedding=h.numpy())
    print("ENCODER_EXTRACTION_PASSED", tuple(h.shape), flush=True)
