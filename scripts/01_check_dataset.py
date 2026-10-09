"""Build every selected graph; check cached reload, collation, CUDA backward."""
import _bootstrap

import argparse
import time
import torch

from src.data.dataset import CrystalPropertyDataset, make_loader, to_device
from src.models.alignn_wrapper import make_alignn
from src.utils.logging import save_json
from src.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="outputs/dataset_check.json")
    a = p.parse_args()
    seed_everything()
    torch.set_num_threads(4)
    dataset = CrystalPropertyDataset(a.processed)
    start = time.monotonic()
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["graph"].ndata["atom_features"].shape[1] == 92
        if (i + 1) % 100 == 0:
            print(f"Cached {i+1}/{len(dataset)}", flush=True)
    batch = next(iter(make_loader(dataset, a.batch_size)))
    assert torch.equal(batch["graphs"][0].ndata["atom_features"],
                       next(iter(make_loader(dataset, a.batch_size)))["graphs"][0].ndata["atom_features"])
    batch = to_device(batch, "cuda")
    model = make_alignn().cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred = model(batch["graphs"])["out"]
    assert pred.shape == batch["target"].shape
    loss = torch.nn.functional.huber_loss(pred, batch["target"])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    record = dict(status="passed", samples=len(dataset),
                  prediction_shape=list(pred.shape), target_shape=list(batch["target"].shape),
                  loss=float(loss.detach()), gpu=torch.cuda.get_device_name(),
                  graph_cache_namespace=dataset.cache.namespace,
                  elapsed_seconds=time.monotonic()-start)
    save_json(a.output, record)
    print(record, flush=True)


if __name__ == "__main__":
    main()
