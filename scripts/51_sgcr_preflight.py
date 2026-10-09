import _bootstrap
import copy
import json
from types import SimpleNamespace
from pathlib import Path
import torch
from src.utils.config import load_config
from src.utils.logging import save_json
from src.continual.sgcr_trainer import SGCRRunner,METHODS
from src.continual.residual_method import ResidualRunner
from src.continual.replay_augmented_rae import partition_memory
from src.models.sp_crystal_residual import SPCrystalResidual
from src.continual.v2_protocol import tensor_digest
from src.data.dataset import make_loader,to_device


def main():
    cfg=load_config('configs/experiment/sgcr.yaml');r=SGCRRunner(cfg,METHODS[1],preflight=True)
    r.stage=r.stages[0];r.original_selection_features(r.stage);r.update_memory()
    r.stage=r.stages[1];r.original_selection_features(r.stage)
    r.replay_train,r.probe=partition_memory(r.memory.ids,2042+r.stage.stage_id,.2)
    ref=SPCrystalResidual(copy.deepcopy(r.model.predictor),dict(shared_lr=.001,residual_lr=.001),with_residual=False).cuda()
    ref.prototypes.fit(0,r.hs(r.stages[0].train))
    optimizer=ref.make_optimizer(cfg['training']['weight_decay'])
    def forward(ids,**kwargs):
        batch=to_device(next(iter(make_loader(r.data.subset(ids),len(ids)))),'cuda')
        return ref(batch['graphs'],r.hs(ids).cuda(),**kwargs),batch['target']
    proxy=SimpleNamespace(model=ref,optimizer=optimizer,stage=r.stage,replay_train=r.replay_train,probe=r.probe,cfg=cfg,batch_forward=forward)
    plan=r.batches(r.stage);equality=[]
    for index in range(2):
        r.train_batch(plan[index],index+1)
        ResidualRunner.train_batch(proxy,plan[index],None,'BASE_ONLY',False)
        mismatch=[n for n,v in r.model.predictor.state_dict().items() if not torch.equal(v,ref.predictor.state_dict()[n])]
        assert not mismatch,mismatch
        equality.append(True)
    r.shift=r.conflict=r.stabilize=True
    r.train_batch(plan[2],3)
    assert r.actual_optimizer_calls==3
    assert tensor_digest(r.model.stable_encoder.state_dict().items())==r.initial_stable
    r.flush()
    result=dict(status='passed',native_verified_baseline_vs_disabled_sgcr_bitwise=equality,
                sgcr_disposable_steps=3,reference_disposable_steps=2,all_weights_discarded=True,
                full_sgcr_diagnostic=r.diagnostics[-1],gradient_scope_names=[n for n,p in r.scope])
    save_json(Path(cfg['sgcr']['output_root'])/'preflight_status.json',result)
    print('SGCR_PREFLIGHT_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
