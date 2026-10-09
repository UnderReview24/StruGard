import _bootstrap
import argparse
from pathlib import Path

from src.continual.strategies import setup_experiment,shared_stage1
from src.continual.v2_protocol import load_or_create_prefix_cache,build_protocol
from src.models.alignn_wrapper import make_alignn
from src.models.v2 import SPCrystalV2
from src.utils.config import load_config
from src.utils.logging import save_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment/v2.yaml');a=p.parse_args()
    cfg=load_config(a.config)
    stream,data=setup_experiment(cfg)
    if len(data)!=5000 or len(stream)!=3 or cfg['experiment']['seed']!=42:
        raise ValueError('V2 is restricted to the fixed 5k / 3-domain / seed-42 benchmark')
    root=Path(cfg['experiment']['run_root'])
    plain=make_alignn(cfg['model'])
    shared_stage1(plain,list(stream)[0],root/'shared_stage1',cfg)
    model=SPCrystalV2(plain,cfg['v2']).cuda().eval()
    records,signature=load_or_create_prefix_cache(model.encoder,data,cfg['v2']['cache'])
    protocol=build_protocol(cfg,stream,data,records,signature)
    groups=model.parameter_groups()
    save_json(root/'parameter_split.json',{k:dict(parameters=sum(p.numel() for _,p in entries),
        names=[n for n,p in entries],learning_rate=cfg['v2']['slow_lr'] if k=='slow' else (cfg['v2']['expert_lr'] if k=='plastic' else 0.))
        for k,entries in groups.items()})
    print('V2_PROTOCOL_READY',[(s['stage'],s['optimizer_slots'],s['current_exposures'],s['replay_exposures']) for s in protocol['stages']],flush=True)
