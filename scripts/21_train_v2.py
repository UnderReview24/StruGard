import _bootstrap
import argparse
from src.continual.v2_trainer import V2Trainer,VARIANTS
from src.utils.config import load_config

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/experiment/v2.yaml')
    p.add_argument('--variant',choices=VARIANTS,default='full')
    a=p.parse_args()
    V2Trainer(load_config(a.config),a.variant).run()
