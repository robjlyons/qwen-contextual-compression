#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from predictor.training import train
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results-dir",type=Path,required=True);p.add_argument("--layer",type=int,required=True);p.add_argument("--model",choices=["factorized","mlp"],default="factorized");p.add_argument("--latent-dim",type=int,default=128);p.add_argument("--loss",default="distribution_ce");p.add_argument("--device",default="cuda");p.add_argument("--epochs",type=int,default=100);a,_=p.parse_known_args(argv);name=f"{a.model}_d{a.latent_dim}";print(train(a.results_dir/f"layer_{a.layer:03d}/targets",a.results_dir/f"layer_{a.layer:03d}/{name}",a.model,a.latent_dim,a.loss,a.device,a.epochs))
if __name__=="__main__":main()
