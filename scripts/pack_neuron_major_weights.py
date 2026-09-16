#!/usr/bin/env python
import argparse
from pathlib import Path
import _bootstrap
from runtime.weights import save_neuron_major_layout
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);print(save_neuron_major_layout(**vars(p.parse_args(argv))))
if __name__=="__main__":main()
