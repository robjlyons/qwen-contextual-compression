#!/usr/bin/env python
"""Optional CUDA diagnostic for bounded down-column norm preparation."""
import argparse,json
import _bootstrap  # noqa: F401
import torch
from end_to_end.oracle_sparse_mlp import compute_down_column_norms_chunked

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--out-features",type=int,default=512);p.add_argument("--intermediate-size",type=int,default=1740);p.add_argument("--norm-chunk-columns",type=int,default=256);p.add_argument("--device",default="cuda:0");a,_=p.parse_known_args(argv)
 if not torch.cuda.is_available():raise RuntimeError("CUDA is required for this diagnostic")
 device=torch.device(a.device);weight=torch.randn(a.out_features,a.intermediate_size,device=device,dtype=torch.float16);torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);before=torch.cuda.memory_allocated(device);norms=compute_down_column_norms_chunked(weight,a.norm_chunk_columns);torch.cuda.synchronize(device);peak=torch.cuda.max_memory_allocated(device)
 print(json.dumps({"shape":list(weight.shape),"chunk_columns":a.norm_chunk_columns,"estimated_chunk_fp32_mib":a.out_features*min(a.intermediate_size,a.norm_chunk_columns)*4/2**20,"measured_peak_delta_mib":(peak-before)/2**20,"norm_bytes":norms.numel()*norms.element_size()},indent=2))
if __name__=="__main__":main()
