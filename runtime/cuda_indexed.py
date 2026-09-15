"""Optional correctness-first CUDA indexed backend, buildable with MSVC/NVCC."""
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME,load_inline

CPP_SOURCE=r'''#include <torch/extension.h>
torch::Tensor indexed_ffn_cuda(torch::Tensor x, torch::Tensor gate, torch::Tensor up, torch::Tensor down, torch::Tensor ids, bool transposed);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &indexed_ffn_cuda, "Indexed FFN forward"); }
'''
CUDA_SOURCE=r'''#include <torch/extension.h>
#include <cuda_fp16.h>
__global__ void gate_up(const half* x,const half* g,const half* u,const int64_t* ids,half* a,int H){int k=blockIdx.x;int n=ids[k];float sg=0,su=0;for(int h=threadIdx.x;h<H;h+=blockDim.x){float xv=__half2float(x[h]);sg+=xv*__half2float(g[n*H+h]);su+=xv*__half2float(u[n*H+h]);}__shared__ float G[256],U[256];G[threadIdx.x]=sg;U[threadIdx.x]=su;__syncthreads();for(int s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s){G[threadIdx.x]+=G[threadIdx.x+s];U[threadIdx.x]+=U[threadIdx.x+s];}__syncthreads();}if(threadIdx.x==0){float v=G[0]/(1.f+expf(-G[0]))*U[0];a[k]=__float2half(v);}}
__global__ void down_kernel(const half* a,const half* d,const int64_t* ids,half* y,int H,int I,int K,bool t){int h=blockIdx.x;float sum=0;for(int k=threadIdx.x;k<K;k+=blockDim.x){int n=ids[k];sum+=__half2float(a[k])*__half2float(t?d[n*H+h]:d[h*I+n]);}__shared__ float S[256];S[threadIdx.x]=sum;__syncthreads();for(int s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)S[threadIdx.x]+=S[threadIdx.x+s];__syncthreads();}if(threadIdx.x==0)y[h]=__float2half(S[0]);}
torch::Tensor indexed_ffn_cuda(torch::Tensor x,torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor ids,bool t){TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat16,"CUDA FP16 required");TORCH_CHECK(x.dim()==2&&x.size(0)==1,"batch=1 required");auto H=x.size(1),I=g.size(0),K=ids.numel();auto a=torch::empty({K},x.options());auto y=torch::empty({1,H},x.options());gate_up<<<K,256>>>(reinterpret_cast<half*>(x.data_ptr()),reinterpret_cast<half*>(g.data_ptr()),reinterpret_cast<half*>(u.data_ptr()),ids.data_ptr<int64_t>(),reinterpret_cast<half*>(a.data_ptr()),H);down_kernel<<<H,256>>>(reinterpret_cast<half*>(a.data_ptr()),reinterpret_cast<half*>(d.data_ptr()),ids.data_ptr<int64_t>(),reinterpret_cast<half*>(y.data_ptr()),H,I,K,t);return y;}
'''

_EXTENSION=None
def cuda_build_diagnostic():
    if not torch.cuda.is_available():return {"available":False,"reason":"PyTorch CUDA unavailable"}
    if CUDA_HOME is None:return {"available":False,"reason":"CUDA toolkit/NVCC not found"}
    if os.name=="nt" and not os.environ.get("VCINSTALLDIR"):return {"available":False,"reason":"MSVC developer environment not detected; run from a VS Developer Prompt"}
    return {"available":True,"cuda_home":str(CUDA_HOME),"platform":os.name}

def load_cuda_indexed(verbose=False):
    global _EXTENSION
    diagnostic=cuda_build_diagnostic()
    if not diagnostic["available"]:return None,diagnostic
    if _EXTENSION is None:
        try:_EXTENSION=load_inline(name="qcc_indexed_ffn",cpp_sources=CPP_SOURCE,cuda_sources=CUDA_SOURCE,functions=None,extra_cuda_cflags=["-O3"],verbose=verbose)
        except Exception as error:return None,{"available":False,"reason":f"CUDA extension build failed: {error}"}
    return _EXTENSION,diagnostic

def cuda_indexed_ffn(x,selected_ids,gate,up,down,transposed_down=False):
    extension,diagnostic=load_cuda_indexed()
    if extension is None:raise RuntimeError(diagnostic["reason"])
    return extension.forward(x,gate,up,down,selected_ids,transposed_down)
