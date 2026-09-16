"""ATen-based, current-stream CUDA indexed FFN backends for Windows and Linux."""
from __future__ import annotations
import hashlib,os
import torch
from torch.utils.cpp_extension import CUDA_HOME,load_inline

CPP_SOURCE=r'''#include <torch/extension.h>
at::Tensor indexed_ffn_cuda(at::Tensor x, at::Tensor gate, at::Tensor up, at::Tensor down, at::Tensor ids, bool transposed, int64_t variant);
at::Tensor indexed_gate_up_cuda(at::Tensor x, at::Tensor gate, at::Tensor up, at::Tensor ids, int64_t variant);
at::Tensor indexed_down_cuda(at::Tensor activation, at::Tensor down, at::Tensor ids, int64_t hidden, int64_t intermediate, bool transposed, int64_t variant);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("forward",&indexed_ffn_cuda);m.def("gate_up",&indexed_gate_up_cuda);m.def("down",&indexed_down_cuda);}
'''
CUDA_SOURCE=r'''#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_sum(float v){for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__global__ void gate_up_reference(const half*x,const half*g,const half*u,const int64_t*ids,half*a,int H){int k=blockIdx.x,n=ids[k];float sg=0,su=0;for(int h=threadIdx.x;h<H;h+=blockDim.x){float xv=__half2float(x[h]);sg+=xv*__half2float(g[n*H+h]);su+=xv*__half2float(u[n*H+h]);}__shared__ float G[256],U[256];G[threadIdx.x]=sg;U[threadIdx.x]=su;__syncthreads();for(int s=128;s;s>>=1){if(threadIdx.x<s){G[threadIdx.x]+=G[threadIdx.x+s];U[threadIdx.x]+=U[threadIdx.x+s];}__syncthreads();}if(!threadIdx.x)a[k]=__float2half(G[0]/(1.f+expf(-G[0]))*U[0]);}
__global__ void gate_up_warp8(const half*x,const half*g,const half*u,const int64_t*ids,half*a,int H,int K){int warp=threadIdx.x>>5,lane=threadIdx.x&31,k=blockIdx.x*8+warp;if(k>=K)return;int n=ids[k];float sg=0,su=0;for(int h=lane;h<H;h+=32){float xv=__half2float(x[h]);sg+=xv*__half2float(g[n*H+h]);su+=xv*__half2float(u[n*H+h]);}sg=warp_sum(sg);su=warp_sum(su);if(!lane)a[k]=__float2half(sg/(1.f+expf(-sg))*su);}
__global__ void down_reference(const half*a,const half*d,const int64_t*ids,half*y,int H,int I,int K,bool t){int h=blockIdx.x;float s=0;for(int k=threadIdx.x;k<K;k+=256){int n=ids[k];s+=__half2float(a[k])*__half2float(t?d[n*H+h]:d[h*I+n]);}__shared__ float S[256];S[threadIdx.x]=s;__syncthreads();for(int q=128;q;q>>=1){if(threadIdx.x<q)S[threadIdx.x]+=S[threadIdx.x+q];__syncthreads();}if(!threadIdx.x)y[h]=__float2half(S[0]);}
__global__ void down_warp8(const half*a,const half*d,const int64_t*ids,half*y,int H,int I,int K,bool t){int warp=threadIdx.x>>5,lane=threadIdx.x&31,h=blockIdx.x*8+warp;if(h>=H)return;float s=0;for(int k=lane;k<K;k+=32){int n=ids[k];s+=__half2float(a[k])*__half2float(t?d[n*H+h]:d[h*I+n]);}s=warp_sum(s);if(!lane)y[h]=__float2half(s);}
static void common(const at::Tensor&x,const at::Tensor&g,const at::Tensor&u,const at::Tensor&ids){TORCH_CHECK(x.is_cuda()&&g.is_cuda()&&u.is_cuda()&&ids.is_cuda(),"all tensors must be CUDA");TORCH_CHECK(x.scalar_type()==at::kHalf&&g.scalar_type()==at::kHalf&&u.scalar_type()==at::kHalf,"FP16 required");TORCH_CHECK(ids.scalar_type()==at::kLong,"selected IDs must be int64");TORCH_CHECK(x.is_contiguous()&&g.is_contiguous()&&u.is_contiguous()&&ids.is_contiguous(),"contiguous tensors required");TORCH_CHECK(x.dim()==2&&x.size(0)==1&&g.dim()==2&&u.sizes()==g.sizes()&&x.size(1)==g.size(1),"invalid batch/weight dimensions");TORCH_CHECK(x.device()==g.device()&&x.device()==u.device()&&x.device()==ids.device(),"all tensors must share a device");}
at::Tensor indexed_gate_up_cuda(at::Tensor x,at::Tensor g,at::Tensor u,at::Tensor ids,int64_t v){common(x,g,u,ids);int H=x.size(1),K=ids.numel();auto a=at::empty({K},x.options());cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();if(v==0)gate_up_reference<<<K,256,0,stream>>>((half*)x.data_ptr(),(half*)g.data_ptr(),(half*)u.data_ptr(),ids.data_ptr<int64_t>(),(half*)a.data_ptr(),H);else gate_up_warp8<<<(K+7)/8,256,0,stream>>>((half*)x.data_ptr(),(half*)g.data_ptr(),(half*)u.data_ptr(),ids.data_ptr<int64_t>(),(half*)a.data_ptr(),H,K);C10_CUDA_KERNEL_LAUNCH_CHECK();return a;}
at::Tensor indexed_down_cuda(at::Tensor a,at::Tensor d,at::Tensor ids,int64_t H,int64_t I,bool t,int64_t v){TORCH_CHECK(a.is_cuda()&&d.is_cuda()&&ids.is_cuda()&&a.scalar_type()==at::kHalf&&d.scalar_type()==at::kHalf&&ids.scalar_type()==at::kLong,"CUDA FP16 activation/down and int64 IDs required");TORCH_CHECK(a.is_contiguous()&&d.is_contiguous()&&ids.is_contiguous(),"contiguous tensors required");TORCH_CHECK(a.device()==d.device()&&a.device()==ids.device(),"same device required");TORCH_CHECK(d.dim()==2,"down must be a matrix");if(t){TORCH_CHECK(d.size(0)==I&&d.size(1)==H,"invalid transposed down shape");}else{TORCH_CHECK(d.size(0)==H&&d.size(1)==I,"invalid original down shape");}int K=ids.numel();auto y=at::empty({1,H},a.options());cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();if(v==0)down_reference<<<H,256,0,stream>>>((half*)a.data_ptr(),(half*)d.data_ptr(),ids.data_ptr<int64_t>(),(half*)y.data_ptr(),H,I,K,t);else down_warp8<<<(H+7)/8,256,0,stream>>>((half*)a.data_ptr(),(half*)d.data_ptr(),ids.data_ptr<int64_t>(),(half*)y.data_ptr(),H,I,K,t);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
at::Tensor indexed_ffn_cuda(at::Tensor x,at::Tensor g,at::Tensor u,at::Tensor d,at::Tensor ids,bool t,int64_t v){auto a=indexed_gate_up_cuda(x,g,u,ids,v);return indexed_down_cuda(a,d,ids,x.size(1),g.size(0),t,v);}
'''
VARIANTS={"reference":0,"warp8":1}
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
        digest=hashlib.sha256((CPP_SOURCE+CUDA_SOURCE).encode()).hexdigest()[:10]
        try:_EXTENSION=load_inline(name=f"qcc_indexed_ffn_{digest}",cpp_sources=CPP_SOURCE,cuda_sources=CUDA_SOURCE,functions=None,extra_cuda_cflags=["-O3"],verbose=verbose)
        except Exception as error:return None,{"available":False,"reason":f"CUDA extension build failed: {error}","compiler_claimed_available":True}
    return _EXTENSION,diagnostic

def validate_selected_ids(ids,intermediate,expected_k=None,check_bounds=False):
    if ids.dtype!=torch.int64 or ids.ndim!=1 or not ids.is_contiguous():raise ValueError("selected IDs must be contiguous one-dimensional int64")
    if expected_k is not None and ids.numel()!=expected_k:raise ValueError(f"expected {expected_k} selected IDs, got {ids.numel()}")
    if check_bounds and (int(ids.min())<0 or int(ids.max())>=intermediate):raise ValueError("selected ID out of range")

def cuda_indexed_parts(x,selected_ids,gate,up,down,transposed_down=False,variant="reference"):
    extension,diagnostic=load_cuda_indexed()
    if extension is None:raise RuntimeError(diagnostic["reason"])
    validate_selected_ids(selected_ids,gate.shape[0]);code=VARIANTS[variant];activation=extension.gate_up(x,gate,up,selected_ids,code);output=extension.down(activation,down,selected_ids,x.shape[-1],gate.shape[0],transposed_down,code);return activation,output

def cuda_indexed_ffn(x,selected_ids,gate,up,down,transposed_down=False,variant="reference"):
    extension,diagnostic=load_cuda_indexed()
    if extension is None:raise RuntimeError(diagnostic["reason"])
    validate_selected_ids(selected_ids,gate.shape[0]);return extension.forward(x,gate,up,down,selected_ids,transposed_down,VARIANTS[variant])
