"""Optional fused-latent FP16 CUDA selector; exact Top-K remains PyTorch."""
from __future__ import annotations
import hashlib
import torch
from torch.utils.cpp_extension import CUDA_HOME,load_inline
CPP=r'''#include <torch/extension.h>
at::Tensor selector_scores_cuda(at::Tensor x,at::Tensor enc_w,at::Tensor enc_b,at::Tensor f1w,at::Tensor f1b,at::Tensor f2w,at::Tensor f2b,at::Tensor lnw,at::Tensor lnb,at::Tensor outw,at::Tensor outb,double eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("scores",&selector_scores_cuda);}
'''
CUDA=r'''#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ float ws(float v){for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__global__ void latent_kernel(const half*x,const half*ew,const half*eb,const half*f1w,const half*f1b,const half*f2w,const half*f2b,const half*lw,const half*lb,half*z,int H,float eps){__shared__ float r[32],u[32];int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int q=warp;q<32;q+=8){float s=0;for(int h=lane;h<H;h+=32)s+=__half2float(x[h])*__half2float(ew[q*H+h]);s=ws(s);if(!lane){s+=__half2float(eb[q]);r[q]=s/(1.f+expf(-s));}}__syncthreads();if(threadIdx.x==0){for(int q=0;q<32;q++){float s=__half2float(f1b[q]);for(int j=0;j<32;j++)s+=__half2float(f1w[q*32+j])*r[j];u[q]=s/(1.f+expf(-s));}float mean=0,var=0;for(int q=0;q<32;q++){float s=__half2float(f2b[q]);for(int j=0;j<32;j++)s+=__half2float(f2w[q*32+j])*u[j];u[q]=s+r[q];mean+=u[q];}mean/=32;for(int q=0;q<32;q++){float d=u[q]-mean;var+=d*d;}float inv=rsqrtf(var/32+eps);for(int q=0;q<32;q++)z[q]=__float2half((u[q]-mean)*inv*__half2float(lw[q])+__half2float(lb[q]));}}
__global__ void head_kernel(const half*z,const half*w,const half*b,half*scores,int I){int n=blockIdx.x,lane=threadIdx.x;float s=lane<32?__half2float(z[lane])*__half2float(w[n*32+lane]):0;s=ws(s);if(!lane)scores[n]=__float2half(s+__half2float(b[n]));}
at::Tensor selector_scores_cuda(at::Tensor x,at::Tensor ew,at::Tensor eb,at::Tensor f1w,at::Tensor f1b,at::Tensor f2w,at::Tensor f2b,at::Tensor lw,at::Tensor lb,at::Tensor ow,at::Tensor ob,double eps){TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.dim()==2&&x.size(0)==1,"CUDA FP16 batch=1 required");TORCH_CHECK(ew.size(0)==32&&f1w.dim()==2&&f1w.size(0)==32&&f1w.size(1)==32&&f2w.dim()==2&&f2w.size(0)==32&&f2w.size(1)==32&&ow.size(1)==32,"latent=32 required");auto z=at::empty({32},x.options());auto scores=at::empty({1,ow.size(0)},x.options());auto stream=c10::cuda::getCurrentCUDAStream().stream();latent_kernel<<<1,256,0,stream>>>((half*)x.data_ptr(),(half*)ew.data_ptr(),(half*)eb.data_ptr(),(half*)f1w.data_ptr(),(half*)f1b.data_ptr(),(half*)f2w.data_ptr(),(half*)f2b.data_ptr(),(half*)lw.data_ptr(),(half*)lb.data_ptr(),(half*)z.data_ptr(),x.size(1),eps);head_kernel<<<ow.size(0),32,0,stream>>>((half*)z.data_ptr(),(half*)ow.data_ptr(),(half*)ob.data_ptr(),(half*)scores.data_ptr(),ow.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return scores;}
'''
_EXT=None
def load_cuda_selector():
 global _EXT
 if not torch.cuda.is_available() or CUDA_HOME is None:return None,{"available":False,"reason":"CUDA device/toolkit unavailable"}
 if _EXT is None:
  name="qcc_cuda_selector_"+hashlib.sha256((CPP+CUDA).encode()).hexdigest()[:10]
  try:_EXT=load_inline(name=name,cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cuda_cflags=["-O3"],verbose=False)
  except Exception as error:return None,{"available":False,"reason":f"CUDA selector build failed: {error}","compiler_claimed_available":True}
 return _EXT,{"available":True}
class CudaSelector:
 def __init__(self,folded_selector):self.source=folded_selector;self.k=folded_selector.predictor.neurons.out_features//2
 def scores(self,x):
  extension,status=load_cuda_selector()
  if extension is None:raise RuntimeError(status["reason"])
  p=self.source.predictor;return extension.scores(x.half().contiguous(),p.encoder.weight,p.encoder.bias,p.residual_fc1.weight,p.residual_fc1.bias,p.residual_fc2.weight,p.residual_fc2.bias,p.layer_norm.weight,p.layer_norm.bias,p.neurons.weight,p.neurons.bias,p.layer_norm.eps)
 def select(self,x):return torch.topk(self.scores(x),self.k,-1,sorted=False).indices
