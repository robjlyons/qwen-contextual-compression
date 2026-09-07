"""Layer drift hooks that retain only one prompt's dense reference states."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from evaluation.metrics import output_metrics


def tensor_metrics(dense:Tensor,sparse:Tensor)->dict[str,Tensor]:
    return output_metrics(dense.float().flatten(0,-2),sparse.float().flatten(0,-2))


class LayerStateMonitor:
    def __init__(self,layers:nn.ModuleList):
        self.mode="dense";self.dense_inputs={};self.dense_outputs={};self.rows=[];self.handles=[]
        for index,layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre(index)))
            self.handles.append(layer.register_forward_hook(self._post(index)))
    def _pre(self,index):
        def hook(_module,args):
            value=args[0].detach()
            if self.mode=="dense":self.dense_inputs[index]=value.cpu()
            else:self._record(index,"incoming",self.dense_inputs[index],value)
        return hook
    def _post(self,index):
        def hook(_module,_args,output):
            value=(output[0] if isinstance(output,tuple) else output).detach()
            if self.mode=="dense":self.dense_outputs[index]=value.cpu()
            else:self._record(index,"outgoing",self.dense_outputs[index],value)
        return hook
    def _record(self,index,stage,dense,sparse):
        metrics=tensor_metrics(dense.to(sparse.device),sparse)
        sequence_length=sparse.shape[-2]
        for token in range(len(metrics["cosine_similarity"])):self.rows.append({"layer":index,"stage":stage,"token_position":token%sequence_length,**{key:float(value[token].cpu()) for key,value in metrics.items()}})
    def start_dense(self):self.mode="dense";self.dense_inputs.clear();self.dense_outputs.clear();self.rows.clear()
    def start_sparse(self):self.mode="sparse"
    def close(self):
        for handle in self.handles:handle.remove()
