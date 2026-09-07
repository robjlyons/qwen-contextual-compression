"""Reversible replacement of only discovered language-model MLP modules."""
from __future__ import annotations
from torch import nn
from extract.extract_ffn import locate_ffn
from end_to_end.oracle_sparse_mlp import OracleSparseMLP


def _parent_and_name(model:nn.Module,path:str):
    parent_path,name=path.rsplit(".",1); return model.get_submodule(parent_path),name


class OracleMLPPatcher:
    def __init__(self,model:nn.Module,telemetry=None):
        self.model=model; self.wrappers={}; self.originals={}
        first=locate_ffn(model,0); layers=model.get_submodule(first.layers_path)
        for index in range(len(layers)):
            found=locate_ffn(model,index); parent,name=_parent_and_name(model,found.path); original=getattr(parent,name)
            wrapper=OracleSparseMLP(original,index,1.,telemetry); setattr(parent,name,wrapper); self.wrappers[index]=wrapper; self.originals[index]=(parent,name,original)

    def dense(self)->None:
        for wrapper in self.wrappers.values(): wrapper.set_mode(False)

    def apply_schedule(self,schedule:list[float])->None:
        if len(schedule)!=len(self.wrappers): raise ValueError(f"schedule has {len(schedule)} layers; model has {len(self.wrappers)}")
        for index,wrapper in self.wrappers.items(): wrapper.set_mode(True,schedule[index])

    def restore(self)->None:
        for index,wrapper in self.wrappers.items():
            wrapper.close(); parent,name,original=self.originals[index]; setattr(parent,name,original)
