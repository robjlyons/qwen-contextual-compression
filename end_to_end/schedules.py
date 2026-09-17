"""Validated measured-only, interpolated, and uniform retention schedules."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml


def expand_schedule(spec:dict,num_layers:int,interpolation_override:str|None=None)->list[float]:
    kind=spec["type"]
    if kind=="uniform": values=[float(spec["retention"])]*num_layers
    elif kind=="measured_only":
        values=[float(spec.get("default_retention",1.))]*num_layers
        for layer,value in spec["layers"].items(): values[int(layer)]=float(value)
    elif kind=="interpolated":
        knots={int(k):float(v) for k,v in spec["knots"].items()}; xs=np.array(sorted(knots)); ys=np.array([knots[x] for x in xs]); mode=interpolation_override or spec.get("interpolation","linear")
        if xs[0]!=0 or xs[-1]!=num_layers-1: raise ValueError("interpolated schedule knots must include first and last layer")
        if mode=="linear": values=np.interp(np.arange(num_layers),xs,ys).tolist()
        elif mode=="nearest": values=[knots[int(xs[np.abs(xs-layer).argmin()])] for layer in range(num_layers)]
        else: raise ValueError("interpolation must be linear or nearest")
    else: raise ValueError(f"unknown schedule type {kind!r}")
    if len(values)!=num_layers or any(not 0 < value <= 1 for value in values): raise ValueError("schedule retentions must lie in (0, 1]")
    return values


def load_schedules(path:Path,num_layers:int,names:list[str]|None=None,interpolation_override:str|None=None)->dict[str,list[float]]:
    document=json.loads(path.read_text()) if path.suffix.lower()==".json" else yaml.safe_load(path.read_text()); selected=names or list(document)
    missing=set(selected)-set(document)
    if missing: raise ValueError(f"unknown schedules: {sorted(missing)}")
    return {name:expand_schedule(document[name],num_layers,interpolation_override) for name in selected}


def active_parameter_accounting(schedule:list[float],ffn_parameters:list[int],total_model_parameters:int)->dict:
    if len(schedule)!=len(ffn_parameters): raise ValueError("schedule and FFN parameter counts differ")
    dense_ffn=sum(ffn_parameters); selected=sum(p*r for p,r in zip(ffn_parameters,schedule)); non_ffn=total_model_parameters-dense_ffn; active=non_ffn+selected
    result={"average_ffn_retention":selected/dense_ffn,"dense_ffn_parameters":dense_ffn,"selected_ffn_parameters":selected,"skipped_ffn_parameters":dense_ffn-selected,"non_ffn_parameters":non_ffn,"estimated_active_model_parameters":active}
    for name,bits in {"bf16":16,"8bit":8,"4bit":4,"3bit":3,"2bit":2}.items(): result[f"active_{name}_bytes"]=active*bits/8
    return result

