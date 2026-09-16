"""Numerical, payload, locality, timing, and hardware diagnostics."""
from __future__ import annotations
import importlib.util,platform
import torch
import torch.nn.functional as F

def comparison(reference,value):
    r,v=reference.float(),value.float();difference=v-r;return {"cosine":float(F.cosine_similarity(r,v,dim=-1).mean()),"relative_l2":float(difference.norm()/r.norm().clamp_min(1e-12)),"max_abs_error":float(difference.abs().max()),"mean_abs_error":float(difference.abs().mean())}

def distribution_metrics(dense,sparse):
    cosine=F.cosine_similarity(dense.float(),sparse.float(),dim=-1);relative=(dense.float()-sparse.float()).norm(dim=-1)/dense.float().norm(dim=-1).clamp_min(1e-12);return {"cosine_mean":float(cosine.mean()),"cosine_median":float(cosine.median()),"cosine_p01":float(cosine.quantile(.01)),"cosine_p05":float(cosine.quantile(.05)),"relative_l2_mean":float(relative.mean()),"relative_l2_median":float(relative.median()),"relative_l2_p95":float(relative.quantile(.95)),"relative_l2_p99":float(relative.quantile(.99))}

def weight_payload(hidden,intermediate,k,element_size=2):
    dense=3*hidden*intermediate*element_size;sparse=3*hidden*k*element_size
    describe=lambda value:{"bytes":value,"MiB":value/2**20,"GiB":value/2**30}
    return {"dense_weight_read":describe(dense),"sparse_weight_read":describe(sparse),"resident_weights":describe(dense)}

def index_locality(ids):
    rows=ids.reshape(-1,ids.shape[-1]);runs=[];gaps=[];adjacent=[]
    for row in rows:
        values=torch.sort(row).values;delta=values[1:]-values[:-1];runs.append(int((delta!=1).sum())+1);gaps.append(float(delta.float().mean()) if delta.numel() else 0.);adjacent.append(float((delta==1).float().mean()) if delta.numel() else 0.)
    return {"samples":len(rows),"selected_per_sample":rows.shape[-1],"mean_contiguous_runs":sum(runs)/len(runs),"mean_contiguous_run_length":rows.shape[-1]/(sum(runs)/len(runs)),"mean_index_gap":sum(gaps)/len(gaps),"fraction_adjacent":sum(adjacent)/len(adjacent)}

def hardware_metadata():
    result={"os":platform.platform(),"python":platform.python_version(),"pytorch":torch.__version__,"cuda_version":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"triton_available":importlib.util.find_spec("triton") is not None}
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(torch.cuda.current_device());result.update(gpu=props.name,gpu_total_bytes=props.total_memory)
    return result

def summarize_times(values):
    tensor=torch.tensor(values);return {"mean_ms":float(tensor.mean()),"median_ms":float(tensor.median()),"p05_ms":float(tensor.quantile(.05)),"p50_ms":float(tensor.quantile(.5)),"p95_ms":float(tensor.quantile(.95)),"p99_ms":float(tensor.quantile(.99)),"ffn_evaluations_per_second":1000/float(tensor.mean())}
