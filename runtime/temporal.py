"""Prompt-safe temporal mask analysis and persistent neuron-cache simulation."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import torch

PROMPT_KEYS=("prompt_id","prompt_index")
POSITION_KEYS=("token_position","position","token_index")

def recover_sequence_metadata(path:Path):
    path=Path(path)
    if not path.is_file():return None
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()];result=[]
    for sample,row in enumerate(rows):
        prompt=next((row[key] for key in PROMPT_KEYS if key in row),None);position=next((row[key] for key in POSITION_KEYS if key in row),None)
        if prompt is None or position is None:return None
        result.append({"sample":sample,"prompt_id":str(prompt),"token_position":int(position)})
    return result

def ordered_prompts(metadata):
    prompts={}
    for row in metadata:prompts.setdefault(row["prompt_id"],[]).append(row)
    for rows in prompts.values():rows.sort(key=lambda row:row["token_position"])
    return prompts

def contiguous_sequences(metadata):
    for rows in ordered_prompts(metadata).values():
        sequence=[]
        for row in rows:
            if sequence and row["token_position"]!=sequence[-1]["token_position"]+1:
                yield sequence;sequence=[]
            sequence.append(row)
        if sequence:yield sequence

def _summary(values):
    value=torch.tensor(values,dtype=torch.float32);return {"mean":float(value.mean()),"median":float(value.median()),"p05":float(value.quantile(.05)),"p95":float(value.quantile(.95)),"p99":float(value.quantile(.99))}

def temporal_mask_statistics(masks,metadata):
    pairs=[];lifetimes=[];reappear=0;evictions=0
    for rows in contiguous_sequences(metadata):
        active={};recently_evicted={}
        for offset,row in enumerate(rows):
            current=set(masks[row["sample"]].tolist())
            if offset:
                previous=set(masks[rows[offset-1]["sample"]].tolist());intersection=len(current&previous);union=len(current|previous);added=current-previous;removed=previous-current;pairs.append({"intersection":intersection,"union":union,"jaccard":intersection/union,"retained":intersection,"new":len(added),"evicted":len(removed),"fraction_retained":intersection/len(current),"fraction_changed":len(current^previous)/len(current)})
                for neuron in removed:lifetimes.append(active.pop(neuron));recently_evicted[neuron]=offset;evictions+=1
                reappear+=sum(neuron in recently_evicted and offset-recently_evicted[neuron]<=4 for neuron in added)
            for neuron in current:active[neuron]=active.get(neuron,0)+1
        lifetimes.extend(active.values())
    if not pairs:return {"status":"unsupported: no within-prompt consecutive token pairs"}
    keys=pairs[0].keys();life=torch.tensor(lifetimes,dtype=torch.float32);buckets={"1":int((life==1).sum()),"2":int((life==2).sum()),"3-4":int(((life>=3)&(life<=4)).sum()),"5-8":int(((life>=5)&(life<=8)).sum()),"9-16":int(((life>=9)&(life<=16)).sum()),"17+":int((life>=17).sum())}
    return {"pair_count":len(pairs),"pair_metrics":{key:_summary([row[key] for row in pairs]) for key in keys},"lifetimes":{"mean":float(life.mean()),"median":float(life.median()),"p95":float(life.quantile(.95)),"buckets":buckets,"evicted_reappears_within_4_fraction":reappear/max(evictions,1)}}

def neuron_payload(hidden=5120,bits=16):return int(3*hidden*bits/8)

def simulate_lru_cache(masks,metadata,capacity_fraction,neuron_count,bandwidth_gbps=13.):
    capacity=max(1,int(neuron_count*capacity_fraction+.999999));active_k=masks.shape[-1]
    if capacity<active_k:raise ValueError("cache capacity must hold the complete active set")
    events=[]
    for rows in contiguous_sequences(metadata):
        cache=OrderedDict()
        for token,row in enumerate(rows):
            active=[int(value) for value in masks[row["sample"]].tolist()];active_set=set(active);hits=sum(value in cache for value in active);misses=active_k-hits
            for value in active:
                if value in cache:cache.move_to_end(value)
                else:cache[value]=None
            evicted=0
            while len(cache)>capacity:
                victim=next(key for key in cache if key not in active_set);del cache[victim];evicted+=1
            transferred=misses*neuron_payload();events.append({"cold_start":token==0,"hits":hits,"misses":misses,"new_neurons":misses,"evicted":evicted,"bytes_transferred":transferred,"estimated_transfer_ms":transferred/(bandwidth_gbps*1e9)*1000,"cache_size":len(cache)})
    cold=[row for row in events if row["cold_start"]];steady=[row for row in events if not row["cold_start"]]
    summarize=lambda rows:{"tokens":len(rows),"mean_hits":sum(r["hits"] for r in rows)/max(len(rows),1),"mean_misses":sum(r["misses"] for r in rows)/max(len(rows),1),"mean_new_neurons":sum(r["new_neurons"] for r in rows)/max(len(rows),1),"mean_evicted":sum(r["evicted"] for r in rows)/max(len(rows),1),"mean_bytes":sum(r["bytes_transferred"] for r in rows)/max(len(rows),1),"transfer_ms":_summary([r["estimated_transfer_ms"] for r in rows]) if rows else None}
    fp16=capacity*neuron_payload();raw4=capacity*neuron_payload(bits=4)
    return {"capacity_fraction":capacity_fraction,"capacity_neurons":capacity,"maximum_observed_cache_size":max((row["cache_size"] for row in events),default=0),"fp16_payload":{"bytes":fp16,"MiB":fp16/2**20,"GiB":fp16/2**30},"raw_4bit_payload_estimate":{"bytes":raw4,"MiB":raw4/2**20,"GiB":raw4/2**30,"warning":"excludes quantization metadata and runtime buffers"},"all":summarize(events),"cold_start":summarize(cold),"steady_state":summarize(steady),"events":events,"bandwidth_assumption_GBps":bandwidth_gbps}
