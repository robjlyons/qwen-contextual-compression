import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from conftest import TinyFFN
from evaluation.multilayer_metrics import adjacent_pair_indices,bootstrap_intervals,minimum_retention,threshold_table
from extract.extract_ffn import explicit_dense_ffn,locate_ffn
from extract.multilayer_capture import load_corpus,mixer_type,validate_layer_sample_alignment

class DeltaLayer(nn.Module):
 def __init__(self): super().__init__();self.linear_attn=nn.Identity();self.mlp=TinyFFN()
class AttentionLayer(nn.Module):
 def __init__(self): super().__init__();self.self_attn=nn.Identity();self.mlp=TinyFFN()
class Multi(nn.Module):
 def __init__(self): super().__init__();self.layers=nn.ModuleList([DeltaLayer(),AttentionLayer()])

def test_every_selected_layer_explicit_ffn_matches_original():
 model=Multi();x=torch.randn(5,4)
 for layer in (0,1):
  found=locate_ffn(model,layer);torch.testing.assert_close(found.module(x),explicit_dense_ffn(x,found))

def test_mixer_type_is_recorded_from_real_child_modules():
 assert mixer_type(DeltaLayer())=="gated_deltanet"
 assert mixer_type(AttentionLayer())=="full_attention"

def test_sample_ids_align_across_layer_shards(tmp_path):
 for layer in (0,8):
  directory=tmp_path/f"layer_{layer:03d}";directory.mkdir();torch.save({"sample_ids":torch.tensor([10,11]),"inputs":torch.randn(2,4)},directory/"shard_00000.pt")
 assert validate_layer_sample_alignment(tmp_path,[0,8])==2

def test_sample_alignment_detects_resume_corruption(tmp_path):
 for layer,ids in ((0,[1,2]),(8,[1,3])):
  directory=tmp_path/f"layer_{layer:03d}";directory.mkdir();torch.save({"sample_ids":torch.tensor(ids)},directory/"shard_00000.pt")
 try: validate_layer_sample_alignment(tmp_path,[0,8])
 except RuntimeError as error: assert "Sample IDs differ" in str(error)
 else: raise AssertionError("misaligned resumed shards were accepted")

def test_threshold_selection_logic():
 frame=pd.DataFrame({"layer":[0,0,0],"retention":[.3,.4,.5],"cosine_similarity_mean":[.98,.9991,.9998],"relative_l2_mean":[.08,.04,.01],"cosine_similarity_p95":[.99,.995,.999],"cosine_similarity_p99":[.995,.999,.9999]})
 table=threshold_table(frame);assert table.loc[0,"keep_mean_cosine_0.999"]==.4;assert table.loc[0,"keep_mean_l2_0.05"]==.4

def test_bootstrap_ci_contains_constant_synthetic_mean():
 frame=pd.DataFrame({"layer":[0]*20,"retention":[.4]*20,"cosine_similarity":[.998]*20,"relative_l2":[.05]*20})
 result=bootstrap_intervals(frame,50,3);assert np.allclose(result.ci95_low,result["mean"]);assert np.allclose(result.ci95_high,result["mean"])

def test_adjacent_matching_never_crosses_prompt_boundaries():
 pairs=adjacent_pair_indices([0,0,1,1,0],[4,5,0,1,9]);assert pairs==[(0,1),(2,3)]

def test_missing_corpus_fails_before_model_loading(tmp_path):
 try: load_corpus(tmp_path/"does-not-exist.jsonl")
 except FileNotFoundError as error: assert "before loading Qwen weights" in str(error)
 else: raise AssertionError("missing calibration corpus was accepted")
