import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from conftest import TinyFFN
from evaluation.multilayer_metrics import (
 aggregate_oracle,adjacent_pair_indices,bootstrap_intervals,canonical_retention,
 experiment_status,representative_schedules,threshold_table,validate_raw_and_summary,
)
from evaluation.layer_comparison import _phase1_comparison,_raw_oracle_files,analyse_multilayer
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
 frame=pd.DataFrame({"layer":[0,0,0],"retention":[.3,.4,.5],"cosine_similarity_mean":[.98,.9991,.9998],"cosine_similarity_p05":[.97,.995,.999],"cosine_similarity_p01":[.96,.99,.998],"relative_l2_mean":[.08,.04,.01],"relative_l2_p95":[.12,.08,.04],"relative_l2_p99":[.15,.09,.05]})
 table=threshold_table(frame);assert table.loc[0,"retention_for_mean_cosine_0.999"]==.4;assert table.loc[0,"retention_for_mean_l2_0.05"]==.4

def test_bootstrap_ci_contains_constant_synthetic_mean():
 frame=pd.DataFrame({"layer":[0]*20,"retention":[.4]*20,"cosine_similarity":[.998]*20,"relative_l2":[.05]*20})
 result=bootstrap_intervals(frame,50,3);assert np.allclose(result.ci_low,result["observed_mean"]);assert np.allclose(result.ci_high,result["observed_mean"]);assert (result.status=="ok").all()

def test_adjacent_matching_never_crosses_prompt_boundaries():
 pairs=adjacent_pair_indices([0,0,1,1,0],[4,5,0,1,9]);assert pairs==[(0,1),(2,3)]

def test_missing_corpus_fails_before_model_loading(tmp_path):
 try: load_corpus(tmp_path/"does-not-exist.jsonl")
 except FileNotFoundError as error: assert "before loading Qwen weights" in str(error)
 else: raise AssertionError("missing calibration corpus was accepted")

def _raw(samples=4):
 rows=[]
 for retention,cosine,l2 in ((.2,.97,.2),(.3,.996,.08),(.30000000000000004,.996,.08),(.4,.9987,.05),(.5,.9994,.03),(.6,.9997,.02),(.75,.9999,.01),(1.,1.,0.)):
  for sample in range(samples): rows.append({"sample_id":sample,"layer":0,"retention":retention,"cosine_similarity":cosine+sample*1e-6,"relative_l2":l2,"mse":l2*l2,"max_absolute_error":l2,"dense_ffn_parameters":100,"selected_ffn_parameters":100*retention,"weight_traffic_fraction":retention})
 return pd.DataFrame(rows)

def test_canonical_retention_coalesces_float_representations():
 assert canonical_retention(.3)==canonical_retention(.30000000000000004)==canonical_retention(np.float32(.3))

def test_regression_valid_retention_groups_never_become_nan():
 fixture=Path(__file__).parent/"fixtures/multilayer_retention_regression.csv";summary=aggregate_oracle(pd.read_csv(fixture),["layer","retention"])
 for retention in (.3,.4,.5): assert np.isfinite(summary.loc[np.isclose(summary.retention,retention),"cosine_similarity_mean"]).all()

def test_unique_sample_count_not_retention_row_count():
 raw=pd.concat([_raw(2000).query("retention != 0.3").groupby(["sample_id","retention"]).first().reset_index()])
 summary=aggregate_oracle(raw,["layer","retention"]);assert (summary.samples==2000).all()

def test_cosine_lower_tail_and_l2_upper_tail():
 raw=_raw(4); target=raw[np.isclose(raw.retention,.4)].copy();target.cosine_similarity=[.90,.95,.99,1.];target.relative_l2=[.01,.02,.1,.4]
 summary=aggregate_oracle(target,["layer","retention"]).iloc[0]
 assert np.isclose(summary.cosine_similarity_p05,np.quantile([.90,.95,.99,1.],.05));assert np.isclose(summary.relative_l2_p95,np.quantile([.01,.02,.1,.4],.95))

def test_full_retention_and_raw_summary_crosscheck_pass():
 raw=_raw();summary=aggregate_oracle(raw,["layer","retention"]);result=validate_raw_and_summary(raw,summary)
 assert result["full_retention_validation"]=="PASS" and result["raw_to_summary_cross_check"]=="PASS"

def test_representative_schedule_uses_exact_layer_requirements():
 rows=[]
 for layer,required in ((0,.4),(1,.5)):
  for retention in (.3,.4,.5,1.): rows.append({"layer":layer,"retention":retention,"cosine_similarity_mean":.9991 if retention>=required else .998,"dense_ffn_parameters":100,"selected_ffn_parameters":100*retention})
 schedules=representative_schedules(pd.DataFrame(rows),2);conservative=schedules[schedules.schedule=="conservative"].iloc[0]
 assert np.isclose(conservative.mean_active_ffn_fraction,.45);assert conservative.estimated_selected_ffn_parameters==90

def test_preliminary_partial_and_verified_status():
 assert experiment_status({0:273,8:273},1000)[0]=="PRELIMINARY"
 assert experiment_status({0:2000,8:2000},1000)[0]=="VERIFIED"
 assert experiment_status({0:2000,8:273},1000)[0]=="PARTIALLY VERIFIED"

def test_raw_file_discovery_excludes_stability_csv_that_caused_nan(tmp_path):
 (tmp_path/"layer_000.csv").touch();(tmp_path/"layer_000.stability.csv").touch()
 assert [path.name for path in _raw_oracle_files(tmp_path)]==["layer_000.csv"]

def test_phase1_comparison_uses_canonical_retention(tmp_path):
 layer0=tmp_path/"layer0";layer0.mkdir();pd.DataFrame({"retention":[.30000000000000004,.4,.5],"cosine_similarity_mean":[.996,.998,.999],"relative_l2_mean":[.08,.05,.03]}).to_csv(layer0/"summary.csv",index=False)
 results=tmp_path/"multilayer";results.mkdir();summary=pd.DataFrame({"layer":[0,0,0],"retention":[.3,.4000000000000001,.5],"cosine_similarity_mean":[.997,.9987,.9994],"relative_l2_mean":[.07,.045,.025]})
 comparison=_phase1_comparison(results,summary);assert comparison.phase1_comparison_available.all();assert np.isclose(comparison.loc[np.isclose(comparison.retention,.4),"cosine_absolute_difference"].iloc[0],.0007)

def test_end_to_end_analysis_ignores_stability_file_and_preserves_key_values(tmp_path):
 oracle=tmp_path/"oracle";oracle.mkdir();fixture=Path(__file__).parent/"fixtures/multilayer_retention_regression.csv";raw=pd.read_csv(fixture);raw.to_csv(oracle/"layer_000.csv",index=False)
 # This reproduces the root cause: the old layer_*.csv glob ingested this
 # schema-incompatible diagnostic file and NumPy propagated its metric NaNs.
 pd.DataFrame({"layer":[0,0,0],"retention":[.3,.4,.5],"frequency_gini":[.1,.2,.3]}).to_csv(oracle/"layer_000.stability.csv",index=False)
 (tmp_path/"metadata.json").write_text(json.dumps({"model":"synthetic","layers":[0],"layer_count":1,"mixers":{"0":"full_attention"},"samples_per_layer":2,"requested_max_tokens_per_layer":2000,"activation_dtype":"float16","seed":42}))
 decision=analyse_multilayer(tmp_path,1000,20,7,"_corrected");summary=pd.read_csv(tmp_path/"layer_summary_corrected.csv")
 assert decision.startswith("PRELIMINARY")
 assert np.isclose(summary.loc[np.isclose(summary.retention,.3),"cosine_similarity_mean"].iloc[0],.9965)
 assert np.isclose(summary.loc[np.isclose(summary.retention,.4),"cosine_similarity_mean"].iloc[0],.9987)
 assert np.isclose(summary.loc[np.isclose(summary.retention,.5),"cosine_similarity_mean"].iloc[0],.9994)
 assert (tmp_path/"report_corrected.md").is_file()
