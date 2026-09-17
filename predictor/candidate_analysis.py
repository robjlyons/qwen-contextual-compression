"""Validation-only candidate coverage and restricted-oracle ceiling analysis."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from predictor.candidate_reranker import candidate_topk, retention_count
from predictor.metrics import reconstruct_metrics
from predictor.models import create_predictor
from predictor.reproducibility import tensor_hash


def load_stage1(target_dir: Path, stage1_run: Path, device="cpu"):
    checkpoint_path = stage1_run / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Stage-1 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = torch.load(target_dir / "targets.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = create_predictor(config["kind"], data["inputs"].shape[1], data["scores"].shape[1], config["latent_dim"])
    model.load_state_dict(checkpoint["model"], strict=True);model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model, checkpoint, data, tensor_hash(torch.cat([v.flatten() for v in checkpoint["model"].values()]))


def distribution_summary(values):
    values=values.float().cpu();return {"mean":float(values.mean()),"median":float(values.median()),"p05":float(values.quantile(.05)),"p01":float(values.quantile(.01)),"minimum":float(values.min())}


def reconstruction_summary(metrics):
    cosine,relative=metrics["cosine_similarity"].float(),metrics["relative_l2"].float()
    return {"ffn_cosine":float(cosine.mean()),"ffn_cosine_median":float(cosine.median()),"ffn_cosine_p05":float(cosine.quantile(.05)),"ffn_cosine_p01":float(cosine.quantile(.01)),"relative_l2":float(relative.mean()),"relative_l2_median":float(relative.median()),"relative_l2_p95":float(relative.quantile(.95)),"relative_l2_p99":float(relative.quantile(.99))}


def recommend_candidate_retention(rows, cosine_tolerance=5e-4):
    """Choose on validation: smallest pool within tolerance of the best ceiling."""
    best=max(row["restricted_candidate_oracle"]["ffn_cosine"] for row in rows)
    eligible=[row for row in rows if best-row["restricted_candidate_oracle"]["ffn_cosine"]<=cosine_tolerance]
    return min(eligible,key=lambda row:row["candidate_retention"])["candidate_retention"]


def analyze_candidate_retention(stage1_scores, oracle_scores, activations, down_weight, candidate_retention, final_retention=.5, bias=None):
    if candidate_retention <= final_retention:raise ValueError("candidate retention must be greater than final retention")
    candidates=candidate_topk(stage1_scores,candidate_retention);final_k=retention_count(stage1_scores.shape[-1],final_retention);oracle_ids=torch.topk(oracle_scores,final_k,-1,sorted=False).indices
    candidate_bool=torch.zeros_like(stage1_scores,dtype=torch.bool).scatter_(-1,candidates,True);coverage=candidate_bool.gather(-1,oracle_ids).float().mean(-1)
    candidate_oracle=oracle_scores.gather(-1,candidates);within=torch.topk(candidate_oracle,final_k,-1,sorted=False).indices;restricted_ids=candidates.gather(-1,within)
    restricted=reconstruct_metrics(activations,down_weight,restricted_ids,bias);stage1_ids=torch.topk(stage1_scores,final_k,-1,sorted=False).indices;stage1=reconstruct_metrics(activations,down_weight,stage1_ids,bias)
    selected_mass=oracle_scores.gather(-1,restricted_ids).sum(-1)/oracle_scores.sum(-1).clamp_min(1e-12)
    result={"candidate_retention":candidate_retention,"final_retention":final_retention,"oracle_top50_coverage":distribution_summary(coverage),"restricted_candidate_oracle":{**reconstruction_summary(restricted),"captured_mass":float(selected_mass.mean())},"stage1":reconstruction_summary(stage1)}
    result["headroom_cosine"]=result["restricted_candidate_oracle"]["ffn_cosine"]-result["stage1"]["ffn_cosine"];result["headroom_relative_l2"]=result["stage1"]["relative_l2"]-result["restricted_candidate_oracle"]["relative_l2"];return result


def run_candidate_analysis(results_dir: Path, layer: int, stage1_run: str, candidate_retentions, final_retention=.5, device="cpu", split_name="validation"):
    layer_dir=results_dir/f"layer_{layer:03d}";target_dir=layer_dir/"targets";model,checkpoint,data,stage1_hash=load_stage1(target_dir,layer_dir/stage1_run,device);splits=json.loads((layer_dir/"splits.json").read_text());ids=torch.tensor(splits[split_name],dtype=torch.long);x=data["inputs"].index_select(0,ids).float();mean=checkpoint["mean"].float();std=checkpoint["std"].float().clamp_min(1e-6);oracle=data["raw_scores"].index_select(0,ids).float().to(device);activations=data["gated_activations"].index_select(0,ids).to(device);down=torch.load(target_dir/"down_projection.pt",map_location="cpu",weights_only=True);weight=down["weight"].to(device);bias=None if down["bias"] is None else down["bias"].to(device)
    with torch.inference_mode():stage1_scores=model(((x-mean)/std).to(device));rows=[analyze_candidate_retention(stage1_scores,oracle,activations,weight,value,final_retention,bias) for value in candidate_retentions]
    output={"selection_split":split_name,"stage1_run":stage1_run,"stage1_checkpoint_sha256":stage1_hash,"selection_rule":"smallest validation pool within 0.0005 cosine of best restricted-oracle ceiling","recommended_candidate_retention":recommend_candidate_retention(rows),"rows":rows};out=layer_dir/"candidate_pool_analysis";out.mkdir(exist_ok=True);(out/"analysis.json").write_text(json.dumps(output,indent=2)+"\n");return output
