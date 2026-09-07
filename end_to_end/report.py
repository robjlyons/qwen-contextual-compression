"""Aggregate compact end-to-end rows and render the compression-quality frontier."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _markdown(frame:pd.DataFrame)->str:
    if frame.empty:return "N/A — no completed rows"
    headers=list(frame.columns);lines=["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]
    for row in frame.itertuples(index=False,name=None):lines.append("| "+" | ".join(f"{value:.8g}" if isinstance(value,float) else str(value) for value in row)+" |")
    return "\n".join(lines)


def _bootstrap(data:pd.DataFrame,metric:str,resamples:int,seed:int,mode:str="prompt")->dict:
    rng=np.random.default_rng(seed)
    if mode=="prompt" and "prompt_id" in data:
        units=data.groupby("prompt_id")[metric].mean().to_numpy();unit="prompt"
    else:units=data[metric].to_numpy();unit="token"
    draws=np.array([rng.choice(units,len(units),replace=True).mean() for _ in range(resamples)])
    return {"metric":metric,"bootstrap_unit":unit,"observed_mean":float(data[metric].mean()),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"resamples":resamples}


def analyse_end_to_end(results_dir:Path,resamples:int=500,seed:int=42,top1_threshold:float=.98,ppl_threshold:float=.03)->str:
    teacher=results_dir/"teacher_forced";files=sorted(teacher.glob("*.csv"))
    if not files:raise FileNotFoundError(f"No completed schedule CSVs in {teacher}")
    expansion=json.loads((results_dir/"schedule_expansions.json").read_text());summary=[];category=[];position=[];bootstrap=[]
    for path in files:
      data=pd.read_csv(path);name=path.stem;dense_nll=float(data.dense_nll.mean());sparse_nll=float(data.sparse_nll.mean());dense_ppl=float(np.exp(dense_nll));sparse_ppl=float(np.exp(sparse_nll));ppl_delta=(sparse_ppl-dense_ppl)/dense_ppl
      schedule_key="dense" if name=="dense_baseline" else name
      row={"schedule":schedule_key,"tokens":len(data),"prompts":data.prompt_id.nunique(),"active_ffn_fraction":expansion.get(schedule_key,{"average_ffn_retention":1.})["average_ffn_retention"],"logit_cosine":data.logit_cosine.mean(),"kl_dense_sparse":data.kl_dense_sparse.mean(),"top1_agreement":data.top1_agreement.mean(),"dense_top1_in_sparse_top5":data.dense_top1_in_sparse_top5.mean(),"dense_ppl":dense_ppl,"sparse_ppl":sparse_ppl,"relative_ppl_delta":ppl_delta};summary.append(row)
      for category_name,group in data.groupby("category"):category.append({"schedule":schedule_key,"category":category_name,"tokens":len(group),"logit_cosine":group.logit_cosine.mean(),"kl_dense_sparse":group.kl_dense_sparse.mean(),"top1_agreement":group.top1_agreement.mean(),"dense_ppl":np.exp(group.dense_nll.mean()),"sparse_ppl":np.exp(group.sparse_nll.mean()),"relative_ppl_delta":np.exp(group.sparse_nll.mean()-group.dense_nll.mean())-1})
      for bucket,group in data.groupby("position_bucket"):position.append({"schedule":schedule_key,"position_bucket":bucket,"tokens":len(group),"kl_dense_sparse":group.kl_dense_sparse.mean(),"top1_agreement":group.top1_agreement.mean(),"logit_cosine":group.logit_cosine.mean()})
      for metric in ("logit_cosine","kl_dense_sparse","top1_agreement"):
        bootstrap.append({"schedule":schedule_key,**_bootstrap(data,metric,resamples,seed,"prompt")})
      data=data.assign(relative_nll_change=(data.sparse_nll-data.dense_nll)/data.dense_nll.clip(lower=1e-12));bootstrap.append({"schedule":schedule_key,**_bootstrap(data,"relative_nll_change",resamples,seed,"prompt")})
    summary=pd.DataFrame(summary).sort_values("active_ffn_fraction",ascending=False);generation_summary=results_dir/"generation/generation_summary.csv"
    if generation_summary.is_file():summary=summary.merge(pd.read_csv(generation_summary)[["schedule","first_divergence_median"]],on="schedule",how="left")
    else:summary["first_divergence_median"]=np.nan
    summary.to_csv(results_dir/"summary.csv",index=False);pd.DataFrame(category).to_csv(results_dir/"category_summary.csv",index=False);pd.DataFrame(position).to_csv(results_dir/"position_summary.csv",index=False);pd.DataFrame(bootstrap).to_csv(results_dir/"bootstrap_ci.csv",index=False);summary[["schedule","tokens","dense_ppl","sparse_ppl","relative_ppl_delta"]].to_csv(results_dir/"perplexity_summary.csv",index=False);summary[["schedule","tokens","top1_agreement","dense_top1_in_sparse_top5"]].to_csv(results_dir/"top_token_agreement.csv",index=False)
    hidden=[];local=[]
    for path in results_dir.glob("hidden_state_drift_*.csv"):hidden.append(pd.read_csv(path))
    for path in results_dir.glob("ffn_local_error_*.csv"):local.append(pd.read_csv(path))
    hidden_frame=pd.concat(hidden,ignore_index=True) if hidden else pd.DataFrame();local_frame=pd.concat(local,ignore_index=True) if local else pd.DataFrame()
    def aggregate_drift(frame,groups):
        if frame.empty:return frame
        return frame.groupby(groups,as_index=False).agg(samples=("cosine_similarity","count"),cosine_mean=("cosine_similarity","mean"),cosine_median=("cosine_similarity","median"),cosine_p05=("cosine_similarity",lambda x:x.quantile(.05)),cosine_p01=("cosine_similarity",lambda x:x.quantile(.01)),relative_l2_mean=("relative_l2","mean"),relative_l2_p95=("relative_l2",lambda x:x.quantile(.95)),relative_l2_p99=("relative_l2",lambda x:x.quantile(.99)),mse_mean=("mse","mean"))
    hidden_summary=aggregate_drift(hidden_frame,["schedule","layer","stage"]);local_summary=aggregate_drift(local_frame,["schedule","layer"]);hidden_summary.to_csv(results_dir/"hidden_state_drift.csv",index=False);local_summary.to_csv(results_dir/"ffn_local_error.csv",index=False)
    if len(hidden_summary):hidden_summary[hidden_summary.stage=="incoming"].to_csv(results_dir/"residual_error.csv",index=False)
    plots=results_dir/"plots";plots.mkdir(exist_ok=True)
    for metric,filename,ylabel in (("top1_agreement","top1_agreement_by_schedule.png","Top-1 agreement"),("relative_ppl_delta","perplexity_delta_by_schedule.png","Relative PPL delta"),("kl_dense_sparse","kl_by_schedule.png","KL dense→sparse")):
        plt.scatter(summary.active_ffn_fraction,summary[metric]);
        for _,row in summary.iterrows():plt.annotate(row.schedule,(row.active_ffn_fraction,row[metric]),fontsize=7)
        plt.xlabel("Average active FFN fraction");plt.ylabel(ylabel);plt.tight_layout();plt.savefig(plots/filename,dpi=170);plt.close()
    if len(hidden_summary):
      aggregate=hidden_summary[hidden_summary.stage=="outgoing"]
      for name,group in aggregate.groupby("schedule"):plt.plot(group.layer,group.cosine_mean,label=name)
      plt.xlabel("Layer");plt.ylabel("Hidden-state cosine");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(plots/"hidden_drift_by_layer.png",dpi=170);plt.close()
      incoming=hidden_summary[hidden_summary.stage=="incoming"]
      for name,group in incoming.groupby("schedule"):plt.plot(group.layer,group.relative_l2_mean,label=name)
      plt.xlabel("Layer");plt.ylabel("Incoming residual relative L2");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(plots/"residual_error_by_layer.png",dpi=170);plt.close()
    if len(local_summary):
      for name,group in local_summary.groupby("schedule"):plt.plot(group.layer,group.relative_l2_mean,label=name)
      plt.xlabel("Layer");plt.ylabel("Local FFN relative L2 on sparse input");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(plots/"local_ffn_error_by_layer.png",dpi=170);plt.close()
    category_frame=pd.DataFrame(category)
    if len(category_frame):
      for name,group in category_frame.groupby("schedule"):plt.plot(group.category,group.top1_agreement,"o",label=name)
      plt.xticks(rotation=35,ha="right");plt.ylabel("Top-1 agreement");plt.legend(fontsize=6);plt.tight_layout();plt.savefig(plots/"category_sensitivity.png",dpi=170);plt.close()
    measured_ok=lambda name:name in set(summary.schedule) and float(summary.loc[summary.schedule==name,"top1_agreement"].iloc[0])>=top1_threshold and float(summary.loc[summary.schedule==name,"relative_ppl_delta"].iloc[0])<=ppl_threshold
    if measured_ok("all_moderate"):outcome="OUTCOME A — MODERATE ALL-LAYER SPARSITY SURVIVES";recommendation="Train the lightweight importance predictor."
    elif measured_ok("all_conservative"):outcome="OUTCOME B — CONSERVATIVE SPARSITY SURVIVES, MODERATE DOES NOT";recommendation="Optimise per-layer retention under an end-to-end constraint."
    elif measured_ok("measured_conservative") or measured_ok("measured_moderate"):outcome="OUTCOME C — MEASURED-LAYER SPARSITY WORKS, ALL-LAYER SCHEDULE DOES NOT";recommendation="Measure all 64 isolated FFN curves before extrapolating."
    else:outcome="OUTCOME D — ERRORS ACCUMULATE STRONGLY";recommendation="Derive stricter layer-specific error budgets and identify sensitive layers."
    report=f"""# End-to-End Oracle Sparse FFN Propagation

## Main results
{_markdown(summary)}

## Interpretation
Final logits, KL, token agreement, perplexity, and propagated hidden-state drift—not isolated FFN cosine—are the source of truth. Engineering thresholds are configurable (`top1 >= {top1_threshold}`, relative PPL increase `<= {ppl_threshold}`) and are not universal quality laws.

## Required answers
The compact CSVs and plots show whether drift grows or remains bounded; final-logit divergence; conservative/moderate behaviour; layer-specific versus uniform schedules; measured-only versus interpolated schedules; and perplexity cost versus active FFN fraction.

## Caveats
This oracle evaluates dense gate/up projections and ordinary dense PyTorch operations. **No runtime speedup or actual VRAM reduction is claimed.** Payload equivalents exclude quantisation metadata and runtime buffers. Teacher-forced and generation conclusions are separate.

## Decision
### {outcome}
Recommended next step: {recommendation}
""";(results_dir/"report.md").write_text(report);return outcome
