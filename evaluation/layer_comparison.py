"""Multi-layer tables, plots, Phase-1 comparison, and automatic decision report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.multilayer_metrics import aggregate_oracle,bootstrap_intervals,position_bucket,threshold_table,working_set_estimates

PHASE1={.3:.9965,.4:.9985,.5:.99935}

def _fixed(summary,metric,ratio):
    data=summary[np.isclose(summary.retention,ratio)].set_index("layer"); return data[metric]

def analyse_multilayer(results_dir:Path,min_report_samples:int=1000,bootstrap_resamples:int=500)->str:
    metadata=json.loads((results_dir/"metadata.json").read_text()); oracle_dir=results_dir/"oracle"
    files=sorted(oracle_dir.glob("layer_*.csv"))
    if not files: raise FileNotFoundError(f"No layer_*.csv oracle results found in {oracle_dir}")
    rows=pd.concat([pd.read_csv(path) for path in files],ignore_index=True); samples=pd.read_json(results_dir/"samples.jsonl",lines=True); rows=rows.merge(samples,on=["sample_id","prompt_id","token_position","token_id"],how="left",validate="many_to_one")
    all_summary=aggregate_oracle(rows,["layer","retention"]); validation=aggregate_oracle(rows[rows.split=="validation"],["layer","retention"])
    # Parameter/traffic columns are constant within layer-retention groups.
    constants=rows.groupby(["layer","retention"],as_index=False)[["dense_ffn_parameters","selected_ffn_parameters","weight_traffic_fraction"]].first(); all_summary=all_summary.merge(constants,on=["layer","retention"]); validation=validation.merge(constants,on=["layer","retention"])
    all_summary.to_csv(results_dir/"layer_summary.csv",index=False); thresholds=threshold_table(all_summary); thresholds.to_csv(results_dir/"threshold_summary.csv",index=False)
    validation.to_csv(results_dir/"validation_layer_summary.csv",index=False)
    category=aggregate_oracle(rows,["layer","category","retention"]); category.to_csv(results_dir/"category_summary.csv",index=False)
    rows["position_bucket"]=rows.token_position.map(position_bucket); positions=aggregate_oracle(rows,["layer","position_bucket","retention"]); positions.to_csv(results_dir/"token_position_summary.csv",index=False)
    bootstrap=bootstrap_intervals(rows,bootstrap_resamples); bootstrap.to_csv(results_dir/"bootstrap_ci.csv",index=False)
    stability_files=sorted(oracle_dir.glob("layer_*.stability.csv")); stability=pd.concat([pd.read_csv(x) for x in stability_files],ignore_index=True) if stability_files else pd.DataFrame(); stability.to_csv(results_dir/"selection_stability.csv",index=False)
    adjacent=stability[[c for c in ["layer","retention","adjacent_pairs","adjacent_jaccard","random_jaccard","adjacent_reuse_fraction"] if c in stability]] if len(stability) else pd.DataFrame(); adjacent.to_csv(results_dir/"adjacent_token_overlap.csv",index=False)
    estimates=working_set_estimates(all_summary,thresholds,int(metadata["layer_count"])); estimates.to_csv(results_dir/"model_wide_estimate.csv",index=False)
    layers=sorted(all_summary.layer.unique()); mixers={int(k):v for k,v in metadata["mixers"].items()}; main=pd.DataFrame({"layer":layers,"mixer":[mixers.get(x,"unknown") for x in layers]})
    for ratio in (.2,.3,.4,.5): main[f"cosine_{ratio}"]=main.layer.map(_fixed(all_summary,"cosine_similarity_mean",ratio))
    main["l2_0.4"]=main.layer.map(_fixed(all_summary,"relative_l2_mean",.4)); main=main.merge(thresholds[["layer","keep_mean_cosine_0.999","keep_mean_l2_0.05"]],on="layer",how="left")
    plots=results_dir/"plots"; plots.mkdir(exist_ok=True)
    for metric,name,ylabel in [("cosine_similarity_mean","depth_cosine.png","Mean cosine"),("relative_l2_mean","depth_l2.png","Mean relative L2")]:
      for ratio in (.2,.3,.4,.5):
        data=all_summary[np.isclose(all_summary.retention,ratio)]; plt.plot(data.layer,data[metric],"o-",label=f"{ratio:.0%}")
      plt.xlabel("Layer");plt.ylabel(ylabel);plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(plots/name,dpi=170);plt.close()
    for column,name,ylabel in [("keep_mean_cosine_0.999","required_retention_999.png","Retention for mean cosine ≥ .999"),("keep_mean_l2_0.05","required_retention_l2_5.png","Retention for mean L2 ≤ 5%")]:
        plt.plot(thresholds.layer,thresholds[column],"o-");plt.xlabel("Layer");plt.ylabel(ylabel);plt.tight_layout();plt.savefig(plots/name,dpi=170);plt.close()
    pivot=all_summary.pivot(index="layer",columns="retention",values="cosine_similarity_mean");plt.imshow(pivot,aspect="auto",vmin=min(.99,float(np.nanmin(pivot))),vmax=1,cmap="viridis");plt.xticks(range(len(pivot.columns)),[f"{x:.0%}" for x in pivot.columns]);plt.yticks(range(len(pivot.index)),pivot.index);plt.colorbar(label="Mean cosine");plt.tight_layout();plt.savefig(plots/"retention_heatmap.png",dpi=170);plt.close()
    tail=all_summary.pivot(index="layer",columns="retention",values="relative_l2_p95");plt.imshow(tail,aspect="auto",cmap="magma");plt.xticks(range(len(tail.columns)),[f"{x:.0%}" for x in tail.columns]);plt.yticks(range(len(tail.index)),tail.index);plt.colorbar(label="P95 relative L2");plt.tight_layout();plt.savefig(plots/"tail_risk_heatmap.png",dpi=170);plt.close()
    at40=category[np.isclose(category.retention,.4)];
    for category_name,data in at40.groupby("category"): plt.plot(data.layer,data.cosine_similarity_mean,"o-",label=category_name)
    plt.xlabel("Layer");plt.ylabel("40% mean cosine");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(plots/"category_comparison.png",dpi=170);plt.close()
    if len(adjacent):
      for ratio,data in adjacent.groupby("retention"): plt.plot(data.layer,data.adjacent_reuse_fraction,"o-",label=f"{ratio:.0%}")
      plt.xlabel("Layer");plt.ylabel("Adjacent selected-neuron reuse");plt.legend();plt.tight_layout();plt.savefig(plots/"adjacent_overlap.png",dpi=170);plt.close()
    layer0=all_summary[all_summary.layer==0]; phase=[]
    for ratio,old in PHASE1.items():
        found=layer0[np.isclose(layer0.retention,ratio)]; new=float(found.cosine_similarity_mean.iloc[0]) if len(found) else np.nan; phase.append({"retention":ratio,"phase1_approximate":old,"new":new,"difference":new-old})
    phase=pd.DataFrame(phase); phase.to_csv(results_dir/"phase1_comparison.csv",index=False)
    counts=rows.groupby("layer").sample_id.nunique(); status="VERIFIED" if counts.min()>=min_report_samples else "PRELIMINARY"
    required=thresholds["keep_mean_cosine_0.999"]; strong=(required<=.5).sum(); fraction=strong/len(required); spread=float(required.max()-required.min()) if required.notna().all() else np.inf
    phase40=phase[np.isclose(phase.retention,.4)].difference.iloc[0]
    if status=="PRELIMINARY": outcome="NO VERIFIED OUTCOME — INSUFFICIENT SAMPLES"; recommendation="Capture at least the configured minimum samples per layer before choosing a research direction."
    elif fraction>=.75 and spread<=.2: outcome="OUTCOME A — MODEL-WIDE CONTEXTUAL SPARSITY STRONG"; recommendation="Train a lightweight individual-neuron importance predictor."
    elif fraction>=.5: outcome="OUTCOME B — LAYER-DEPENDENT SPARSITY"; recommendation="Design per-layer retention budgets and train predictors only where worthwhile."
    elif strong>0: outcome="OUTCOME C — EARLY-LAYER / LOCAL PHENOMENON"; recommendation="Restrict contextual sparsity to qualifying layers and investigate mathematical compression elsewhere."
    else: outcome="OUTCOME D — INITIAL RESULT DOES NOT GENERALISE"; recommendation="Stop predictor work and pivot to mathematical weight compression."
    report=f"""# Multi-Layer Oracle Validation

**Status: {status}** — minimum samples per tested layer: {int(counts.min())}; required for verified conclusions: {min_report_samples}.

## Experimental setup
Model: `{metadata['model']}`; revision: `{metadata.get('revision')}`; layers: {layers}; mixers: {mixers}; tokens/layer: {metadata['samples_per_layer']}; corpus categories: {metadata.get('corpus_categories')}; activation dtype: {metadata['activation_dtype']}; hardware: `{metadata.get('hardware')}`; seed: {metadata['seed']}.

## Previous result
The earlier layer-0 run reported approximately 0.9965/0.9985/0.99935 cosine at 30%/40%/50%. Larger-corpus comparison:
{phase.to_markdown(index=False)}

## Layer results
{main.to_markdown(index=False)}

## Depth, category, and token-position behaviour
Raw, validation-only, category, and position-bucket tables are saved alongside this report. Plots report measured patterns without semantic causal claims.

## Selection stability and adjacent-token reuse
{adjacent.to_markdown(index=False) if len(adjacent) else 'Not available.'}

## Model-wide theoretical implications
{estimates.to_markdown(index=False)}
These are representative-layer estimates, not interpolation of untested-layer errors and **not actual runtime VRAM**.

## Confidence intervals
{bootstrap.to_markdown(index=False)}

## Caveats
- The oracle uses full FFN activations unavailable cheaply at inference time; no runtime speedup exists.
- Arbitrary neuron access may be hardware inefficient, and FFN reconstruction does not guarantee generation quality.
- Errors may compound across layers. Attention/DeltaNet, embeddings, and output weights remain dense.
- Model VRAM is not the oracle-selected FFN working-set size.

## Decision
### {outcome}
Quantitative rule: A requires ≥75% of representative layers to reach .999 cosine by 50% retention with ≤20-point retention spread; B requires ≥50%; C requires a nonzero minority; D requires none. Preliminary runs suppress these outcomes.

## Recommended next experiment
{recommendation}
"""; (results_dir/"report.md").write_text(report); return outcome
