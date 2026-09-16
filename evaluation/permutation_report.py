"""Plots and evidence-based decision report for the neuron-layout experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clustering.similarity import SparseGraph
from evaluation.permutation_metrics import paired_bootstrap


def _line_plot(data, x, y, filename, xlabel, ylabel):
    for name,group in data.groupby("ordering"):
        group=group.sort_values(x); plt.plot(group[x],group[y],marker="o",label=name)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(filename,dpi=170); plt.close()


def _near(frame: pd.DataFrame, retention=.4, block=32, mode="equal_budget") -> pd.DataFrame:
    return frame[(frame["mode"]==mode)&(frame.block_size==block)&np.isclose(frame.retention,retention)]


def analyse_permutation(directory: Path, bootstrap_repeats: int=2000, seed: int=42) -> str:
    required = [
        directory / "block_sweep.csv",
        directory / "validation_summary.csv",
        directory / "locality_metrics.csv",
        directory / "neuron_clusters.csv",
        directory / "coactivation_graph.npz",
        directory / "orderings.npz",
        directory / "dense_permutation_validation.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot analyse an incomplete permutation experiment. Run "
            "build_coactivation.py, cluster_neurons.py, and "
            "run_permuted_block_sweep.py successfully first. Missing: "
            + ", ".join(missing)
        )
    raw=pd.read_csv(directory/"block_sweep.csv"); summary=pd.read_csv(directory/"validation_summary.csv")
    locality=pd.read_csv(directory/"locality_metrics.csv"); neurons=pd.read_csv(directory/"neuron_clusters.csv")
    graph=SparseGraph.load(directory/"coactivation_graph.npz"); orderings=np.load(directory/"orderings.npz")
    plots=directory/"plots"; plots.mkdir(exist_ok=True)
    budget=summary[summary["mode"]=="equal_budget"]
    budget40=budget[np.isclose(budget.retention,.4)]
    coverage40=summary[(summary["mode"]=="equal_coverage")&np.isclose(summary.retention,.4)]
    _line_plot(budget40,"block_size","cosine_similarity_mean",plots/"block_size_cosine.png","Block size","Mean cosine at 40% budget")
    _line_plot(budget40,"block_size","relative_l2_mean",plots/"block_size_relative_l2.png","Block size","Mean relative L2 at 40% budget")
    _line_plot(coverage40,"block_size","block_expansion_factor_mean",plots/"block_expansion.png","Block size","Expansion for equal oracle coverage")
    block=budget[budget.ordering=="clustered"]
    individual=budget[budget.ordering=="individual"]
    plt.plot(individual.retention,individual.cosine_similarity_mean,"o-",label="individual")
    best_block=int(block.groupby("block_size").cosine_similarity_mean.mean().idxmax()) if len(block) else 32
    chosen=block[block.block_size==best_block]; plt.plot(chosen.retention,chosen.cosine_similarity_mean,"o-",label=f"clustered block{best_block}"); plt.xlabel("Retention"); plt.ylabel("Mean cosine"); plt.legend(); plt.tight_layout(); plt.savefig(plots/"retention_cosine.png",dpi=170); plt.close()
    _line_plot(summary,"loaded_fraction_mean","oracle_recall_mean",plots/"recall_vs_loaded.png","Loaded neuron fraction","Oracle recall")
    for name,data in raw[(raw["mode"]=="equal_budget")&np.isclose(raw.retention,.4)&(raw.block_size==32)].groupby("ordering"):
        plt.hist(data.block_utilisation,bins=40,alpha=.35,label=name)
    plt.xlabel("Block utilisation"); plt.ylabel("Tokens"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(plots/"block_utilisation.png",dpi=170); plt.close()
    # Sparse affinity pictures avoid allocating the full 17k-square matrix.
    sample=min(512,graph.nodes); chosen_nodes=np.argsort(-graph.node_frequency)[:sample]
    for name,perm in (("original",orderings["original"]),("clustered",orderings["clustered"])):
        positions=np.empty(graph.nodes,int); positions[perm]=np.arange(graph.nodes); selected=set(chosen_nodes.tolist())
        matrix=np.zeros((sample,sample),np.float32); local={node:i for i,node in enumerate(sorted(chosen_nodes,key=lambda n:positions[n]))}
        for a,b,w in zip(graph.source,graph.target,graph.affinity):
            if int(a) in selected and int(b) in selected: matrix[local[int(a)],local[int(b)]]=matrix[local[int(b)],local[int(a)]]=w
        plt.imshow(matrix,aspect="auto",cmap="magma"); plt.colorbar(label="Affinity"); plt.title(name); plt.tight_layout(); plt.savefig(plots/f"affinity_{name}.png",dpi=170); plt.close()
    comparison=_near(raw)
    individual_rows=raw[(raw.ordering=="individual")&np.isclose(raw.retention,.4)]
    means=pd.concat([comparison,individual_rows]).groupby("ordering").cosine_similarity.mean()
    candidates=[x for x in ("clustered","greedy","signature","hybrid") if x in means]
    best=max(candidates,key=lambda x:means[x]) if candidates else "clustered"
    bootstrap=paired_bootstrap(raw,best,32,.4,bootstrap_repeats,seed); bootstrap.to_csv(directory/"bootstrap_results.csv",index=False)
    # Category labels were not recorded by legacy Phase-1 shards; retain an explicit, non-fabricated artifact.
    pd.DataFrame(columns=["category","samples","ordering","cosine_similarity","relative_l2"]).to_csv(directory/"category_summary.csv",index=False)
    edge_classes=[]
    temperature=neurons.set_index("neuron").temperature
    cluster=neurons.set_index("neuron").graph_cluster
    for label in ("HOT","WARM","COLD"):
        edge_mask=np.array([(temperature[int(a)]==label and temperature[int(b)]==label) for a,b in zip(graph.source,graph.target)])
        edge_classes.append((label,float(graph.affinity[edge_mask].mean()) if edge_mask.any() else np.nan,
                             float(np.mean([cluster[int(a)]==cluster[int(b)] for a,b in zip(graph.source[edge_mask],graph.target[edge_mask])])) if edge_mask.any() else np.nan))
    classes=neurons.groupby("temperature").agg(neurons=("neuron","count"),mean_frequency=("frequency","mean")); classes["selection_fraction"]=classes.neurons*classes.mean_frequency/(neurons.frequency.sum())
    for label,affinity,within in edge_classes:
        if label in classes.index: classes.loc[label,"mean_same_class_edge_affinity"]=affinity; classes.loc[label,"fraction_edges_within_graph_cluster"]=within
    clustered=float(means.get(best,np.nan)); original=float(means.get("original",np.nan)); individual=float(means.get("individual",np.nan))
    ci=bootstrap[bootstrap.metric=="cosine_similarity"]
    ci_low=float(ci.ci95_low.iloc[0]) if len(ci) else np.nan
    block64=budget[(budget.ordering==best)&(budget.block_size==64)&budget.retention.between(.4,.5)]
    viable64=bool(len(block64) and block64.cosine_similarity_mean.max()>=.99)
    viable=(clustered>=.995 and individual-clustered<=.005) or viable64
    partial=(clustered-original>=.002 and (np.isnan(ci_low) or ci_low>0))
    if viable: outcome="OUTCOME A — BLOCK SPARSITY VIABLE"; next_step="Train a block predictor."
    elif partial: outcome="OUTCOME B — PARTIAL LOCALITY"; next_step="Test a predictor with redundant block selection and/or permutation optimisation."
    else: outcome="OUTCOME C — BLOCK SPARSITY NOT VIABLE"; next_step="Skip fixed block prediction and investigate irregular sparse access or mathematical weight compression."
    validation=json.loads((directory/"dense_permutation_validation.json").read_text())
    table=pd.concat([_near(summary),summary[(summary.ordering=="individual")&np.isclose(summary.retention,.4)]],ignore_index=True).sort_values("cosine_similarity_mean",ascending=False)[["ordering","block_size","loaded_fraction_mean","cosine_similarity_mean","relative_l2_mean","oracle_recall_mean","block_precision_mean","block_expansion_factor_mean"]]
    locality_by_name=locality.set_index("ordering")
    original_distance=float(locality_by_name.loc["original","weighted_mean_distance"])
    best_distance=float(locality_by_name.loc[best,"weighted_mean_distance"])
    coverage_best=coverage40[coverage40.ordering==best]
    best_expansion=float(coverage_best.block_expansion_factor_mean.min()) if len(coverage_best) else np.nan
    report=f"""# Co-activation neuron permutation report

## Protocol
The ordering was learned only from the calibration prefix. All FFN reconstruction metrics below use the held-out validation suffix. Graph construction uses bounded LSH candidates followed by exact binary affinity, never a dense neuron-by-neuron matrix.

## Dense permutation validation
```json
{json.dumps(validation,indent=2)}
```
Sparse results are interpreted only after the automated dense-equivalence gate passes.

## Key comparison (40% target, block 32, equal neuron budget)
{table.to_markdown(index=False)}

## Locality
{locality.to_markdown(index=False)}

## Hot / warm / cold
{classes.reset_index().to_markdown(index=False)}

Legacy Phase-1 activation shards do not contain prompt-category IDs, so `category_summary.csv` is intentionally empty rather than assigning invented semantic labels.

## Answers to the research questions
1. Original-order locality has affinity-weighted mean distance {original_distance:.2f}; compare its neighbourhood columns above rather than assuming it is useful.
2. The best learned ordering (`{best}`) changes weighted mean distance to {best_distance:.2f}.
3. At 40%/block32, validation cosine changes from {original:.6f} (original) to {clustered:.6f} (`{best}`), versus {individual:.6f} for individual selection.
4. The measured quality/hardware options are listed in the key table; the highest average clustered quality occurred at block size {best_block}.
5. At 99% oracle-coverage mode, the best observed `{best}` expansion is {best_expansion:.3f}x relative to ideal neuron loading.
6. All reconstruction and bootstrap values are held-out, so they directly measure generalisation beyond calibration tokens.
7. Predictor work is justified only by the automatic Outcome A/B/C criteria below; no positive conclusion is assumed.

## Significance
{bootstrap.to_markdown(index=False) if len(bootstrap) else 'Insufficient paired validation rows for bootstrap confidence intervals.'}

## Automatic decision thresholds
- Outcome A: clustered mean cosine >= 0.995 with gap to individual <= 0.005 at 40%/block32, or block64 cosine >= 0.99 at 40–50% loaded.
- Outcome B: mean cosine improvement over original >= 0.002 and the paired 95% bootstrap CI excludes zero when available.
- Outcome C: neither criterion is met.

## {outcome}
Recommended next step: {next_step}
"""
    (directory/"report.md").write_text(report); return outcome
