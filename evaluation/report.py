from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _minimum_keep(df, column, threshold, greater=True):
    valid=df[df[column]>=threshold] if greater else df[df[column]<=threshold]
    return "not achieved" if valid.empty else f"{100*valid.retention.min():.2f}%"


def analyse(results_dir: Path) -> None:
    summary=pd.read_csv(results_dir/"summary.csv"); stats=pd.read_csv(results_dir/"neuron_stats.csv")
    coverage=pd.read_csv(results_dir/"hot_coverage.csv"); plots=results_dir/"plots"; plots.mkdir(exist_ok=True)
    oracle=summary[summary.method=="oracle"].sort_values("retention")
    specs=[("cosine_similarity","Mean cosine similarity","cosine.png"),
           ("relative_l2","Mean relative L2 error","relative_l2.png"),
           ("theoretical_weight_traffic_fraction","Theoretical FFN weight traffic","weight_traffic.png")]
    for col,label,name in specs:
        for method,group in summary.groupby("method"): plt.plot(group.retention,group[col],marker="o",label=method)
        plt.xlabel("Retention ratio"); plt.ylabel(label); plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(plots/name,dpi=160); plt.close()
    plt.hist(stats.selection_frequency,bins=50); plt.xlabel("Top-30% selection frequency"); plt.ylabel("Neurons"); plt.tight_layout(); plt.savefig(plots/"frequency_histogram.png",dpi=160); plt.close()
    sorted_hot=np.sort(stats.selection_frequency)[::-1]; plt.plot(np.arange(1,len(sorted_hot)+1)/len(sorted_hot),sorted_hot); plt.xlabel("Fraction of neurons"); plt.ylabel("Selection frequency"); plt.tight_layout(); plt.savefig(plots/"sorted_hotness.png",dpi=160); plt.close()
    plt.plot(coverage.hot_fraction,coverage.selection_coverage); plt.xlabel("Globally-hot neuron fraction"); plt.ylabel("Oracle selection coverage"); plt.tight_layout(); plt.savefig(plots/"cumulative_coverage.png",dpi=160); plt.close()
    def cov(frac): return coverage.iloc[min(len(coverage)-1,max(0,int(np.ceil(frac*len(coverage)))-1))].selection_coverage
    dense=int(oracle.dense_ffn_parameters.iloc[0]); best=oracle[oracle.cosine_similarity>=.99]
    point=oracle.iloc[-1] if best.empty else best.loc[best.retention.idxmin()]
    report=f"""# Oracle sparse FFN report

## Dense FFN

- Intermediate neurons: {len(stats):,}
- Projection parameters (including any biases): {dense:,}
- Estimated BF16 size: {dense*2/2**20:.2f} MiB
- Estimated Q4 payload (packing overhead excluded): {dense*.5/2**20:.2f} MiB

## Oracle sparsity

- At 99.9% mean cosine similarity: minimum retention = {_minimum_keep(oracle,'cosine_similarity',.999)}
- At 99%: minimum retention = {_minimum_keep(oracle,'cosine_similarity',.99)}
- At 98%: minimum retention = {_minimum_keep(oracle,'cosine_similarity',.98)}
- At relative L2 < 1%: minimum retention = {_minimum_keep(oracle,'relative_l2',.01,False)}
- At relative L2 < 5%: minimum retention = {_minimum_keep(oracle,'relative_l2',.05,False)}

## Hot/cold structure

- Top 10% most frequently selected neurons cover {cov(.10):.2%} of top-30% selection events.
- Top 25% cover {cov(.25):.2%}.
- Top 30% cover {cov(.30):.2%}.

## Theoretical implications

At the lowest measured oracle retention with mean cosine >= .99 ({point.retention:.2%}), the perfect-selector subset is {int(point.theoretical_selected_parameters):,} parameters versus {dense:,}; its ideal packed-Q4 payload is {point.theoretical_selected_parameters*.5/2**20:.2f} MiB, a theoretical {dense/point.theoretical_selected_parameters:.2f}x reduction.

## Important caveats

- The oracle sees full FFN activations and this PyTorch implementation still executes dense gate/up projections; **no runtime speedup is claimed**.
- Results are layer-specific, and FFN-output preservation does not imply preserved perplexity or generation quality.
- The built-in prompts are only a smoke test; final validation needs a representative corpus.
- A future predictor's parameters, FLOPs, latency, and errors must be included.
- Sparse access can reduce GPU efficiency even when mathematical weight traffic falls.
- Gated SwiGLU sparsity can differ from ReLU-family sparsity.
"""
    (results_dir/"report.md").write_text(report)

