"""Validated multi-layer summaries, plots, schedules, and scientific report."""
from __future__ import annotations

import json
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from evaluation.multilayer_metrics import (
    aggregate_oracle, bootstrap_intervals, canonical_retention,
    experiment_status, normalise_retention, position_bucket,
    representative_schedules, retention_rows, threshold_table,
    validate_raw_and_summary,
)

PHASE1_APPROXIMATE = {
    .3: {"cosine": .9965, "relative_l2": np.nan},
    .4: {"cosine": .9985, "relative_l2": .05},
    .5: {"cosine": .99935, "relative_l2": .035},
}
RAW_LAYER_PATTERN = re.compile(r"^layer_(\d+)\.csv$")


def _artifact(results_dir: Path, stem: str, suffix: str, extension: str = ".csv") -> Path:
    return results_dir / f"{stem}{suffix}{extension}"


def _raw_oracle_files(oracle_dir: Path) -> list[Path]:
    # Root cause of the previous NaNs: layer_*.csv also selected
    # layer_XXX.stability.csv. NumPy aggregation then propagated those files'
    # absent metric columns as NaN specifically at 30/40/50% retention.
    return sorted(path for path in oracle_dir.glob("layer_*.csv") if RAW_LAYER_PATTERN.fullmatch(path.name))


def _lookup(summary: pd.DataFrame, layer: int, retention: float, column: str) -> float:
    rows = retention_rows(summary[summary.layer == layer], retention)
    if rows.empty: return np.nan
    value = pd.to_numeric(rows[column], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def _display(value, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}" if value is not None and np.isfinite(value) else "N/A — no measured value"


def _markdown(frame: pd.DataFrame) -> str:
    """Render a compact table without Pandas' optional ``tabulate`` dependency."""
    if frame.empty: return "N/A — no rows available"
    def cell(value):
        if pd.isna(value): return "N/A — unavailable"
        if isinstance(value, float): return f"{value:.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers=[str(column) for column in frame.columns]
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]
    lines.extend("| "+" | ".join(cell(value) for value in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def _phase1_comparison(results_dir: Path, layer_summary: pd.DataFrame) -> pd.DataFrame:
    candidates = [results_dir.parent / "layer0" / "summary.csv", results_dir.parent / "layer0" / "oracle.csv"]
    source = next((path for path in candidates if path.is_file()), None)
    phase = None
    if source is not None:
        loaded = pd.read_csv(source)
        if "method" in loaded: loaded = loaded[loaded.method == "oracle"]
        if "retention" in loaded and "cosine_similarity" in loaded:
            loaded = normalise_retention(loaded)
            phase = loaded.groupby("retention",as_index=False).agg(
                cosine_similarity_mean=("cosine_similarity","mean"),
                relative_l2_mean=("relative_l2","mean"),
            ).assign(layer=0)
        elif "retention" in loaded and "cosine_similarity_mean" in loaded:
            phase = normalise_retention(loaded)
            if "layer" not in phase:
                phase["layer"] = 0
    records = []
    for retention, fallback in PHASE1_APPROXIMATE.items():
        new_cos = _lookup(layer_summary, 0, retention, "cosine_similarity_mean")
        new_l2 = _lookup(layer_summary, 0, retention, "relative_l2_mean")
        old_cos = _lookup(phase, 0, retention, "cosine_similarity_mean") if phase is not None else fallback["cosine"]
        old_l2 = _lookup(phase, 0, retention, "relative_l2_mean") if phase is not None and "relative_l2_mean" in phase else fallback["relative_l2"]
        records.append({"retention": canonical_retention(retention), "phase1_comparison_available": source is not None,
          "phase1_source": str(source) if source else "approximate values supplied in experiment brief",
          "phase1_mean_cosine": old_cos, "multilayer_mean_cosine": new_cos,
          "cosine_absolute_difference": new_cos-old_cos, "cosine_relative_difference": (new_cos-old_cos)/old_cos,
          "phase1_relative_l2": old_l2, "multilayer_relative_l2": new_l2,
          "relative_l2_difference": new_l2-old_l2 if np.isfinite(old_l2) else np.nan})
    return pd.DataFrame(records)


def _main_table(summary: pd.DataFrame, thresholds: pd.DataFrame, metadata: dict,
                sample_counts: dict[int, int]) -> pd.DataFrame:
    mixers = {int(key): value for key, value in metadata.get("mixers", {}).items()}
    threshold_index = thresholds.set_index("layer")
    rows = []
    for layer in sorted(summary.layer.unique()):
        record = {"Layer": int(layer), "Mixer": mixers.get(int(layer), "unknown"),
                  "Samples": sample_counts[int(layer)]}
        for retention in (.2, .3, .4, .5, .6, .75):
            record[f"cos@{retention:g}"] = _lookup(summary, layer, retention, "cosine_similarity_mean")
        record["L2@0.4"] = _lookup(summary, layer, .4, "relative_l2_mean")
        record["L2@0.5"] = _lookup(summary, layer, .5, "relative_l2_mean")
        record["keep>=.999"] = threshold_index.loc[layer, "retention_for_mean_cosine_0.999"]
        rows.append(record)
    return pd.DataFrame(rows)


def _plot_lines(summary: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    for retention in (.2, .3, .4, .5):
        data = retention_rows(summary, retention).sort_values("layer")
        if len(data): plt.plot(data.layer, data[metric], "o-", label=f"{retention:.0%}")
    plt.xlabel("Layer"); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(output,dpi=170); plt.close()


def analyse_multilayer(results_dir: Path, min_report_samples: int = 1000,
                       bootstrap_resamples: int = 500, seed: int = 42,
                       output_suffix: str = "", monotonic_tolerance: float = 1e-5) -> str:
    metadata_path = results_dir / "metadata.json"
    if not metadata_path.is_file(): raise FileNotFoundError(f"Multi-layer capture is incomplete: missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text()); oracle_dir = results_dir / "oracle"
    files = _raw_oracle_files(oracle_dir)
    if not files: raise FileNotFoundError(f"No raw layer_NNN.csv oracle results found in {oracle_dir}")
    raw = normalise_retention(pd.concat([pd.read_csv(path) for path in files], ignore_index=True))
    samples_path = results_dir / "samples.jsonl"
    if samples_path.is_file():
        samples = pd.read_json(samples_path, lines=True)
        metadata_columns = [column for column in ("sample_id","prompt_id","token_position","token_id") if column in raw and column in samples]
        raw = raw.merge(samples, on=metadata_columns, how="left", validate="many_to_one", suffixes=("", "_sample"))
    summary = aggregate_oracle(raw, ["layer", "retention"])
    constants = raw.groupby(["layer","retention"],as_index=False)[["dense_ffn_parameters","selected_ffn_parameters","weight_traffic_fraction"]].first()
    summary = summary.merge(constants,on=["layer","retention"],validate="one_to_one")
    integrity = validate_raw_and_summary(raw, summary, monotonic_tolerance)
    for message in integrity["warnings"]: warnings.warn(message, RuntimeWarning)
    sample_counts = {int(layer): int(count) for layer,count in raw.groupby("layer").sample_id.nunique().items()}
    expected_layers=[int(layer) for layer in metadata.get("layers",sample_counts)]
    for layer in expected_layers: sample_counts.setdefault(layer,0)
    unexpected=sorted(set(sample_counts)-set(expected_layers));missing_layers=sorted(set(expected_layers)-set(raw.layer.astype(int).unique()))
    if unexpected: raise ValueError(f"Raw oracle CSV contains layers absent from capture metadata: {unexpected}")
    if missing_layers: integrity["warnings"].append(f"No raw oracle rows for expected layers: {missing_layers}")
    config_path=results_dir/"config.yaml";config=yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    requested = metadata.get("requested_max_tokens_per_layer", config.get("max_tokens_per_layer",metadata.get("samples_per_layer")))
    count_rows=[]
    for (layer,retention),data in raw.groupby(["layer","retention"]):
        valid=np.isfinite(data.cosine_similarity)&np.isfinite(data.relative_l2)
        count_rows.append({"layer":int(layer),"retention":retention,"actual_unique_samples":int(data.sample_id.nunique()),
          "valid_samples":int(data.loc[valid,"sample_id"].nunique()),"dropped_invalid_samples":int((~valid).sum()),"requested_max_tokens_per_layer":requested})
    counts_frame=pd.DataFrame(count_rows); counts_frame.to_csv(_artifact(results_dir,"sample_counts",output_suffix),index=False)
    consistency = counts_frame.groupby("layer").valid_samples.nunique(); integrity["sample_count_consistency"]="PASS" if (consistency==1).all() else "WARNING"
    thresholds = threshold_table(summary); bootstrap = bootstrap_intervals(raw,bootstrap_resamples,seed,sorted(raw.retention.unique()))
    schedules = representative_schedules(summary, int(metadata["layer_count"])); phase1 = _phase1_comparison(results_dir,summary)
    summary.to_csv(_artifact(results_dir,"layer_summary",output_suffix),index=False); thresholds.to_csv(_artifact(results_dir,"threshold_summary",output_suffix),index=False)
    bootstrap.to_csv(_artifact(results_dir,"bootstrap_ci",output_suffix),index=False); schedules.to_csv(_artifact(results_dir,"model_wide_estimate",output_suffix),index=False); phase1.to_csv(_artifact(results_dir,"phase1_comparison",output_suffix),index=False)
    split_summary=aggregate_oracle(raw,["split","layer","retention"]) if "split" in raw else pd.DataFrame(); split_summary.to_csv(_artifact(results_dir,"split_summary",output_suffix),index=False)
    category=aggregate_oracle(raw,["layer","category","retention"]) if "category" in raw else pd.DataFrame(); category.to_csv(_artifact(results_dir,"category_summary",output_suffix),index=False)
    if "token_position" in raw:
        raw["position_bucket"]=raw.token_position.map(position_bucket); aggregate_oracle(raw,["layer","position_bucket","retention"]).to_csv(_artifact(results_dir,"token_position_summary",output_suffix),index=False)
    status,below=experiment_status(sample_counts,min_report_samples); main=_main_table(summary,thresholds,metadata,sample_counts)
    # Finite-value invariant: a displayed metric may be absent only when raw data is absent.
    for layer in main.Layer:
      for retention in (.2,.3,.4,.5,.6,.75):
        if len(retention_rows(raw[raw.layer==layer],retention)) and not np.isfinite(main.loc[main.Layer==layer,f"cos@{retention:g}"]).all():
            raise ValueError(f"Valid raw data became non-finite in report table: layer={layer}, retention={retention}")
    plot_dir=results_dir/f"plots{output_suffix}";plot_dir.mkdir(exist_ok=True)
    _plot_lines(summary,"cosine_similarity_mean","Mean cosine",plot_dir/"depth_cosine.png");_plot_lines(summary,"relative_l2_mean","Mean relative L2",plot_dir/"depth_l2.png")
    pivot=summary.pivot(index="layer",columns="retention",values="cosine_similarity_mean");plt.imshow(pivot,aspect="auto",vmin=min(.99,float(np.nanmin(pivot))),vmax=1);plt.xticks(range(len(pivot.columns)),pivot.columns);plt.yticks(range(len(pivot.index)),pivot.index);plt.colorbar(label="Mean cosine");plt.tight_layout();plt.savefig(plot_dir/"retention_heatmap.png",dpi=170);plt.close()
    tail=summary.pivot(index="layer",columns="retention",values="cosine_similarity_p01");plt.imshow(tail,aspect="auto");plt.xticks(range(len(tail.columns)),tail.columns);plt.yticks(range(len(tail.index)),tail.index);plt.colorbar(label="P01 cosine (bad tail)");plt.tight_layout();plt.savefig(plot_dir/"tail_risk_heatmap.png",dpi=170);plt.close()
    for column,filename,ylabel in (("retention_for_mean_cosine_0.999","required_retention_999.png","Retention for mean cosine ≥ .999"),("retention_for_mean_l2_0.05","required_retention_l2_5.png","Retention for mean L2 ≤ 5%")):
        plt.plot(thresholds.layer,thresholds[column],"o-");plt.xlabel("Layer");plt.ylabel(ylabel);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(plot_dir/filename,dpi=170);plt.close()
    if len(category):
        for name,data in retention_rows(category,.4).groupby("category"): plt.plot(data.layer,data.cosine_similarity_mean,"o-",label=name)
        plt.xlabel("Layer");plt.ylabel("40% mean cosine");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(plot_dir/"category_comparison.png",dpi=170);plt.close()
    required=thresholds["retention_for_mean_cosine_0.999"]; qualifying=int((required<=.5).sum()); fraction=qualifying/len(required); spread=float(required.max()-required.min()) if required.notna().all() else np.inf
    layer0_40=_lookup(summary,0,.4,"cosine_similarity_mean"); phase1_40=float(PHASE1_APPROXIMATE[.4]["cosine"])
    if fraction>=.75 and spread<=.2: outcome="OUTCOME A — MODEL-WIDE CONTEXTUAL SPARSITY STRONG"; recommendation="Train a lightweight individual-neuron predictor."
    elif fraction>=.5: outcome="OUTCOME B — LAYER-DEPENDENT CONTEXTUAL SPARSITY"; recommendation="Use per-layer retention budgets and predictors only where worthwhile."
    elif qualifying>0: outcome="OUTCOME C — LIMITED SPARSITY"; recommendation="Restrict contextual sparsity to qualifying layers and study compression elsewhere."
    else: outcome="OUTCOME D — DOES NOT GENERALISE"; recommendation="Stop predictor work and pivot to mathematical weight compression."
    decision=f"PRELIMINARY {outcome}" if status!="VERIFIED" else outcome
    tail_table=summary[["layer","retention","cosine_similarity_p01","cosine_similarity_p05","cosine_similarity_min","relative_l2_p95","relative_l2_p99","relative_l2_max"]]
    integrity_path=_artifact(results_dir,"analysis_integrity",output_suffix,".json");integrity_path.write_text(json.dumps(integrity|{"status":status,"layers_below_minimum":below},indent=2)+"\n")
    report=f"""# Multi-Layer Oracle Validation

## Status
**{status}**. Actual unique samples/layer: `{sample_counts}`. Requested maximum: `{requested}`. Layers below {min_report_samples}: `{below}`.

## Experimental Setup
Model `{metadata.get('model')}`; layers `{sorted(sample_counts)}`; mixers `{metadata.get('mixers')}`; dtype `{metadata.get('activation_dtype')}`; seed `{metadata.get('seed')}`.

## Data Integrity Checks
- 100% retention validation: **{integrity['full_retention_validation']}**
- Raw→summary cross-check: **{integrity['raw_to_summary_cross_check']}** ({integrity['cross_checked_values']} values)
- Sample-count consistency: **{integrity['sample_count_consistency']}**
- Warnings: `{integrity['warnings'] or 'none'}`

## Layer Results
{_markdown(main)}

## Depth Behaviour
See `layer_summary{output_suffix}.csv`, `split_summary{output_suffix}.csv`, and depth plots. No untested layers are interpolated.

## Quality Thresholds
{_markdown(thresholds)}

## Tail-Risk Analysis
Cosine risk uses the **lower** p01/p05/minimum tail; error risk uses the **upper** p95/p99/maximum tail.
{_markdown(tail_table)}

## Phase 1 Reproduction
{_markdown(phase1)}

## Representative Model-Wide Estimates
{_markdown(schedules)}
Raw payloads exclude quantisation metadata, KV cache, runtime buffers, predictors, and attention/DeltaNet weights. These are FFN-only representative-layer estimates, not actual VRAM.

## Bootstrap Confidence Intervals
{_markdown(bootstrap)}

## Caveats
The oracle consumes full FFN activations; no runtime speedup exists. Irregular access may be inefficient. FFN output accuracy does not guarantee generation quality, and layer errors may compound.

## Decision
### {decision}
Rule: A requires ≥75% of tested layers to reach mean cosine .999 by 50% retention with ≤20-point spread; B requires ≥50%; C a nonzero minority; D none. Status prefixes do not suppress measured metrics. Layer-0 40% difference from the supplied Phase-1 reference: {_display(layer0_40-phase1_40)}.

## Recommended Next Experiment
{recommendation}
"""
    _artifact(results_dir,"report",output_suffix,".md").write_text(report)
    return decision
