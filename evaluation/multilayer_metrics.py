"""Trustworthy aggregation and validation for multi-layer oracle CSVs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

RETENTION_DECIMALS = 6
REQUIRED_RAW_COLUMNS = {
    "sample_id", "layer", "retention", "cosine_similarity", "relative_l2",
    "mse", "max_absolute_error", "dense_ffn_parameters",
}


def adjacent_pair_indices(prompt_ids, token_positions) -> list[tuple[int, int]]:
    """Return consecutive positions only within the same prompt/sequence."""
    previous: dict[int, tuple[int, int]] = {}
    pairs = []
    for index, (prompt, position) in enumerate(zip(prompt_ids, token_positions)):
        prompt, position = int(prompt), int(position)
        if prompt in previous and previous[prompt][1] + 1 == position:
            pairs.append((previous[prompt][0], index))
        previous[prompt] = (index, position)
    return pairs


def canonical_retention(value: float) -> float:
    """Return the sole numeric representation used for grouping and lookup."""
    return round(float(value), RETENTION_DECIMALS)


def normalise_retention(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = pd.to_numeric(result["retention"], errors="raise")
    result["retention"] = numeric.map(canonical_retention).astype(float)
    return result


def retention_rows(frame: pd.DataFrame, retention: float) -> pd.DataFrame:
    """Tolerance-safe lookup after canonicalisation; never use direct float equality."""
    target = canonical_retention(retention)
    values = pd.to_numeric(frame["retention"], errors="raise").map(canonical_retention)
    return frame.loc[values == target]


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def aggregate_oracle(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate finite raw observations and count unique samples, not CSV rows."""
    rows = normalise_retention(rows)
    result: list[dict] = []
    for keys, data in rows.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys))
        record["samples"] = int(data["sample_id"].nunique())
        record["rows"] = int(len(data))
        for metric in ("cosine_similarity", "relative_l2", "mse", "max_absolute_error"):
            numeric = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = numeric[np.isfinite(numeric)]
            record[f"{metric}_valid_samples"] = int(len(finite))
            record[f"{metric}_dropped_samples"] = int(len(numeric) - len(finite))
            if not len(finite):
                for name in ("mean", "median", "std", "p01", "p05", "p10", "p90", "p95", "p99", "min", "max"):
                    record[f"{metric}_{name}"] = np.nan
                continue
            record.update({
                f"{metric}_mean": float(finite.mean()),
                f"{metric}_median": float(np.median(finite)),
                f"{metric}_std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                f"{metric}_p01": _quantile(finite, .01),
                f"{metric}_p05": _quantile(finite, .05),
                f"{metric}_p10": _quantile(finite, .10),
                f"{metric}_p90": _quantile(finite, .90),
                f"{metric}_p95": _quantile(finite, .95),
                f"{metric}_p99": _quantile(finite, .99),
                f"{metric}_min": float(finite.min()),
                f"{metric}_max": float(finite.max()),
            })
        result.append(record)
    return pd.DataFrame(result).sort_values(group_columns).reset_index(drop=True)


def minimum_retention(summary: pd.DataFrame, column: str, target: float,
                      greater: bool = True) -> tuple[float, str]:
    data = normalise_retention(summary).sort_values("retention")
    numeric = pd.to_numeric(data[column], errors="coerce")
    valid = data.loc[numeric >= target] if greater else data.loc[numeric <= target]
    valid = valid.loc[np.isfinite(pd.to_numeric(valid[column], errors="coerce"))]
    if valid.empty:
        return np.nan, "not_met"
    return canonical_retention(valid.iloc[0]["retention"]), "met"


THRESHOLDS = (
    ("mean_cosine_0.99", "cosine_similarity_mean", .99, True),
    ("mean_cosine_0.995", "cosine_similarity_mean", .995, True),
    ("mean_cosine_0.999", "cosine_similarity_mean", .999, True),
    ("mean_cosine_0.9995", "cosine_similarity_mean", .9995, True),
    ("p05_cosine_0.99", "cosine_similarity_p05", .99, True),
    ("p01_cosine_0.99", "cosine_similarity_p01", .99, True),
    ("p05_cosine_0.995", "cosine_similarity_p05", .995, True),
    ("mean_l2_0.10", "relative_l2_mean", .10, False),
    ("mean_l2_0.05", "relative_l2_mean", .05, False),
    ("mean_l2_0.02", "relative_l2_mean", .02, False),
    ("mean_l2_0.01", "relative_l2_mean", .01, False),
    ("p95_l2_0.10", "relative_l2_p95", .10, False),
    ("p95_l2_0.05", "relative_l2_p95", .05, False),
    ("p99_l2_0.10", "relative_l2_p99", .10, False),
)


def threshold_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, data in normalise_retention(summary).groupby("layer", sort=True):
        row = {"layer": int(layer)}
        for name, column, target, greater in THRESHOLDS:
            value, status = minimum_retention(data, column, target, greater)
            row[f"retention_for_{name}"] = value
            row[f"status_{name}"] = status
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_intervals(rows: pd.DataFrame, repeats: int = 500, seed: int = 42,
                        retentions: Iterable[float] = (.3, .4, .5)) -> pd.DataFrame:
    rows = normalise_retention(rows)
    requested = {canonical_retention(value) for value in retentions}
    rng = np.random.default_rng(seed)
    result = []
    for layer in sorted(rows.layer.unique()):
        layer_rows = rows[rows.layer == layer]
        for retention in sorted(requested & set(layer_rows.retention.unique())):
            data = retention_rows(layer_rows, retention)
            for metric in ("cosine_similarity", "relative_l2"):
                values = pd.to_numeric(data[metric], errors="coerce").to_numpy(float)
                values = values[np.isfinite(values)]
                record = {"layer": int(layer), "retention": retention, "metric": metric,
                          "sample_count": int(len(values)), "bootstrap_resamples": repeats, "seed": seed}
                if len(values) < 2:
                    record.update({"observed_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                   "status": "insufficient_samples"})
                else:
                    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
                    record.update({"observed_mean": float(values.mean()), "ci_low": _quantile(draws, .025),
                                   "ci_high": _quantile(draws, .975), "status": "ok"})
                result.append(record)
    return pd.DataFrame(result)


def experiment_status(sample_counts: dict[int, int], minimum: int) -> tuple[str, list[int]]:
    below = sorted(layer for layer, count in sample_counts.items() if count < minimum)
    if not below: return "VERIFIED", []
    if len(below) == len(sample_counts): return "PRELIMINARY", below
    return "PARTIALLY VERIFIED", below


def validate_raw_and_summary(raw: pd.DataFrame, summary: pd.DataFrame,
                             monotonic_tolerance: float = 1e-5) -> dict:
    missing = REQUIRED_RAW_COLUMNS - set(raw.columns)
    if missing: raise ValueError(f"Raw oracle CSVs are missing required columns: {sorted(missing)}")
    raw = normalise_retention(raw)
    if raw[["layer", "retention", "sample_id"]].isna().any().any(): raise ValueError("Raw layer, retention, or sample IDs contain nulls")
    if (~raw.retention.between(0, 1)).any(): raise ValueError("Raw retention values must lie in (0, 1]")
    for metric in ("cosine_similarity", "relative_l2"):
        if not np.isfinite(pd.to_numeric(raw[metric], errors="coerce")).all(): raise ValueError(f"Raw {metric} contains non-finite values")
    summary = normalise_retention(summary)
    warnings = []; cross_checks = 0
    for (layer, retention), data in raw.groupby(["layer", "retention"], sort=True):
        aggregate = summary[(summary.layer == layer) & (summary.retention == retention)]
        if len(aggregate) != 1: raise ValueError(f"Expected one summary row for layer={layer}, retention={retention}; found {len(aggregate)}")
        for metric in ("cosine_similarity", "relative_l2", "mse", "max_absolute_error"):
            direct = float(data[metric].to_numpy(float).mean()); reported = float(aggregate.iloc[0][f"{metric}_mean"])
            if not np.isfinite(reported) or not np.isclose(direct, reported, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Raw→summary mismatch at layer={layer}, retention={retention}, metric={metric}: {direct} vs {reported}")
            cross_checks += 1
    full = retention_rows(summary, 1.0)
    full_pass = bool(len(full) and (full.cosine_similarity_mean >= .9999).all() and (full.relative_l2_mean <= 1e-3).all())
    if not full_pass: warnings.append("HIGH SEVERITY: 100% retention does not reproduce dense FFN output")
    for layer, data in summary.groupby("layer"):
        data = data.sort_values("retention")
        if np.any(np.diff(data.cosine_similarity_mean) < -monotonic_tolerance): warnings.append(f"Layer {layer}: cosine decreases materially with retention")
        if np.any(np.diff(data.relative_l2_mean) > monotonic_tolerance): warnings.append(f"Layer {layer}: relative L2 increases materially with retention")
    counts = raw.groupby(["layer", "retention"]).sample_id.nunique().unstack()
    consistent = bool((counts.nunique(axis=1) == 1).all())
    if not consistent: warnings.append("Sample counts differ across retention values within at least one layer")
    return {"raw_to_summary_cross_check": "PASS", "cross_checked_values": cross_checks,
            "full_retention_validation": "PASS" if full_pass else "FAIL",
            "sample_count_consistency": "PASS" if consistent else "WARNING", "warnings": warnings}


def position_bucket(position: int) -> str:
    if position <= 32: return "0-32"
    if position <= 128: return "33-128"
    if position <= 512: return "129-512"
    return "513+"


def representative_schedules(summary: pd.DataFrame, total_layers: int) -> pd.DataFrame:
    """Parameter-weighted schedules selected only from measured retentions."""
    summary = normalise_retention(summary)
    dense = summary.groupby("layer").dense_ffn_parameters.first().astype(float)
    rows = []
    for label, target in (("conservative", .999), ("moderate", .995), ("aggressive", .990)):
        schedule = {}; selected_tested = 0.0; dense_tested = float(dense.sum()); all_met = True
        for layer, data in summary.groupby("layer"):
            retention, status = minimum_retention(data, "cosine_similarity_mean", target, True)
            schedule[str(int(layer))] = None if status == "not_met" else retention
            if status == "not_met": all_met = False
            else:
                measured = retention_rows(data, retention)
                selected_tested += float(measured.iloc[0]["selected_ffn_parameters"])
        active = selected_tested / dense_tested if all_met else np.nan
        estimated_dense = float(dense.mean() * total_layers); estimated_selected = estimated_dense * active
        record = {"schedule": label, "quality_target": target, "status": "met_all_tested_layers" if all_met else "not_met",
                  "scope": "MEASURED REPRESENTATIVE-LAYER ESTIMATE", "per_layer_retention_schedule": json.dumps(schedule, sort_keys=True),
                  "mean_active_ffn_fraction": active, "mean_skipped_ffn_fraction": 1-active,
                  "estimated_dense_ffn_parameters": estimated_dense, "estimated_selected_ffn_parameters": estimated_selected,
                  "estimated_saved_ffn_parameters": estimated_dense-estimated_selected}
        for name, bits in {"bf16":16, "fp16":16, "8bit":8, "q4_raw":4, "3bit":3, "2bit":2}.items():
            record[f"selected_{name}_bytes"] = estimated_selected * bits / 8
        rows.append(record)
    return pd.DataFrame(rows)
