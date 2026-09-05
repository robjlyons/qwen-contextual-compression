# Qwen contextual compression: oracle sparse FFNs

Dense LLM inference normally evaluates nearly every dense FFN weight for every token. This repository tests a deliberately falsifiable hypothesis: **only an input-dependent subset of intermediate neurons may materially contribute to one token's FFN output**. If that subset can eventually be predicted cheaply *before* FFN evaluation, a future system might avoid multiplications and weight loads, keep hot/common weights plus a dynamic cache on GPU, and hold cold compressed weights in system RAM.

This Phase-1 code tests only the mathematical **oracle upper bound** on a real `Qwen/Qwen3.8-27B` layer and genuine model hidden states. It does not save runtime: oracle ranking first evaluates every gate/up activation. It includes neuron and block oracles, random/static-hot/static-weight baselines, weight accounting, hotness statistics, plots, and a report. It does not implement predictors, kernels, model compression, GGUF, or llama.cpp changes.

## Architecture safety

The loader does not hard-code `model.layers[0].mlp`. It walks candidate language-layer `ModuleList`s and requires a module with `gate_proj`, `up_proj`, `down_proj`, and a callable activation. It reports the discovered paths and real shapes, or fails with an actionable diagnostic. The explicit reference computes:

```text
a = act_fn(gate_proj(x)) * up_proj(x)
y = down_proj(a)
```

For selected neuron `j`, a perfect pre-FFN selector needs row `j` of both input projections and column `j` of the down projection. Thus exact accounting is `K*(gate_in + up_in + down_out)` weights, plus any biases. With conventional equal-width gate/up matrices this is exactly proportional to `K/N` for projection weights; the implementation validates dimensions rather than assuming that result.

## Selective checkpoint loading (default)

The initial experiment does **not** download the complete 27B checkpoint. The loader downloads `model.safetensors.index.json`, discovers the language embedding tensor and every tensor under the selected `layers.N` prefix, and asks `huggingface_hub` for only the unique shards named by those entries. Shard numbers are never hard-coded. For the currently published layer-0 index this is expected to resolve the embedding and layer-0 shards, but the downloaded filenames printed by inspection are the source of truth.

Transformers builds the official causal-LM architecture on the `meta` device, after which only selected tensors are materialized. The loader calls the official selected MLP and compares it with the independent explicit gated-FFN equation before returning. During partial activation capture, the official model executes embeddings, layer-0 attention/DeltaNet, normalization, and its MLP; a targeted post-hook then stops execution before an unloaded later layer is reached.

Checkpoint tensor names and instantiated module names are reconciled explicitly. In particular, a serialized key such as `model.language_model.layers.0...` can map to the text-only Transformers wrapper's `model.layers.0...`. Resolution uses the longest unique dotted suffix, validates tensor shapes before assignment, and fails rather than guessing when a match is missing or ambiguous. Inspection prints the complete mapping for auditability.

Pass `--full-model` to use normal `from_pretrained()` as an explicit fallback. Only that mode applies Accelerate `--device-map` and `--offload-folder`; selective mode places its small materialized subset on `--device` (or CUDA when available, otherwise CPU). `--cache-dir`, `--revision`, and `--trust-remote-code` are supported in both relevant paths.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
```

Use a recent Transformers release that recognizes the model architecture. Access to the model repository, sufficient host storage/RAM, and acceptance of any model terms may be required.

## First workflow

```bash
python scripts/inspect_model.py --model Qwen/Qwen3.8-27B --layer 0
python scripts/capture_activations.py --model Qwen/Qwen3.8-27B --layer 0 \
  --max-tokens 2000 --output-dir results/layer0
python scripts/run_oracle_sweep.py --model Qwen/Qwen3.8-27B --layer 0 \
  --activation-dir results/layer0 \
  --retention 1.0,0.75,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.05
python scripts/analyse_results.py --results-dir results/layer0
```

The first command prints `downloaded_shards`; use that field to audit that only
the dynamically selected files were fetched. To deliberately fetch and dispatch
the entire checkpoint instead, append `--full-model --device-map auto`.

The scripts accept Accelerate `--device-map` and `--offload-folder`, dtype and batch controls. Unknown arguments are ignored so Jupyter's injected `-f kernel.json` is harmless. In Colab, prefix commands with `!`. Activation inputs are written incrementally as bounded `.pt` shards; sweeps load and process one batch at a time. A TXT file supplies one prompt per line; JSONL accepts `text` or `prompt`. Special tokens are excluded by default.

Outputs include `oracle.csv`, `summary.csv`, `neuron_stats.csv`, `hot_coverage.csv`, configuration/activation metadata, six PNG plots, and `report.md`. `run_layer_comparison.py` summarizes already-completed representative layers without accidentally launching nine large experiments.

## Interpretation

`theoretical_weight_traffic_fraction` is the ideal projection-weight subset a perfect advance selector could request. `actual_oracle_executes_dense_gate_up=true` records the crucial fact that the current PyTorch oracle accesses dense gate/up weights and should never be presented as a benchmark or speedup. Output similarity is layer-local; corpus diversity, end-to-end perplexity/generation, predictor cost, and sparse-hardware efficiency remain future validation work. See [FUTURE_WORK.md](FUTURE_WORK.md).

## Phase 2: co-activation neuron layouts

The layout experiment reuses Phase-1 `inputs_*.pt` shards. It packs oracle masks
neuron-major (one bit per token), learns all orderings from the calibration prefix,
and evaluates reconstruction only on the held-out validation suffix. Its sparse
graph uses random-projection selection signatures to propose bounded candidate
neighbours, then computes exact Jaccard/cosine/conditional/raw/NPMI affinity only
for those candidates. It never allocates a dense 17,408-by-17,408 matrix.

```bash
python scripts/build_coactivation.py --activation-dir results/layer0 \
  --retention 0.20,0.30,0.40,0.50 --clustering-retention 0.40
python scripts/cluster_neurons.py --clustering-dir results/layer0/clustering
python scripts/run_permuted_block_sweep.py --activation-dir results/layer0 \
  --clustering-dir results/layer0/clustering \
  --retention 0.30,0.40,0.50 --block-sizes 16,32,64,128
python scripts/analyse_permutation.py --clustering-dir results/layer0/clustering
```

The sweep reports both equal-neuron-budget and equal-oracle-coverage modes. The
generated report includes tail metrics, paired bootstrap confidence intervals,
dense permutation validation, locality plots, and an automatically thresholded
Outcome A/B/C. Reordering itself never changes the dense mathematical function:
gate/up rows, their biases, and down columns are transformed consistently.

## Multi-layer oracle validation

Deeper hidden states must come from a mathematically valid pass through every
preceding hybrid layer. The reference capture therefore uses one normal official
Transformers forward with simultaneous MLP pre-hooks and requires the full model,
normally dispatched with Accelerate CPU/disk offload. It does not approximate or
independently reimplement attention, Gated DeltaNet state, masks, or positions.
After capture, the oracle command materializes only one selected FFN at a time.

The same stable `sample_id` identifies a prompt/token position in every layer.
One prompt is committed per shard and capture state is updated atomically, making
capture resumable. The default 2,000-token, nine-layer FP16/BF16 activation payload
is about 175.8 MiB; 10,000 tokens are about 878.9 MiB, before small metadata and
container overhead.

First 2,000-token experiment:

```bash
python scripts/capture_multilayer_activations.py \
  --model Qwen/Qwen3.8-27B \
  --layers 0,8,16,24,32,40,48,56,63 \
  --max-tokens-per-layer 2000 \
  --min-tokens-per-layer 1800 \
  --input /path/to/your/calibration.jsonl \
  --device-map auto --offload-folder offload/qwen38 \
  --output-dir results/multilayer

python scripts/run_multilayer_oracle.py \
  --model Qwen/Qwen3.8-27B \
  --activation-dir results/multilayer/activations \
  --layers 0,8,16,24,32,40,48,56,63 \
  --retention 0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0 \
  --output-dir results/multilayer/oracle

python scripts/analyse_multilayer_results.py \
  --results-dir results/multilayer --min-report-samples 1000
```

Supply an **existing** `--input corpus.jsonl` (fields `text`, optional `category` and `source`)
or `--dataset NAME --text-field text --category-field category`. The bundled
category-balanced prompts are a smoke test, not enough for a verified conclusion.
Corpus existence and content are validated before the full checkpoint is loaded,
so a path typo cannot trigger a 55 GB download before failing. Omit `--input` for
the built-in smoke test; it will correctly produce a `PRELIMINARY` report rather
than pretending to contain 2,000 tokens.
Use `--all-layers` after representative-layer validation. If the detected model
does not have 64 layers, the default representative indices are adjusted and
recorded rather than silently indexing invalid layers.

In Colab the same commands work with `!python`; point `--output-dir` at a mounted
Google Drive directory if `/content` persistence is insufficient. Unknown kernel
arguments are ignored. The one-command resumable wrapper is:

```bash
python scripts/run_multilayer_experiment.py --max-tokens-per-layer 2000 \
  --min-tokens-per-layer 1800 \
  --input /path/to/your/calibration.jsonl \
  --offload-folder offload/qwen38 --output-dir results/multilayer
```

Reanalyse existing raw multi-layer CSVs without loading Qwen or recapturing data:

```bash
python scripts/analyse_multilayer_results.py \
  --results-dir results/multilayer \
  --min-report-samples 1000 --bootstrap-resamples 500 --seed 42
```

Add `--output-suffix _corrected` to preserve an older report and write corrected
analysis artifacts and plots under suffixed names. Retentions are canonicalised
to six decimal places before every group, pivot, threshold, bootstrap, and report
lookup. Analysis accepts only raw `oracle/layer_NNN.csv` files; diagnostic
`layer_NNN.stability.csv` files are deliberately excluded.
