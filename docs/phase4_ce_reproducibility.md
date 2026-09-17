# Phase 4 layer-0 d64 CE reproducibility audit

## Evidence available in this repository

The repository history is squashed: commit `558cdc8` introduces the complete
predictor implementation, and its parent contains no predictor code. Therefore
Git cannot establish settings from an earlier, stronger experimental run. The
checked-in implementation does establish the following original defaults:

- factorized predictor with dropout `0.1`;
- AdamW, learning rate `1e-3`, and weight decay `1e-4`;
- 100 epochs, batch size 16, patience 10, and seed 42;
- training-split mean and sample standard deviation, clamped at `1e-6`;
- `targets.pt["scores"]`, renormalized to a per-token distribution by
  `distribution_ce`;
- CUDA autocast for predictor forward when AMP is enabled;
- no scheduler, gradient accumulation, clipping, DataLoader, or STE path;
- checkpoint selection by validation captured importance mass.

These settings are pinned by `configs/phase4_d64_ce_baseline.json`. They reflect
code evidence, not a retrospective hyperparameter guess.

## Audit finding

No mathematical CE change is present in the available history. In particular,
plain CE training never executes the output-aware STE path. The strongest
repository-level reproducibility hazard is run-directory reuse: the legacy CLI
name resolves to `factorized_d64_distribution_ce`, and a `latest.pt` in that
directory resumes training rather than initializing a new model. At epoch 100,
the nominal "rerun" can execute no training epochs at all. Reproductions must
therefore use unique `--run-name` values.

The historical metric discrepancy cannot be attributed more narrowly without
the target data, original split/checkpoint, and unsquashed experiment history.
The new run metadata records split and normalization hashes so future results
can be compared rather than inferred.

## Reproduction commands

Use distinct directories for two from-scratch reproductions:

```bash
python scripts/train_predictor.py --results-dir results/predictor_2000 \
  --layer 0 --preset phase4_d64_ce_baseline --run-name factorized_d64_distribution_ce_repro_a --device cuda
python scripts/train_predictor.py --results-dir results/predictor_2000 \
  --layer 0 --preset phase4_d64_ce_baseline --run-name factorized_d64_distribution_ce_repro_b --device cuda
```

Evaluate each with exact hard Top-K:

```bash
python scripts/evaluate_predictor.py --results-dir results/predictor_2000 \
  --layer 0 --model factorized --latent-dim 64 \
  --run-name factorized_d64_distribution_ce_repro_a \
  --retention .3,.4,.5,.6,.75 --device cuda
```

After confirming the restored checkpoint, fine-tune without overwriting the
existing Phase 4.2 run:

```bash
python scripts/train_predictor.py --results-dir results/predictor_2000 \
  --layer 0 --model factorized --latent-dim 64 --loss output_hybrid_rank \
  --run-name factorized_d64_ce_repro_to_hybrid_rank_rw010 \
  --init-checkpoint results/predictor_2000/layer_000/factorized_d64_distribution_ce_repro_a/best.pt \
  --train-retention .5 --ranking-weight .10 --freeze-encoder-epochs 0 \
  --lr 5e-5 --epochs 50 --patience 8 --batch-size 16 --device cuda
```
