# Future work (conditional on Phase 1 evidence)

## Phase 2 — Predictor
Learn `hidden_state -> required neuron/block IDs` with a linear projection, low-rank linear predictor, tiny MLP, or XGBoost/block classifier. Measure oracle-neuron recall@K (initially more important than precision), sparse-output error, predictor FLOPs, parameters, and latency.

## Phase 3 — Hardware-aware sparse inference
Study GPU hot weights, a CPU cold compressed store, predictive block prefetch, and structured sparse kernels.

## Phase 4 — Mathematical compression
Investigate SVD, low-rank plus residual, vector quantisation, additive codebooks, 2/3-bit quantisation, and sparse residual outliers.

## Phase 5 — Evolutionary optimisation
Use GA/NEAT-style search to choose per-layer retention, block size, quantisation precision, predictor size, static GPU hot set, dynamic cache size, and factorisation rank. Fitness should combine quality, VRAM, bandwidth, latency, and PCIe traffic.

