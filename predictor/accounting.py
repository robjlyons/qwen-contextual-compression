"""Theoretical predictor compute and cold-weight-traffic accounting."""
from predictor.models import FactorizedPredictor, ResidualFactorizedPredictor


def predictor_macs(model, input_dim, output_dim):
    if isinstance(model, ResidualFactorizedPredictor):
        latent = model.encoder.out_features
        return input_dim * latent + 2 * latent * latent + latent * output_dim
    if isinstance(model, FactorizedPredictor):
        latent = model.encoder.out_features
        return input_dim * latent + latent * output_dim
    return sum(module.in_features * module.out_features for module in model.modules() if hasattr(module, "in_features"))


def predictor_accounting(model, input_dim, output_dim, retention):
    params = sum(parameter.numel() for parameter in model.parameters())
    macs = predictor_macs(model, input_dim, output_dim)
    dense = 3 * input_dim * output_dim
    return {"parameters": params, "fp32_bytes": params * 4, "fp16_bf16_bytes": params * 2, "int8_bytes": params, "macs_per_token": macs, "flops_per_token": 2 * macs, "dense_ffn_macs": dense, "predictor_mac_fraction": macs / dense, "selected_ffn_fraction": retention, "total_theoretical_compute_fraction": retention + macs / dense, "output_size": output_dim}
