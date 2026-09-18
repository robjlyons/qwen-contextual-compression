"""Cheap pre-FFN selector architectures with an x-only inference API."""
import torch
from torch import nn


class FactorizedPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.neurons = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        return self.neurons(self.dropout(self.encoder(x)))


class ResidualFactorizedPredictor(nn.Module):
    """Nonlinear latent residual context encoder and neuron-score head."""
    activation = "silu"
    residual_depth = 2
    layer_norm_enabled = True

    def __init__(self, input_dim, output_dim, latent_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.residual_fc1 = nn.Linear(latent_dim, latent_dim)
        self.residual_fc2 = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.neurons = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        residual = torch.nn.functional.silu(self.encoder(x))
        update = torch.nn.functional.silu(self.residual_fc1(residual))
        update = self.residual_fc2(self.dropout(update))
        return self.neurons(self.layer_norm(residual + update))


class LowRankMLP(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=256, dropout=0.0, activation="silu"):
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.GELU
        self.net = nn.Sequential(nn.Linear(input_dim, latent_dim), act(), nn.Dropout(dropout), nn.Linear(latent_dim, latent_dim), act(), nn.Linear(latent_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class StaticHot:
    def fit(self, scores):
        self.scores = scores.mean(0)
        return self

    def __call__(self, x):
        return self.scores.expand(*x.shape[:-1], -1)


MODEL_CLASSES = {"factorized": FactorizedPredictor, "residual_factorized": ResidualFactorizedPredictor, "mlp": LowRankMLP}


def create_predictor(kind, input_dim, output_dim, latent_dim, dropout=0.0):
    try:
        cls = MODEL_CLASSES[kind]
    except KeyError as error:
        raise ValueError(f"Unknown predictor architecture: {kind}") from error
    return cls(input_dim, output_dim, latent_dim, dropout)


def architecture_metadata(model, kind, input_dim, output_dim, latent_dim):
    metadata = {"kind": kind, "latent_dim": latent_dim, "input_dim": input_dim, "output_dim": output_dim}
    if isinstance(model, ResidualFactorizedPredictor):
        metadata.update(activation=model.activation, residual_depth=model.residual_depth, layer_norm=model.layer_norm_enabled)
    return metadata
