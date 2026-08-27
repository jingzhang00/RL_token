"""Paper-faithful RL token and compact online actor/critic models."""

from __future__ import annotations

import math

import torch
from torch import nn


def _transformer(
    width: int, heads: int, layers: int, mlp_ratio: int
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * mlp_ratio,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width))


class RLToken(nn.Module):
    """Append ``<rl>`` to VLA tokens and autoregressively reconstruct them."""

    def __init__(
        self,
        width: int = 2048,
        heads: int = 8,
        layers: int = 2,
        mlp_ratio: int = 4,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.width = width
        self.max_tokens = max_tokens
        self.rl = nn.Parameter(torch.randn(1, 1, width) / math.sqrt(width))
        self.encoder_pos = nn.Parameter(
            torch.randn(1, max_tokens + 1, width) / math.sqrt(width)
        )
        self.decoder_pos = nn.Parameter(
            torch.randn(1, max_tokens, width) / math.sqrt(width)
        )
        self.encoder = _transformer(width, heads, layers, mlp_ratio)
        self.decoder = _transformer(width, heads, layers, mlp_ratio)
        self.output = nn.Linear(width, width)

    def encode(
        self, tokens: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(
                f"tokens must be [B,M,{self.width}], got {tuple(tokens.shape)}"
            )
        batch, count, _ = tokens.shape
        if count > self.max_tokens:
            raise ValueError(
                f"token count {count} exceeds max_tokens={self.max_tokens}"
            )
        valid = (
            torch.ones((batch, count), dtype=torch.bool, device=tokens.device)
            if valid is None
            else valid.bool()
        )
        if valid.shape != (batch, count):
            raise ValueError(
                f"valid must be {(batch, count)}, got {tuple(valid.shape)}"
            )
        sequence = torch.cat([tokens, self.rl.expand(batch, -1, -1)], dim=1)
        sequence = sequence + self.encoder_pos[:, : count + 1]
        padding = torch.cat(
            [~valid, torch.zeros((batch, 1), dtype=torch.bool, device=tokens.device)],
            dim=1,
        )
        return self.encoder(sequence, src_key_padding_mask=padding)[:, -1]

    def decode(self, z_rl: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced predictions; VLA targets are deliberately stop-gradient."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(
                f"tokens must be [B,M,{self.width}], got {tuple(tokens.shape)}"
            )
        batch, count, _ = tokens.shape
        if z_rl.shape != (batch, self.width):
            raise ValueError(
                f"z_rl must be {(batch, self.width)}, got {tuple(z_rl.shape)}"
            )
        shifted = torch.cat([z_rl[:, None], tokens[:, :-1].detach()], dim=1)
        shifted = shifted + self.decoder_pos[:, :count]
        causal = torch.triu(
            torch.ones((count, count), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        return self.output(self.decoder(shifted, mask=causal, is_causal=True))

    def forward(
        self, tokens: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_rl = self.encode(tokens, valid)
        prediction = self.decode(z_rl, tokens)
        error = (prediction.float() - tokens.detach().float()).square().sum(dim=-1)
        if valid is None:
            loss = error.sum(dim=1).mean()
        else:
            weights = valid.to(error.dtype)
            loss = (error * weights).sum(dim=1).mean()
        return z_rl, prediction, loss


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden: int = 256, layers: int = 2
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        modules = []
        for index in range(layers):
            modules.extend(
                [
                    nn.Linear(input_dim if index == 0 else hidden, hidden),
                    nn.ReLU(),
                ]
            )
        modules.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class Actor(nn.Module):
    def __init__(
        self,
        z_dim: int = 2048,
        proprio_dim: int = 24,
        horizon: int = 10,
        action_dim: int = 12,
        hidden: int = 256,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.net = MLP(
            z_dim + proprio_dim + horizon * action_dim,
            horizon * action_dim,
            hidden,
            layers,
        )

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        flat = torch.cat([z, proprio, reference.flatten(1)], dim=-1)
        return self.net(flat).reshape(-1, self.horizon, self.action_dim)

    def sample(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        reference: torch.Tensor,
        sigma: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self(z, proprio, reference)
        return mean + torch.randn_like(mean) * sigma, mean


class Critic(nn.Module):
    def __init__(
        self,
        z_dim: int = 2048,
        proprio_dim: int = 24,
        horizon: int = 10,
        action_dim: int = 12,
        hidden: int = 256,
        layers: int = 2,
    ) -> None:
        super().__init__()
        state_dim = z_dim + proprio_dim
        self.q1 = MLP(state_dim + horizon * action_dim, 1, hidden, layers)
        self.q2 = MLP(state_dim + horizon * action_dim, 1, hidden, layers)

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat([z, proprio, action.flatten(1)], dim=-1)
        return self.q1(value).squeeze(-1), self.q2(value).squeeze(-1)


def reference_dropout(
    reference: torch.Tensor, probability: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the complete reference independently for each actor sample."""
    keep = (
        torch.rand((reference.shape[0], 1, 1), device=reference.device) >= probability
    )
    return reference * keep, keep.flatten()


def actor_loss(
    action: torch.Tensor,
    reference: torch.Tensor,
    q: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    return (-q + beta * (action - reference).square().flatten(1).sum(1)).mean()
