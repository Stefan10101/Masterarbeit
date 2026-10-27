#!/usr/bin/env python3
"""Localized SMACNP core with proper batched k-NN attention.

Ported from Dumitrescu 2026 climate-gridder (LocalizedMeanAttributeEncoder
and LocalizedVarianceEncoder). Per-variable, two-step capable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_cluster import knn
except ImportError:
    knn = None


def _create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    dropout_rate: float,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []
    prev = input_dim
    for _ in range(max(0, num_hidden_layers)):
        layers.append(nn.Linear(prev, hidden_size))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev = hidden_size
    layers.append(nn.Linear(prev, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, spatial_dim: int = 2, encoding_dim: int = 32):
        super().__init__()
        self.encoding_dim = encoding_dim
        half = encoding_dim // 2
        self.freqs = nn.Parameter(torch.linspace(0.1, 10.0, half), requires_grad=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords (..., 2)
        freqs = self.freqs.to(coords.device)
        x = coords[..., 0:1] * freqs
        y = coords[..., 1:2] * freqs
        pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
        if pe.shape[-1] > self.encoding_dim:
            pe = pe[..., : self.encoding_dim]
        elif pe.shape[-1] < self.encoding_dim:
            pe = F.pad(pe, (0, self.encoding_dim - pe.shape[-1]))
        return pe


def find_k_nearest_context_points(
    s_target: torch.Tensor,
    s_context: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, int]:
    """Batched k-NN. Returns (B, M, k) local context indices and actual k used.

    Pairwise distances are used. Station counts are ~1k so this is cheap
    and avoids torch_cluster batch-index size mismatches.
    """
    B, M, D = s_target.shape
    N = s_context.shape[1]
    k = min(k, N)
    if M == 0 or k == 0:
        return torch.empty((B, 0, k), dtype=torch.long, device=s_target.device), k

    dist = torch.cdist(s_target, s_context)  # (B, M, N)
    nearest = dist.topk(k, dim=-1, largest=False).indices  # (B, M, k)
    return nearest, k


class MultiHeadCrossAttentionWrapper(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(query, key, value)
        return out


class LocalizedMeanAttributeEncoder(nn.Module):
    """Paper-faithful localized mean-attribute encoder with separate key/value and batched k-NN."""

    def __init__(
        self,
        input_dim: int,
        y_dim: int,
        hidden_size: int,
        r_dim: int,
        num_heads: int,
        dropout_rate: float,
        k: int,
        num_layers_mlp: int = 1,
        use_positional_encoding: bool = True,
        pe_dim: int = 32,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.k = k
        self.r_dim = r_dim
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        self.pos_encoder = SpatialPositionalEncoding(2, pe_dim) if use_positional_encoding else None
        pe_extra = pe_dim if use_positional_encoding else 0
        actual_input = input_dim + pe_extra

        self.mlp_value = _create_mlp(actual_input + y_dim, r_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.mlp_key = _create_mlp(actual_input, r_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.mlp_query = _create_mlp(actual_input, r_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.cross_attention = MultiHeadCrossAttentionWrapper(r_dim, num_heads, dropout_rate)

        if use_layer_norm:
            self.key_norm = nn.LayerNorm(r_dim)
            self.value_norm = nn.LayerNorm(r_dim)
            self.query_norm = nn.LayerNorm(r_dim)
            self.output_norm = nn.LayerNorm(r_dim)

    def forward(
        self,
        x_context_all: torch.Tensor,
        y_context: torch.Tensor,
        x_target_all: torch.Tensor,
        s_context: torch.Tensor,
        s_target: torch.Tensor,
    ) -> torch.Tensor:
        B, M, _ = x_target_all.shape
        N = x_context_all.shape[1]
        D = self.r_dim

        if self.use_positional_encoding:
            pe_c = self.pos_encoder(s_context)
            pe_t = self.pos_encoder(s_target)
            x_c = torch.cat([x_context_all, pe_c], dim=-1)
            x_t = torch.cat([x_target_all, pe_t], dim=-1)
        else:
            x_c = x_context_all
            x_t = x_target_all

        r_value = self.mlp_value(torch.cat([x_c, y_context], dim=-1))
        r_key = self.mlp_key(x_c)
        r_query = self.mlp_query(x_t)

        if self.use_layer_norm:
            r_value = self.value_norm(r_value)
            r_key = self.key_norm(r_key)
            r_query = self.query_norm(r_query)

        nearest_indices, final_k = find_k_nearest_context_points(s_target, s_context, min(self.k, N))
        batch_idx = torch.arange(B, device=r_key.device).view(B, 1, 1).expand(B, M, final_k)
        gathered_keys = r_key[batch_idx, nearest_indices]
        gathered_values = r_value[batch_idx, nearest_indices]

        q = r_query.reshape(B * M, 1, D)
        k = gathered_keys.reshape(B * M, final_k, D)
        v = gathered_values.reshape(B * M, final_k, D)
        out = self.cross_attention(q, k, v).reshape(B, M, D)
        if self.use_layer_norm:
            out = self.output_norm(out)
        return out


class LocalizedVarianceEncoder(nn.Module):
    """Localized variance encoder (no y in the representation)."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        v_dim: int,
        num_heads: int,
        dropout_rate: float,
        k: int,
        num_layers_mlp: int = 1,
        use_positional_encoding: bool = True,
        pe_dim: int = 32,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.k = k
        self.v_dim = v_dim
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        self.pos_encoder = SpatialPositionalEncoding(2, pe_dim) if use_positional_encoding else None
        pe_extra = pe_dim if use_positional_encoding else 0
        actual_input = input_dim + pe_extra

        self.mlp_key = _create_mlp(actual_input, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.mlp_query = _create_mlp(actual_input, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.cross_attention = MultiHeadCrossAttentionWrapper(v_dim, num_heads, dropout_rate)

        if use_layer_norm:
            self.key_norm = nn.LayerNorm(v_dim)
            self.query_norm = nn.LayerNorm(v_dim)
            self.output_norm = nn.LayerNorm(v_dim)

    def forward(
        self,
        x_context_all: torch.Tensor,
        x_target_all: torch.Tensor,
        s_context: torch.Tensor,
        s_target: torch.Tensor,
    ) -> torch.Tensor:
        B, M, _ = x_target_all.shape
        N = x_context_all.shape[1]
        D = self.v_dim

        if self.use_positional_encoding:
            pe_c = self.pos_encoder(s_context)
            pe_t = self.pos_encoder(s_target)
            x_c = torch.cat([x_context_all, pe_c], dim=-1)
            x_t = torch.cat([x_target_all, pe_t], dim=-1)
        else:
            x_c = x_context_all
            x_t = x_target_all

        v_key = self.mlp_key(x_c)
        v_query = self.mlp_query(x_t)
        if self.use_layer_norm:
            v_key = self.key_norm(v_key)
            v_query = self.query_norm(v_query)

        nearest_indices, final_k = find_k_nearest_context_points(s_target, s_context, min(self.k, N))
        batch_idx = torch.arange(B, device=v_key.device).view(B, 1, 1).expand(B, M, final_k)
        gathered_keys = v_key[batch_idx, nearest_indices]

        q = v_query.reshape(B * M, 1, D)
        k = gathered_keys.reshape(B * M, final_k, D)
        out = self.cross_attention(q, k, k).reshape(B, M, D)
        if self.use_layer_norm:
            out = self.output_norm(out)
        return out


class MeanLocationEncoder(nn.Module):
    def __init__(self, spatial_dim: int, y_dim: int, hidden_size: int, w_dim: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.mlp = _create_mlp(spatial_dim + y_dim, w_dim, hidden_size, num_layers, dropout)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, s_context: torch.Tensor, y_context: torch.Tensor, s_target: torch.Tensor) -> torch.Tensor:
        s_context = s_context.float()
        s_target = s_target.float()
        y_context = y_context.float()
        w_ctx = self.mlp(torch.cat([s_context, y_context], dim=-1))
        temp = torch.exp(self.log_temp).clamp(min=1e-3, max=10.0)
        dist = torch.cdist(s_target, s_context, p=1.0)
        attn = torch.softmax(-dist / temp, dim=-1)
        return torch.bmm(attn, w_ctx)


class MeanDecoder(nn.Module):
    def __init__(self, r_dim: int, w_dim: int, spatial_dim: int, attr_dim: int, hidden_size: int, out_dim: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.mlp = _create_mlp(r_dim + w_dim + spatial_dim + attr_dim, out_dim, hidden_size, num_layers, dropout)

    def forward(self, r: torch.Tensor, w: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([r, w, s, x], dim=-1))


class VarianceDecoder(nn.Module):
    def __init__(self, v_dim: int, spatial_dim: int, attr_dim: int, hidden_size: int, out_dim: int, dropout: float, num_layers: int = 1):
        super().__init__()
        # Predict log-variance; exp is more stable than raw softplus under NLL.
        self.mlp = _create_mlp(v_dim + spatial_dim + attr_dim, out_dim, hidden_size, num_layers, dropout)

    def forward(self, v: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        log_var = self.mlp(torch.cat([v, s, x], dim=-1))
        return torch.exp(log_var.clamp(-8.0, 8.0))


class SMACNP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        spatial_dim: int = 2,
        attr_dim: int = 3,
        y_dim: int = 1,
        two_step: bool = False,
        r_dim: int = 64,
        w_dim: int = 256,
        v_dim: int = 512,
        hidden_size: int = 512,
        num_layers_mlp: int = 3,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        k_localized: int = 15,
        use_positional_encoding: bool = True,
        pe_dim: int = 32,
        use_layer_norm: bool = False,
        variance_floor: float = 2e-4,
    ):
        super().__init__()
        self.two_step = two_step
        self.variance_floor = variance_floor
        self.spatial_dim = spatial_dim
        self.attr_dim = attr_dim

        self.mean_loc = MeanLocationEncoder(spatial_dim, y_dim, hidden_size, w_dim, dropout_rate, num_layers_mlp)
        self.mean_attr = LocalizedMeanAttributeEncoder(
            attr_dim, y_dim, hidden_size, r_dim, num_heads, dropout_rate, k_localized,
            num_layers_mlp, use_positional_encoding, pe_dim, use_layer_norm,
        )
        self.var_enc = LocalizedVarianceEncoder(
            attr_dim, hidden_size, v_dim, num_heads, dropout_rate, k_localized,
            num_layers_mlp, use_positional_encoding, pe_dim, use_layer_norm,
        )
        self.mean_dec = MeanDecoder(r_dim, w_dim, spatial_dim, attr_dim, hidden_size, 1, dropout_rate, num_layers_mlp)
        self.var_dec = VarianceDecoder(v_dim, spatial_dim, attr_dim, hidden_size, 1, dropout_rate, num_layers_mlp)
        if two_step:
            self.occ_dec = MeanDecoder(r_dim, w_dim, spatial_dim, attr_dim, hidden_size, 1, dropout_rate, num_layers_mlp)
            self.amount_mean_dec = MeanDecoder(r_dim, w_dim, spatial_dim, attr_dim, hidden_size, 1, dropout_rate, num_layers_mlp)
            self.amount_var_dec = VarianceDecoder(v_dim, spatial_dim, attr_dim, hidden_size, 1, dropout_rate, num_layers_mlp)

    def forward(self, x_context_all: torch.Tensor, y_context: torch.Tensor, x_target_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_context_all = x_context_all.float()
        x_target_all = x_target_all.float()
        y_context = y_context.float()
        s_ctx = x_context_all[..., : self.spatial_dim]
        x_attr_ctx = x_context_all[..., self.spatial_dim :]
        s_tgt = x_target_all[..., : self.spatial_dim]
        x_attr_tgt = x_target_all[..., self.spatial_dim :]

        w_star = self.mean_loc(s_ctx, y_context, s_tgt)
        r_star = self.mean_attr(x_attr_ctx, y_context, x_attr_tgt, s_ctx, s_tgt)
        v_star = self.var_enc(x_attr_ctx, x_attr_tgt, s_ctx, s_tgt)

        if self.two_step:
            return {
                "occurrence_logits": self.occ_dec(r_star, w_star, s_tgt, x_attr_tgt),
                "amount_mu": self.amount_mean_dec(r_star, w_star, s_tgt, x_attr_tgt),
                "amount_variance": self.amount_var_dec(v_star, s_tgt, x_attr_tgt).clamp(min=self.variance_floor),
            }
        return {
            "mu": self.mean_dec(r_star, w_star, s_tgt, x_attr_tgt),
            "variance": self.var_dec(v_star, s_tgt, x_attr_tgt).clamp(min=self.variance_floor),
        }


def gaussian_nll(mu: torch.Tensor, var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = var.clamp(min=1e-4, max=1e4)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
    nll = 0.5 * (torch.log(2 * math.pi * var) + (target - mu) ** 2 / var)
    return torch.nan_to_num(nll, nan=0.0, posinf=1e4, neginf=1e4).mean()


def two_step_loss(
    pred: Dict[str, torch.Tensor],
    y: torch.Tensor,
    y_raw: torch.Tensor | None = None,
    pos_weight_cap: float = 20.0,
) -> torch.Tensor:
    """Occurrence from raw values; amount NLL on the scaled targets."""
    raw = y if y_raw is None else y_raw
    is_pos = (raw > 1e-6).float()
    num_pos = is_pos.sum().clamp(min=1)
    num_neg = is_pos.numel() - num_pos
    pos_weight = (num_neg / num_pos).clamp(max=pos_weight_cap)
    bce = F.binary_cross_entropy_with_logits(pred["occurrence_logits"], is_pos, pos_weight=pos_weight)
    mask = is_pos.bool()
    if mask.any():
        nll = gaussian_nll(pred["amount_mu"][mask], pred["amount_variance"][mask], y[mask])
    else:
        nll = torch.tensor(0.0, device=y.device)
    return bce + nll
