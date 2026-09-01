#!/usr/bin/env python3
"""
Residual Graph Interpolator.

Node features: [value_norm * mask, mask, x_n, y_n, elev_n] + CLC embedding.
Initial residual: h = alpha * h0 + (1 - alpha) * Mix(h).
Masked / query nodes do not send messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgi_graph import knn_edges_fast
from rgi_data import FeatureScaler


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros((n, src.size(1)))
    cnt = src.new_zeros((n, 1))
    out.index_add_(0, index, src)
    cnt.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp_min(1.0)


class MixLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.lin_src = nn.Linear(hidden, hidden)
        self.lin_out = nn.Linear(hidden * 2, hidden)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            src, dst = edge_index[0], edge_index[1]
            # log1p distance, raw dz already in metres — scale inside MLP
            e = edge_attr.clone()
            e[:, 0] = torch.log1p(e[:, 0].clamp_min(0.0) / 1000.0)
            e[:, 1] = e[:, 1] / 500.0
            msg = self.lin_src(h[src]) * torch.sigmoid(self.edge_mlp(e))
            agg = scatter_mean(msg, dst, h.size(0))
        return self.lin_out(torch.cat([h, agg], dim=-1))


class ResidualGNN(nn.Module):
    def __init__(
        self,
        n_clc: int,
        hidden: int = 64,
        n_layers: int = 3,
        alpha: float = 0.2,
        dropout: float = 0.1,
        clc_embed_dim: int = 8,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.n_layers = int(n_layers)
        self.clc_emb = nn.Embedding(max(n_clc, 1), clc_embed_dim)
        in_dim = 2 + 3 + clc_embed_dim  # value*mask, mask, xyz, clc
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([MixLayer(hidden) for _ in range(self.n_layers)])
        self.dropout = float(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, value, mask, xyz, clc_idx, edge_index, edge_attr):
        clc = self.clc_emb(clc_idx)
        x0 = torch.cat([value * mask, mask, xyz, clc], dim=-1)
        h0 = self.in_proj(x0)
        h = h0
        a = self.alpha
        for layer in self.layers:
            mix = layer(h, edge_index, edge_attr)
            mix = F.relu(mix)
            mix = F.dropout(mix, p=self.dropout, training=self.training)
            h = a * h0 + (1.0 - a) * mix
        return self.head(h).squeeze(-1)


@dataclass
class RGIConfig:
    k: int = 10
    n_layers: int = 3
    alpha: float = 0.2
    alpha_z: float = 10.0
    hidden_dim: int = 64
    dropout: float = 0.1
    clc_embed_dim: int = 8
    epochs: int = 80
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_mask_frac: float = 0.25
    batch_times: int = 8
    seed: int = 22
    min_stations: int = 10
    device: str = "auto"   # auto | cuda | cpu
    use_amp: bool = True


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("rgi.device=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        return f"cuda:{i} {name} ({mem:.1f} GiB)"
    return "cpu"


class RGI:
    """Train / predict wrapper. One instance per variable."""

    def __init__(self, cfg: RGIConfig):
        self.cfg = cfg
        self.scaler: Optional[FeatureScaler] = None
        self.net: Optional[ResidualGNN] = None
        self.device = resolve_device(cfg.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _build_net(self, n_clc: int) -> ResidualGNN:
        torch.manual_seed(self.cfg.seed)
        net = ResidualGNN(
            n_clc=n_clc,
            hidden=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
            alpha=self.cfg.alpha,
            dropout=self.cfg.dropout,
            clc_embed_dim=self.cfg.clc_embed_dim,
        )
        return net.to(self.device)

    def _tensors_for_frame(self, frame: pd.DataFrame, query_idx: np.ndarray, var: str):
        """frame: all nodes this timestamp (obs + queries). query_idx: positions that must not send."""
        n = len(frame)
        sender = np.ones(n, dtype=bool)
        sender[query_idx] = False
        elev = frame["elev"].to_numpy()
        ei, ea = knn_edges_fast(
            frame["x"].to_numpy(),
            frame["y"].to_numpy(),
            elev,
            sender,
            self.cfg.k,
            self.cfg.alpha_z,
        )
        raw_v = frame[var].to_numpy(dtype=np.float32)
        mask = np.ones((n, 1), dtype=np.float32)
        mask[query_idx] = 0.0
        value = self.scaler.transform_value(raw_v).reshape(-1, 1)
        value = value * mask
        xyz = self.scaler.coord_block(frame["x"], frame["y"], elev)
        clc = self.scaler.clc_index(frame["clc_code"])
        t = {
            "value": torch.from_numpy(value).to(self.device),
            "mask": torch.from_numpy(mask).to(self.device),
            "xyz": torch.from_numpy(xyz).to(self.device),
            "clc": torch.from_numpy(clc).to(self.device),
            "edge_index": torch.from_numpy(ei).to(self.device),
            "edge_attr": torch.from_numpy(ea).to(self.device),
        }
        y = torch.from_numpy(self.scaler.transform_value(raw_v)).to(self.device)
        return t, y

    def fit(
        self,
        train_df,
        var: str,
        val_df=None,
        log=print,
    ) -> dict:
        self.scaler = FeatureScaler().fit(train_df, var)
        self.net = self._build_net(len(self.scaler.clc_codes))
        if log:
            log(f"  device {describe_device(self.device)}  amp={self.cfg.use_amp and self.device.type=='cuda'}")
        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        rng = np.random.default_rng(self.cfg.seed)
        best_state = None
        best_val = np.inf
        stale = 0
        history = []
        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        try:
            scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        def epoch_loss(df, train_mode: bool) -> float:
            self.net.train(mode=train_mode)
            total = 0.0
            n_tok = 0
            groups = list(df.groupby("time", sort=False))
            if train_mode:
                rng.shuffle(groups)
            for _t, g in groups:
                g = g.reset_index(drop=True)
                if len(g) < self.cfg.min_stations:
                    continue
                n = len(g)
                n_q = max(1, int(round(self.cfg.train_mask_frac * n)))
                n_q = min(n_q, n - 3) if n > 3 else 1
                q = rng.choice(n, size=n_q, replace=False)
                batch, y = self._tensors_for_frame(g, q, var)
                try:
                    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
                except TypeError:
                    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
                with amp_ctx:
                    pred = self.net(
                        batch["value"], batch["mask"], batch["xyz"], batch["clc"],
                        batch["edge_index"], batch["edge_attr"],
                    )
                    q_t = torch.from_numpy(q.astype(np.int64)).to(self.device)
                    loss = F.mse_loss(pred[q_t], y[q_t])
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    scaler_amp.step(opt)
                    scaler_amp.update()
                total += float(loss.item()) * n_q
                n_tok += n_q
            return total / max(n_tok, 1)

        for epoch in range(1, self.cfg.epochs + 1):
            tr = epoch_loss(train_df, True)
            if val_df is not None and len(val_df):
                with torch.no_grad():
                    va = epoch_loss(val_df, False)
            else:
                va = tr
            history.append({"epoch": epoch, "train": tr, "val": va})
            if va + 1e-6 < best_val:
                best_val = va
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
            if stale >= self.cfg.patience:
                if log:
                    log(f"  early stop epoch {epoch}  val_mse={best_val:.4f}")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return {"best_val_mse": best_val, "epochs": history[-1]["epoch"] if history else 0}

    @torch.no_grad()
    def predict_frame(self, obs_df, query_df, var: str) -> np.ndarray:
        """obs_df sends, query_df receives. Same columns including dummy var on queries."""
        obs = obs_df.reset_index(drop=True)
        qry = query_df.reset_index(drop=True)
        frame = pd.concat([obs, qry], ignore_index=True)
        q_idx = np.arange(len(obs), len(frame))
        batch, _y = self._tensors_for_frame(frame, q_idx, var)
        pred = self.net(
            batch["value"], batch["mask"], batch["xyz"], batch["clc"],
            batch["edge_index"], batch["edge_attr"],
        )
        yhat = pred[len(obs):].detach().cpu().numpy()
        return self.scaler.inverse_value(yhat)

    def predict_stations_at_time(self, df_t, var: str, hold_names) -> np.ndarray:
        hold_names = set(hold_names)
        obs = df_t[~df_t["station_name"].isin(hold_names)]
        qry = df_t[df_t["station_name"].isin(hold_names)]
        if qry.empty:
            return np.zeros(0, dtype=np.float32)
        return self.predict_frame(obs, qry, var)

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "net": self.net.state_dict() if self.net else None,
        }

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path) -> "RGI":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = RGIConfig(**{k: v for k, v in blob["cfg"].items() if k in RGIConfig.__dataclass_fields__})
        obj = cls(cfg)
        obj.scaler = FeatureScaler.from_state(blob["scaler"])
        obj.net = obj._build_net(len(obj.scaler.clc_codes))
        obj.net.load_state_dict(blob["net"])
        obj.net.eval()
        return obj
