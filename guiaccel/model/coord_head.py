"""Coordinate Regression Head for GUI Agent inference acceleration.

Replaces autoregressive token-by-token coordinate decoding with a single
MLP forward pass.  The head is trained on (hidden_state, ground_truth_coord)
pairs extracted from Qwen3-VL at the action-type token position.

Architecture
------------
    h_t ∈ R^4096 → Linear(4096, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 2) → Sigmoid
    output: [x̂, ŷ] ∈ [0, 1]²  →  de-normalize to 0-999 scale
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class CoordRegressionHead(nn.Module):
    """Lightweight MLP that regresses (x, y) coordinates from a VLM hidden state."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # output in [0, 1]²
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict normalised (x, y) from hidden state(s).

        Parameters
        ----------
        h : Tensor of shape ``(*, input_dim)``

        Returns
        -------
        Tensor of shape ``(*, 2)`` with values in ``[0, 1]``.
        """
        return self.mlp(h)

    def predict_999(self, h: torch.Tensor) -> torch.Tensor:
        """Predict coordinates on the 0-999 Qwen3-VL scale."""
        normalised = self.forward(h)
        return torch.round(normalised * 999.0).long()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@dataclass
class CoordHeadTrainConfig:
    """Hyper-parameters for regression head training."""

    input_dim: int = 4096
    hidden_dim: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 256
    patience: int = 10
    smooth_l1_beta: float = 0.01
    val_fraction: float = 0.2
    seed: int = 42


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """Smooth L1 loss on normalised [0, 1] coordinates."""
    return F.smooth_l1_loss(pred, target, beta=beta)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_coord_head(
    model: CoordRegressionHead,
    config: CoordHeadTrainConfig,
    path: Path,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Save model weights + training config to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": asdict(config),
        **(extra_meta or {}),
    }
    torch.save(
        {"state_dict": model.state_dict(), "meta": meta},
        path,
    )
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_coord_head(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CoordRegressionHead, dict[str, Any]]:
    """Load a saved CoordRegressionHead checkpoint.

    Returns (model, meta_dict).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {})
    cfg = meta.get("config", {})
    model = CoordRegressionHead(
        input_dim=int(cfg.get("input_dim", 4096)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, meta
