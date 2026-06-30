"""Métricas de classificação binária (NORMAL vs PNEUMONIA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna y_true e probabilidade da classe PNEUMONIA (índice 1)."""
    model.eval()
    ys: list[int] = []
    probs: list[float] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        ys.extend(y.numpy().tolist())
        probs.extend(p.tolist())
    y_true = np.array(ys, dtype=int)
    y_score = np.array(probs, dtype=float)
    return y_true, y_score


def metrics_dict(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Acerto = acurácia global; Erro = 1 - acurácia (fração de predições erradas).
    Matriz de confusão no sklearn: labels [0,1] -> [[TN, FP], [FN, TP]].
    """
    y_pred = (y_score >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "accuracy_percent": round(acc * 100, 2),
        "error_percent": round((1.0 - acc) * 100, 2),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    split_name: str,
) -> dict[str, Any]:
    y_true, y_score = collect_predictions(model, loader, device)
    m = metrics_dict(y_true, y_score)
    m["split"] = split_name
    return m
