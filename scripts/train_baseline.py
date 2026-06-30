#!/usr/bin/env python3
"""
ResNet-50 — duas estratégias de transfer learning:
  head_only     : backbone congelada, treina só a camada fc (rápido, baseline forte).
  full_finetune : backbone + fc com learning rates diferentes (adaptação maior).

Ao final grava checkpoint e results/metrics_<strategy>.json (acurácia, erro %, AUC, F1, matriz).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_ROOT
from src.config import SEED
from src.data import ChestXRayBinaryDataModule, class_names
from src.evaluation import evaluate_split, save_metrics_json

RESULTS_DIR = ROOT / "results"

STRATEGIES: dict[str, dict[str, Path]] = {
    "head_only": {
        "checkpoint": ROOT / "checkpoints" / "resnet50_head_only.pt",
        "metrics": RESULTS_DIR / "metrics_head_only.json",
    },
    "full_finetune": {
        "checkpoint": ROOT / "checkpoints" / "resnet50_full_finetune.pt",
        "metrics": RESULTS_DIR / "metrics_full_finetune.json",
    },
}


def build_model(device: torch.device) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    m = models.resnet50(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 2)
    return m.to(device)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_strategy(model: nn.Module, strategy: str) -> None:
    if strategy == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif strategy == "full_finetune":
        # fase 2 do plano: destrava layer3 + layer4 + fc
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer3.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(strategy)


def build_optimizer(model: nn.Module, strategy: str, lr: float) -> torch.optim.Optimizer:
    if strategy == "head_only":
        return torch.optim.Adam(model.fc.parameters(), lr=lr)

    backbone_params: list[nn.Parameter] = []
    fc_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc"):
            fc_params.append(p)
        else:
            backbone_params.append(p)
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": fc_params, "lr": lr},
        ]
    )


def mean_ce_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item())
    return float(np.mean(losses))


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    losses: list[float] = []
    for x, y in tqdm(loader, desc="train"):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        required=True,
        help="head_only primeiro (baseline); depois full_finetune para comparar.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="head_only default 1e-3; full_finetune default 1e-4",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print("Dataset não encontrado.", file=sys.stderr)
        return 2

    set_global_seed(args.seed)

    defaults = STRATEGIES[args.strategy]
    checkpoint_path = args.checkpoint or defaults["checkpoint"]
    metrics_path = args.metrics_out or defaults["metrics"]

    if args.lr is None:
        lr = 1e-3 if args.strategy == "head_only" else 1e-4
    else:
        lr = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Classes:", class_names(), "| device:", device)
    print(f"Estratégia: {args.strategy} | lr={lr} | epochs={args.epochs} | seed={args.seed}")

    dm = ChestXRayBinaryDataModule(args.data_root)
    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    test_loader = dm.test_loader()
    class_weights = dm.class_weights_tensor().to(device)
    train_counts = dm.train_class_counts()

    model = build_model(device)
    apply_strategy(model, args.strategy)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, args.strategy, lr)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None

    print(
        f"Train class counts: NORMAL={train_counts[0]} PNEUMONIA={train_counts[1]} | "
        f"class_weights={class_weights.detach().cpu().tolist()}"
    )

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss = mean_ce_loss(model, val_loader, device, criterion)
        val_metrics = evaluate_split(model, val_loader, device, split_name="validation")
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
            f"val_acc={val_metrics['accuracy_percent']}%  "
            f"val_err={val_metrics['error_percent']}%  val_AUC={val_metrics['auc_roc']:.4f}  "
            f"val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["auc_roc"] > best_auc:
            best_auc = float(val_metrics["auc_roc"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Treino sem melhor estado válido.")

    model.load_state_dict(best_state)
    val_final = evaluate_split(model, val_loader, device, split_name="validation")
    test_final = evaluate_split(model, test_loader, device, split_name="test")

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": list(class_names()),
            "strategy": args.strategy,
            "epochs": args.epochs,
            "lr": lr,
            "seed": args.seed,
            "best_epoch_by_val_auc": best_epoch,
            "best_val_auc": best_auc,
            "class_weights": class_weights.detach().cpu().tolist(),
            "train_class_counts": train_counts,
        },
        checkpoint_path,
    )
    print(f"Checkpoint (melhor AUC val): {checkpoint_path} (epoch {best_epoch})")

    payload = {
        "strategy": args.strategy,
        "epochs": args.epochs,
        "lr": lr,
        "seed": args.seed,
        "data_root": str(args.data_root.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch_by_val_auc": best_epoch,
        "best_val_auc": best_auc,
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_class_counts": train_counts,
        "validation": val_final,
        "test": test_final,
        "notes": (
            "accuracy_percent = fração de acertos × 100; "
            "error_percent = fração de erros × 100. "
            "confusion_matrix sklearn com classes ordenadas [0,1]: [[TN, FP], [FN, TP]]."
        ),
    }
    save_metrics_json(metrics_path, payload)
    print(f"Métricas JSON: {metrics_path}")
    print(
        f"Teste — acerto {test_final['accuracy_percent']}% | erro {test_final['error_percent']}% | "
        f"AUC {test_final['auc_roc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
