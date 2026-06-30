#!/usr/bin/env python3
"""
XAI-Guided Fine-Tuning — Semana 6

Estratégia: Attention-Guided Sample Weighting
─────────────────────────────────────────────
Fase 1  (pré-computa):
    Roda Grad-CAM++ em TODO o conjunto de TREINO usando o checkpoint
    full_finetune já existente. Calcula overlap_score para cada imagem
    (fração da ativação fora da máscara pulmonar proxy de 60%).

Fase 2  (fine-tuning guiado):
    Usa os overlap_scores como pesos de sample na função de perda:
        per_sample_weight = 1 + ALPHA * overlap_score
    Amostras onde o modelo "se distrai" (atenção espúria fora do pulmão)
    recebem peso maior → modelo aprende a corrigir esses casos.
    Treina por N épocas com LR muito baixo a partir do checkpoint full_finetune.

Saídas:
    checkpoints/resnet50_xai_guided.pt
    results/metrics_xai_guided.json
    results/xai_cam_weights.json    ← overlap_scores do treino (para análise)

Uso:
    python scripts/xai_guided_finetune.py \
        --checkpoint checkpoints/resnet50_full_finetune.pt \
        --epochs 5 --lr 1e-5 --alpha 2.0

    # Modo rápido (pré-computa só N imagens para testar o pipeline):
    python scripts/xai_guided_finetune.py \
        --checkpoint checkpoints/resnet50_full_finetune.pt \
        --epochs 2 --lr 1e-5 --max-cam-images 100
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_ROOT, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED
from src.data import ChestXRayBinaryDataModule
from src.data.chest_xray import ChestXRayBinaryDataset
from src.evaluation import evaluate_split, save_metrics_json


# ──────────────────────────────────────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 2)
    m.load_state_dict(ck["model_state"])
    return m.to(device)


def central_lung_mask(h: int, w: int, ratio: float = 0.6) -> np.ndarray:
    mh, mw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - mh) // 2, (w - mw) // 2
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y0 + mh, x0:x0 + mw] = 1.0
    return mask


def overlap_outside(heatmap01: np.ndarray, lung_mask: np.ndarray) -> float:
    hm = np.clip(heatmap01.astype(np.float32), 0.0, 1.0)
    total = float(hm.sum() + 1e-8)
    outside = float((hm * (1.0 - lung_mask)).sum())
    return outside / total


# ──────────────────────────────────────────────────────────────────────────────
# Fase 1: Pré-computar overlap_score no conjunto de treino
# ──────────────────────────────────────────────────────────────────────────────

def precompute_overlap_scores(
    model: nn.Module,
    data_root: Path,
    device: torch.device,
    max_images: int | None,
    save_path: Path,
) -> dict[str, float]:
    """Retorna {caminho_da_imagem: overlap_score} para o conjunto de treino."""
    from torchvision import transforms

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # Dataset de treino completo com eval transforms (sem augmentation para CAM estável)
    train_ds = ChestXRayBinaryDataset(data_root, "train", transform=eval_tf)
    n = len(train_ds.samples)
    limit = n if max_images is None else min(n, max_images)

    model.eval()
    cam_extractor = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    lung_mask = central_lung_mask(IMAGE_SIZE, IMAGE_SIZE, ratio=0.6)

    scores: dict[str, float] = {}
    print(f"\nFase 1 — Pré-computando Grad-CAM++ em {limit}/{n} imagens de treino...")

    for i in tqdm(range(limit), desc="CAM treino"):
        img_path, _ = train_ds.samples[i]
        x, y = train_ds[i]
        x_t = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_t)
            y_pred = int(torch.argmax(logits, dim=1).item())

        grayscale_cam = cam_extractor(
            input_tensor=x_t,
            targets=[ClassifierOutputTarget(y_pred)],
        )[0]
        score = overlap_outside(grayscale_cam, lung_mask)
        scores[str(img_path)] = round(float(score), 6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps({"n_images": limit, "scores": scores}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Overlap scores salvos em: {save_path}")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Fase 2: Dataset ponderado por overlap_score
# ──────────────────────────────────────────────────────────────────────────────

class WeightedLossDataset(torch.utils.data.Dataset):
    """
    Wraps um ChestXRayBinaryDataset (ou Subset) e adiciona um peso de perda por amostra.
    O peso é calculado como: 1 + alpha * overlap_score.
    Amostras sem score recebem peso 1.0 (neutro).
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        overlap_scores: dict[str, float],
        alpha: float = 2.0,
    ):
        self.base = base_dataset
        self.alpha = alpha
        self.weights: list[float] = []
        # Suporta Subset (tem .indices + .dataset.samples) ou dataset direto (tem .samples)
        if hasattr(base_dataset, "indices"):
            samples = [base_dataset.dataset.samples[i] for i in base_dataset.indices]
        else:
            samples = base_dataset.samples
        for img_path, _ in samples:
            score = overlap_scores.get(str(img_path), 0.0)
            self.weights.append(1.0 + alpha * score)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        w = self.weights[idx]
        return x, y, torch.tensor(w, dtype=torch.float32)


def weighted_loss_collate(batch):
    xs = torch.stack([item[0] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    ws = torch.stack([item[2] for item in batch])
    return xs, ys, ws


# ──────────────────────────────────────────────────────────────────────────────
# Loop de treino com per-sample weights
# ──────────────────────────────────────────────────────────────────────────────

def train_weighted_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion_base: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    """CrossEntropyLoss redução 'none' multiplicado pelo per-sample weight."""
    model.train()
    # Cria criterion sem redução para aplicar pesos manuais
    criterion_none = nn.CrossEntropyLoss(
        weight=criterion_base.weight, reduction="none"
    )
    losses: list[float] = []
    for x, y, w in tqdm(loader, desc="xai train"):
        x, y, w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss_per_sample = criterion_none(logits, y)  # [B]
        loss = (loss_per_sample * w).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def mean_ce_loss_std(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item())
    return float(np.mean(losses))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="XAI-guided fine-tuning com attention-guided sample weighting"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints" / "resnet50_full_finetune.pt",
        help="Checkpoint de partida (full_finetune recomendado)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Épocas de fine-tuning guiado (recomendado: 3–5)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (muito baixo para não destruir features)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Multiplicador de peso para amostras 'distraídas'")
    parser.add_argument("--max-cam-images", type=int, default=None,
                        help="Limita Grad-CAM no treino (útil para teste rápido em CPU)")
    parser.add_argument(
        "--cam-weights-json", type=Path,
        default=ROOT / "results" / "xai_cam_weights.json",
        help="Onde salvar/ler os overlap_scores pré-computados",
    )
    parser.add_argument(
        "--checkpoint-out", type=Path,
        default=ROOT / "checkpoints" / "resnet50_xai_guided.pt",
    )
    parser.add_argument(
        "--metrics-out", type=Path,
        default=ROOT / "results" / "metrics_xai_guided.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Checkpoint não encontrado: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.data_root.is_dir():
        print(f"Dataset não encontrado: {args.data_root}", file=sys.stderr)
        return 2

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | lr={args.lr} | epochs={args.epochs} | alpha={args.alpha}")

    # num_workers=0 evita erros de shared memory no WSL2
    dm = ChestXRayBinaryDataModule(args.data_root, num_workers=0)

    # ── Fase 1: overlap_scores ─────────────────────────────────────────────────
    if args.cam_weights_json.is_file():
        print(f"\nFase 1 — Reutilizando scores existentes: {args.cam_weights_json}")
        raw = json.loads(args.cam_weights_json.read_text(encoding="utf-8"))
        overlap_scores: dict[str, float] = raw["scores"]
    else:
        model_phase1 = load_model(args.checkpoint, device)
        overlap_scores = precompute_overlap_scores(
            model_phase1, args.data_root, device, args.max_cam_images, args.cam_weights_json
        )
        del model_phase1

    # Estatísticas dos scores
    scores_arr = np.array(list(overlap_scores.values()))
    print(f"\nOverlap scores — treino:")
    print(f"  média={scores_arr.mean():.4f}  mediana={np.median(scores_arr):.4f}"
          f"  max={scores_arr.max():.4f}  min={scores_arr.min():.4f}")
    print(f"  focused  (<0.2):  {(scores_arr < 0.2).sum()} amostras")
    print(f"  partial  (0.2–0.5): {((scores_arr >= 0.2) & (scores_arr <= 0.5)).sum()} amostras")
    print(f"  distracted (>0.5): {(scores_arr > 0.5).sum()} amostras")

    # ── Fase 2: fine-tuning com pesos guiados por XAI ─────────────────────────
    print("\nFase 2 — Fine-tuning guiado por XAI...")
    model = load_model(args.checkpoint, device)

    # Descongela layer3 + layer4 + fc (mesma configuração do full_finetune)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    # Datasets ponderados — usa o subset de treino (dm._train_subset)
    weighted_train_ds = WeightedLossDataset(dm._train_subset, overlap_scores, alpha=args.alpha)
    weighted_train_loader = torch.utils.data.DataLoader(
        weighted_train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=weighted_loss_collate,
    )
    val_loader  = dm.val_loader()
    test_loader = dm.test_loader()

    class_weights = dm.class_weights_tensor().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer com LR diferenciado (backbone mais lento)
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("fc")]
    fc_params = list(model.fc.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": fc_params,       "lr": args.lr},
    ])

    best_auc = -1.0
    best_state: dict | None = None
    best_epoch = 0
    best_val_metrics: dict | None = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_weighted_epoch(model, weighted_train_loader, device, criterion, optimizer)
        va_loss = mean_ce_loss_std(model, val_loader, device, criterion)
        val_m = evaluate_split(model, val_loader, device, split_name="validation")
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
            f"val_acc={val_m['accuracy_percent']}%  val_AUC={val_m['auc_roc']:.4f}  "
            f"val_F1={val_m['f1']:.4f}"
        )
        if val_m["auc_roc"] > best_auc:
            best_auc = float(val_m["auc_roc"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_m

    if best_state is None:
        raise RuntimeError("Sem estado válido após treino.")

    model.load_state_dict(best_state)
    val_final  = evaluate_split(model, val_loader,  device, split_name="validation")
    test_final = evaluate_split(model, test_loader, device, split_name="test")

    # Salva checkpoint
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": ["NORMAL", "PNEUMONIA"],
        "strategy": "xai_guided",
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "seed": args.seed,
        "best_epoch_by_val_auc": best_epoch,
        "best_val_auc": best_auc,
        "class_weights": class_weights.detach().cpu().tolist(),
    }, args.checkpoint_out)
    print(f"\nCheckpoint XAI-guided salvo: {args.checkpoint_out}")

    # Salva métricas
    payload: dict[str, Any] = {
        "strategy": "xai_guided",
        "base_checkpoint": str(args.checkpoint.resolve()),
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "seed": args.seed,
        "max_cam_images": args.max_cam_images,
        "best_epoch_by_val_auc": best_epoch,
        "best_val_auc": best_auc,
        "class_weights": class_weights.detach().cpu().tolist(),
        "overlap_stats": {
            "n": int(len(scores_arr)),
            "mean": round(float(scores_arr.mean()), 6),
            "median": round(float(np.median(scores_arr)), 6),
            "max": round(float(scores_arr.max()), 6),
            "focused": int((scores_arr < 0.2).sum()),
            "partial": int(((scores_arr >= 0.2) & (scores_arr <= 0.5)).sum()),
            "distracted": int((scores_arr > 0.5).sum()),
        },
        "validation": val_final,
        "test": test_final,
        "notes": (
            "XAI-guided fine-tuning: pesos per-sample = 1 + alpha * overlap_score. "
            "accuracy_percent = fração de acertos × 100. "
            "confusion_matrix sklearn: [[TN, FP], [FN, TP]]."
        ),
    }
    save_metrics_json(args.metrics_out, payload)
    print(f"Métricas JSON: {args.metrics_out}")
    print(
        f"\n{'='*60}\n"
        f"RESULTADO XAI-GUIDED FINE-TUNING\n"
        f"  Acurácia: {test_final['accuracy_percent']}%  "
        f"Erro: {test_final['error_percent']}%\n"
        f"  AUC-ROC:  {test_final['auc_roc']:.4f}  "
        f"F1: {test_final['f1']:.4f}\n"
        f"  Precisão: {test_final['precision']:.4f}  "
        f"Recall: {test_final['recall']:.4f}\n"
        f"{'='*60}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
