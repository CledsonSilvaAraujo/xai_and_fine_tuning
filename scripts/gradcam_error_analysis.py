#!/usr/bin/env python3
"""Grad-CAM++ em lote no teste com TP/TN/FP/FN e análise de erro."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_ROOT, IMAGE_SIZE, SEED
from src.data import ChestXRayBinaryDataModule


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(weights_path: Path, device: torch.device) -> tuple[nn.Module, str]:
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 2)
    m.load_state_dict(ck["model_state"])
    strategy = str(ck.get("strategy", weights_path.stem))
    return m.to(device).eval(), strategy


def soft_overlay_jet(rgb_uint8: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat_u8 = np.uint8(np.clip(heatmap01, 0.0, 1.0) * 255.0)
    jet_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = rgb_uint8.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * jet_rgb
    return np.uint8(np.clip(out, 0, 255))


def central_lung_proxy_mask(h: int, w: int, ratio: float = 0.6) -> np.ndarray:
    mh = int(h * ratio)
    mw = int(w * ratio)
    y0 = (h - mh) // 2
    x0 = (w - mw) // 2
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0 : y0 + mh, x0 : x0 + mw] = 1.0
    return mask


def overlap_outside_ratio(heatmap01: np.ndarray, lung_mask01: np.ndarray) -> float:
    hm = np.clip(heatmap01.astype(np.float32), 0.0, 1.0)
    total = float(hm.sum() + 1e-8)
    outside = float((hm * (1.0 - lung_mask01)).sum())
    return outside / total


def category_from_overlap(score: float) -> str:
    if score < 0.2:
        return "focused"
    if score <= 0.5:
        return "partial"
    return "distracted"


def pred_bucket(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gradcam")
    parser.add_argument("--error-json", type=Path, default=ROOT / "results" / "error_analysis.json")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--target-class", type=int, default=1, help="Classe alvo do CAM (0 ou 1)")
    parser.add_argument("--max-images", type=int, default=None, help="Limite opcional de imagens do teste")
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
    model, strategy = load_model(args.checkpoint, device)
    dm = ChestXRayBinaryDataModule(args.data_root)
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])

    for d in ("TP", "TN", "FP", "FN"):
        (args.out_dir / d).mkdir(parents=True, exist_ok=True)

    per_error: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    bucket_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    category_counts = {"focused": 0, "partial": 0, "distracted": 0}
    n = len(dm.test_ds.samples)
    limit = n if args.max_images is None else min(n, args.max_images)

    for i in range(limit):
        img_path, y_true = dm.test_ds.samples[i]
        rgb = Image.open(img_path).convert("RGB")
        rgb_resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
        rgb_np = np.asarray(rgb_resized, dtype=np.uint8)
        x, _ = dm.test_ds[i]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            y_pred = int(np.argmax(probs))
            conf = float(probs[y_pred])
            score_pneumonia = float(probs[1])

        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(args.target_class)])[0]
        overlay = soft_overlay_jet(rgb_np, grayscale_cam, alpha=args.alpha)

        bucket = pred_bucket(int(y_true), y_pred)
        bucket_counts[bucket] += 1
        out_name = f"{img_path.stem}_true{int(y_true)}_pred{y_pred}.png"
        overlay_path = args.out_dir / bucket / out_name
        Image.fromarray(overlay).save(overlay_path)
        predictions.append(
            {
                "file": str(img_path),
                "true_label": int(y_true),
                "pred_label": y_pred,
                "score_pneumonia": score_pneumonia,
                "pred_confidence": conf,
                "bucket": bucket,
                "overlay_file": str(overlay_path.resolve()),
            }
        )

        if bucket in ("FP", "FN"):
            lung_mask = central_lung_proxy_mask(IMAGE_SIZE, IMAGE_SIZE, ratio=0.6)
            overlap = overlap_outside_ratio(grayscale_cam, lung_mask)
            cat = category_from_overlap(overlap)
            category_counts[cat] += 1
            per_error.append(
                {
                    "file": str(img_path),
                    "true_label": int(y_true),
                    "pred_label": y_pred,
                    "score_pneumonia": score_pneumonia,
                    "pred_confidence": conf,
                    "bucket": bucket,
                    "overlap_score": round(float(overlap), 6),
                    "category": cat,
                    "overlay_file": str(overlay_path.resolve()),
                }
            )

    hard_set = [e for e in per_error if e["overlap_score"] > 0.3]
    payload = {
        "strategy": strategy,
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "n_test_images_processed": limit,
        "target_class_for_cam": args.target_class,
        "overlay_alpha": args.alpha,
        "bucket_counts": bucket_counts,
        "error_category_counts": category_counts,
        "predictions": predictions,
        "errors": per_error,
        "hard_set_overlap_gt_0.3": hard_set,
    }
    args.error_json.parent.mkdir(parents=True, exist_ok=True)
    args.error_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Overlays salvos em: {args.out_dir}")
    print(f"Análise de erro salva em: {args.error_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
