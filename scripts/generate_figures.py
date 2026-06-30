#!/usr/bin/env python3
"""Gera figuras do artigo em PDF (300 DPI): ROC, matriz de confusão e barras de erro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.xticks([0.5, 1.5], ["NORMAL", "PNEUMONIA"])
    plt.yticks([0.5, 1.5], ["NORMAL", "PNEUMONIA"], rotation=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()


def plot_error_categories(counts: dict[str, int], out_path: Path, title: str) -> None:
    labels = ["focused", "partial", "distracted"]
    values = [int(counts.get(k, 0)) for k in labels]
    palette = {"focused": "#2ca02c", "partial": "#ffbf00", "distracted": "#d62728"}
    plt.figure(figsize=(6, 4), dpi=300)
    sns.barplot(x=labels, y=values, hue=labels, palette=palette, legend=False)
    plt.title(title)
    plt.xlabel("Categoria")
    plt.ylabel("Contagem")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()


def plot_roc_real(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = float(auc(fpr, tpr))
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Aleatório")
    plt.plot(fpr, tpr, color="navy", label=f"AUC = {auc_value:.4f}")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()
    return auc_value


def _load_overlay(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def plot_error_grid_4x4(errors: list[dict], out_path: Path, title: str) -> None:
    fps = [e for e in errors if e.get("bucket") == "FP"]
    fns = [e for e in errors if e.get("bucket") == "FN"]
    fps = sorted(fps, key=lambda e: e.get("overlap_score", 0.0), reverse=True)[:8]
    fns = sorted(fns, key=lambda e: e.get("overlap_score", 0.0), reverse=True)[:8]

    fig, axes = plt.subplots(4, 4, figsize=(11, 11), dpi=300)
    fig.suptitle(title, fontsize=12)
    all_items = fps + fns
    labels = (["FP"] * len(fps)) + (["FN"] * len(fns))
    while len(all_items) < 16:
        all_items.append(None)
        labels.append("")

    for i, ax in enumerate(axes.flatten()):
        item = all_items[i]
        tag = labels[i]
        ax.axis("off")
        if item is None:
            continue
        try:
            img = _load_overlay(item["overlay_file"])
            ax.imshow(img)
            ov = float(item.get("overlap_score", 0.0))
            ax.set_title(f"{tag} | overlap={ov:.2f}", fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, f"{tag}\nsem imagem", ha="center", va="center", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics_full_finetune.json")
    parser.add_argument("--error-json", type=Path, default=ROOT / "results" / "error_analysis.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    cm = np.array(metrics["test"]["confusion_matrix"], dtype=int)
    strat = metrics.get("strategy", "model")
    plot_confusion_matrix(cm, args.out_dir / f"confusion_matrix_{strat}.pdf", f"Matriz de confusão ({strat})")

    if args.error_json.is_file():
        err = load_json(args.error_json)
        preds = err.get("predictions", [])
        if preds:
            y_true = np.array([int(p["true_label"]) for p in preds], dtype=int)
            y_score = np.array([float(p["score_pneumonia"]) for p in preds], dtype=float)
            auc_real = plot_roc_real(
                y_true,
                y_score,
                args.out_dir / f"roc_curve_{strat}.pdf",
                f"Curva ROC ({strat})",
            )
            print(f"AUC ROC real ({strat}): {auc_real:.4f}")
        else:
            print("Aviso: predictions ausente em error_analysis.json; rode gradcam_error_analysis novamente.")
        counts = err.get("error_category_counts", {})
        plot_error_categories(
            counts, args.out_dir / f"error_categories_{strat}.pdf", f"Categorias de erro ({strat})"
        )
        plot_error_grid_4x4(
            err.get("errors", []),
            args.out_dir / f"error_grid_4x4_{strat}.pdf",
            f"Grid 4x4 de erros (8 FP + 8 FN) - {strat}",
        )

    print(f"Figuras em: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
