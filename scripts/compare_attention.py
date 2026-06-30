#!/usr/bin/env python3
"""
Gera figuras comparativas entre os modelos full_finetune, xai_guided_v1 e xai_guided_v2:

  Figura 1 — grade_3models.pdf
      Grade N×3 mostrando os mesmos FPs (corrigidos pelo XAI) com os overlays
      Grad-CAM++ dos três modelos lado a lado.

  Figura 2 — overlap_score_distribution.pdf
      Boxplot + stripplot do overlap_score dos erros por modelo
      (full_finetune vs xai_v1 vs xai_v2).

  Figura 3 — metrics_4models.pdf
      Tabela visual comparativa de todas as métricas para os 4 modelos
      (head_only, full_finetune, xai_v1, xai_v2).

Uso:
    python scripts/compare_attention.py
    python scripts/compare_attention.py --max-grid 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cores
# ─────────────────────────────────────────────────────────────────────────────
COL_FT   = "#1A6CB0"   # azul  — full_finetune
COL_V1   = "#E86A2B"   # laranja — xai_guided_v1
COL_V2   = "#2CA02C"   # verde  — xai_guided_v2
COL_HEAD = "#7B68EE"   # lilás  — head_only


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def errors_from_json(data: dict) -> list[dict]:
    """Retorna lista de erros com overlap_score."""
    return [e for e in data.get("errors", []) if "overlap_score" in e]


def fp_files_for_model(gradcam_dir: Path) -> dict[str, Path]:
    """Retorna {filename_stem: path} para os FPs de um diretório Grad-CAM."""
    fp_dir = gradcam_dir / "FP"
    if not fp_dir.is_dir():
        return {}
    return {p.stem.split("_true")[0]: p for p in sorted(fp_dir.glob("*.png"))}


# ─────────────────────────────────────────────────────────────────────────────
# Figura 1 — Grade comparativa de FPs corrigidos
# ─────────────────────────────────────────────────────────────────────────────
def figure_grade_3models(max_rows: int = 8, v2_available: bool = True) -> None:
    """
    Para cada FP do full_finetune que foi corrigido no xai_v1 (virou TN/TP),
    mostra os overlays Grad-CAM++ lado a lado dos 3 modelos.
    """
    # Carrega mapeamento de FPs por modelo
    fp_ft  = fp_files_for_model(RESULTS_DIR / "gradcam")
    fp_v1  = fp_files_for_model(RESULTS_DIR / "gradcam_xai")
    fp_v2  = fp_files_for_model(RESULTS_DIR / "gradcam_xai_v2") if v2_available else {}

    # Stems presentes em full_finetune mas ausentes em xai_v1 = corrigidos
    corrected = sorted(set(fp_ft.keys()) - set(fp_v1.keys()))[:max_rows]
    if not corrected:
        # Fallback: primeiros FPs do full_finetune
        corrected = sorted(fp_ft.keys())[:max_rows]

    n_models = 3 if v2_available else 2
    labels_col = ["Full Finetune", "XAI-Guided V1", "XAI-Guided V2"][:n_models]
    colors_col = [COL_FT, COL_V1, COL_V2][:n_models]
    sources = [fp_ft, fp_v1, fp_v2][:n_models]

    n_rows = len(corrected)
    if n_rows == 0:
        print("  [grade] Nenhuma imagem encontrada, pulando figura 1.")
        return

    fig, axes = plt.subplots(n_rows, n_models,
                             figsize=(4.5 * n_models, 3.5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor("#0d0d0d")

    for row, stem in enumerate(corrected):
        for col, (src, lbl, clr) in enumerate(zip(sources, labels_col, colors_col)):
            ax = axes[row][col]
            ax.axis("off")
            ax.set_facecolor("#1a1a1a")
            path = src.get(stem)
            if path and path.is_file():
                img = np.asarray(Image.open(path).convert("RGB"))
                ax.imshow(img)
                status = "FP" if stem in src else "Corrigido"
                ax.set_title(f"{lbl}\n{status}", fontsize=8,
                             color=clr, fontweight="bold", pad=3)
            else:
                ax.text(0.5, 0.5, f"{lbl}\n(Corrigido)", ha="center", va="center",
                        fontsize=9, color=clr, fontweight="bold",
                        transform=ax.transAxes)
                ax.set_title(f"{lbl}", fontsize=8, color=clr,
                             fontweight="bold", pad=3)

    fig.suptitle(
        "FPs Corrigidos pelo XAI-Guided — Comparação de Atenção Grad-CAM++",
        fontsize=13, fontweight="bold", color="white", y=1.002
    )
    plt.tight_layout(pad=0.4)
    out = FIGURES_DIR / "comparison_gradcam_3models.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvo: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 2 — Distribuição de overlap_score por modelo
# ─────────────────────────────────────────────────────────────────────────────
def figure_overlap_distribution(v2_available: bool = True) -> None:
    """Boxplot + jitter de overlap_score nos erros para cada modelo."""
    data_ft = load_json(RESULTS_DIR / "error_analysis.json")
    data_v1 = load_json(RESULTS_DIR / "error_analysis_xai.json")
    data_v2 = load_json(RESULTS_DIR / "error_analysis_xai_v2.json") if v2_available else None

    def get_scores(data):
        return [e["overlap_score"] for e in errors_from_json(data)]

    groups = [
        ("Full Finetune",    get_scores(data_ft),  COL_FT),
        ("XAI-Guided V1",   get_scores(data_v1),  COL_V1),
    ]
    if data_v2:
        groups.append(("XAI-Guided V2", get_scores(data_v2), COL_V2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    positions = list(range(1, len(groups) + 1))
    box_data  = [g[1] for g in groups]
    bp = ax.boxplot(box_data, positions=positions, widths=0.4,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    for patch, (_, _, clr) in zip(bp["boxes"], groups):
        patch.set_facecolor(clr)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for pos, (lbl, scores, clr) in zip(positions, groups):
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(pos + jitter, scores, color=clr, alpha=0.55, s=18, zorder=3)

    ax.axhline(0.20, linestyle="--", color="#888", linewidth=1, label="focused threshold (0,20)")
    ax.axhline(0.50, linestyle=":",  color="#555", linewidth=1, label="distracted threshold (0,50)")

    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_ylabel("overlap_score (fração fora da máscara pulmonar)", fontsize=11)
    ax.set_title("Distribuição do overlap_score nos Erros por Modelo", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    # Anotação com n= e mediana
    for pos, (lbl, scores, clr) in zip(positions, groups):
        if scores:
            med = np.median(scores)
            ax.text(pos, ax.get_ylim()[0] - 0.02,
                    f"n={len(scores)}\nmed={med:.3f}",
                    ha="center", va="top", fontsize=8.5, color=clr, fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "overlap_score_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 3 — Tabela comparativa de métricas (4 modelos)
# ─────────────────────────────────────────────────────────────────────────────
def figure_metrics_table(v2_available: bool = True) -> None:
    """Tabela visual com gradiente de cores para cada métrica."""
    m_head = load_json(RESULTS_DIR / "metrics_head_only.json")["test"]
    m_ft   = load_json(RESULTS_DIR / "metrics_full_finetune.json")["test"]
    m_v1   = load_json(RESULTS_DIR / "metrics_xai_guided.json")["test"]
    m_v2   = load_json(RESULTS_DIR / "metrics_xai_guided_v2.json")["test"] if v2_available else None

    cm_head = m_head["confusion_matrix"]
    cm_ft   = m_ft["confusion_matrix"]
    cm_v1   = m_v1["confusion_matrix"]

    def row(m, cm):
        return [
            f"{m['accuracy_percent']:.2f}%",
            f"{m['auc_roc']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            str(cm[0][1]),   # FP
            str(cm[1][0]),   # FN
        ]

    rows = [
        ("Head Only",      COL_HEAD, row(m_head, cm_head)),
        ("Full Finetune",  COL_FT,   row(m_ft,   cm_ft)),
        ("XAI-Guided V1", COL_V1,   row(m_v1,   cm_v1)),
    ]
    if m_v2:
        cm_v2 = m_v2["confusion_matrix"]
        rows.append(("XAI-Guided V2", COL_V2, row(m_v2, cm_v2)))

    cols = ["Acurácia", "AUC-ROC", "F1", "Precisão", "Recall", "FP", "FN"]

    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.7 * len(rows)))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Cabeçalho
    header_cols = ["Modelo"] + cols
    table_data  = [[lbl] + data for lbl, _, data in rows]

    tbl = ax.table(
        cellText=table_data,
        colLabels=header_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Estiliza cabeçalho
    for j in range(len(header_cols)):
        cell = tbl[0, j]
        cell.set_facecolor("#103A6E")
        cell.set_text_props(color="white", fontweight="bold")

    # Estiliza linhas de dados
    for i, (lbl, clr, _) in enumerate(rows):
        for j in range(len(header_cols)):
            cell = tbl[i + 1, j]
            if j == 0:
                cell.set_facecolor(clr)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

    # Destaca melhor valor de cada coluna numérica (cols 1-5 = maior melhor, 6-7 = menor melhor)
    higher_better = [1, 2, 3, 4, 5]
    lower_better  = [6, 7]
    for col_idx in range(1, len(header_cols)):
        vals = []
        for row_idx in range(1, len(rows) + 1):
            try:
                txt = tbl[row_idx, col_idx].get_text().get_text()
                vals.append(float(txt.replace("%", "")))
            except Exception:
                vals.append(None)
        if all(v is None for v in vals):
            continue
        if (col_idx) in higher_better:
            best = max(v for v in vals if v is not None)
        else:
            best = min(v for v in vals if v is not None)
        for row_idx, v in enumerate(vals, start=1):
            if v == best:
                tbl[row_idx, col_idx].set_facecolor("#D5FFD5")
                tbl[row_idx, col_idx].set_text_props(fontweight="bold")

    ax.set_title("Comparação de Métricas — Conjunto de Teste (624 imagens)",
                 fontsize=13, fontweight="bold", pad=16)

    plt.tight_layout()
    out = FIGURES_DIR / "metrics_4models.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 4 — Evolução das métricas ao longo dos modelos (linha)
# ─────────────────────────────────────────────────────────────────────────────
def figure_metrics_evolution(v2_available: bool = True) -> None:
    """Gráfico de linhas mostrando a evolução de AUC, F1, FP e FN."""
    m_head = load_json(RESULTS_DIR / "metrics_head_only.json")["test"]
    m_ft   = load_json(RESULTS_DIR / "metrics_full_finetune.json")["test"]
    m_v1   = load_json(RESULTS_DIR / "metrics_xai_guided.json")["test"]

    models = ["Head Only", "Full Finetune", "XAI-Guided V1"]
    auc  = [m_head["auc_roc"], m_ft["auc_roc"], m_v1["auc_roc"]]
    f1   = [m_head["f1"],      m_ft["f1"],      m_v1["f1"]]
    fps  = [m_head["confusion_matrix"][0][1], m_ft["confusion_matrix"][0][1],
            m_v1["confusion_matrix"][0][1]]
    fns  = [m_head["confusion_matrix"][1][0], m_ft["confusion_matrix"][1][0],
            m_v1["confusion_matrix"][1][0]]
    acc  = [m_head["accuracy_percent"], m_ft["accuracy_percent"], m_v1["accuracy_percent"]]

    if v2_available:
        m_v2 = load_json(RESULTS_DIR / "metrics_xai_guided_v2.json")["test"]
        models.append("XAI-Guided V2")
        auc.append(m_v2["auc_roc"])
        f1.append(m_v2["f1"])
        fps.append(m_v2["confusion_matrix"][0][1])
        fns.append(m_v2["confusion_matrix"][1][0])
        acc.append(m_v2["accuracy_percent"])

    x = list(range(len(models)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("white")

    # AUC e F1
    ax = axes[0]
    ax.plot(x, auc, "o-", color=COL_FT,   linewidth=2, markersize=8, label="AUC-ROC")
    ax.plot(x, f1,  "s--", color=COL_V1,  linewidth=2, markersize=8, label="F1")
    ax.plot(x, [v/100 for v in acc], "^:", color=COL_V2, linewidth=2, markersize=8, label="Acurácia/100")
    for i, (a, f) in enumerate(zip(auc, f1)):
        ax.annotate(f"{a:.4f}", (x[i], a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=COL_FT)
        ax.annotate(f"{f:.4f}", (x[i], f), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=COL_V1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.set_title("AUC-ROC, F1 e Acurácia", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.82, 1.0)
    ax.grid(True, alpha=0.3)

    # FP
    ax = axes[1]
    bars = ax.bar(x, fps, color=[COL_HEAD, COL_FT, COL_V1, COL_V2][:len(x)], alpha=0.75, width=0.5)
    for bar, val in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.set_title("Falsos Positivos (FP) — menor é melhor", fontweight="bold")
    ax.set_ylabel("Nº de FP")
    ax.grid(True, axis="y", alpha=0.3)

    # FN
    ax = axes[2]
    bars = ax.bar(x, fns, color=[COL_HEAD, COL_FT, COL_V1, COL_V2][:len(x)], alpha=0.75, width=0.5)
    for bar, val in zip(bars, fns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.set_title("Falsos Negativos (FN) — menor é melhor", fontweight="bold")
    ax.set_ylabel("Nº de FN")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Evolução das Métricas ao Longo das Estratégias",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "metrics_evolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Gera figuras comparativas entre modelos")
    parser.add_argument("--max-grid", type=int, default=6,
                        help="Máximo de linhas na grade comparativa (default: 6)")
    args = parser.parse_args()

    v2_available = (
        (RESULTS_DIR / "metrics_xai_guided_v2.json").is_file() and
        (RESULTS_DIR / "error_analysis_xai_v2.json").is_file()
    )
    if not v2_available:
        print("  [aviso] metrics_xai_guided_v2.json ou error_analysis_xai_v2.json não encontrado — "
              "figuras de V2 serão omitidas.")

    print("Gerando figuras comparativas...")

    print("  1/4 — Grade comparativa de FPs corrigidos...")
    figure_grade_3models(max_rows=args.max_grid, v2_available=v2_available)

    print("  2/4 — Distribuição de overlap_score...")
    figure_overlap_distribution(v2_available=v2_available)

    print("  3/4 — Tabela de métricas (4 modelos)...")
    figure_metrics_table(v2_available=v2_available)

    print("  4/4 — Evolução das métricas...")
    figure_metrics_evolution(v2_available=v2_available)

    print("\nConcluído. Figuras salvas em figures/")


if __name__ == "__main__":
    main()
