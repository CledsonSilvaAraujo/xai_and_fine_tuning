#!/usr/bin/env python3
"""
Figura principal: arquitetura ResNet-50 + exemplos GradCAM++ (corretos e erros).

Gerada com matplotlib apenas — sem dependências extras.
Saída: figures/network_gradcam_analysis.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# Diagrama ResNet-50
# ─────────────────────────────────────────────────────────────────────────────

def draw_resnet_diagram(ax: plt.Axes) -> None:
    """Desenha blocos empilhados representando a ResNet-50."""
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 12.5)
    ax.axis("off")
    ax.set_title(
        "Arquitetura ResNet-50\n(camada alvo do GradCAM++)",
        fontsize=9, fontweight="bold", pad=6
    )

    # (y_bottom, height, label, cor_hex)
    blocks = [
        (0.2,  0.75, "FC  (2 classes) + Softmax",             "#d62728"),
        (1.1,  0.55, "AvgPool  Global",                        "#ff7f0e"),
        (1.8,  1.40, "Layer 4  ×3 blocos  [512→2048]",         "#e377c2"),  # alvo
        (3.35, 1.20, "Layer 3  ×6 blocos  [256→1024]",         "#17becf"),
        (4.70, 1.00, "Layer 2  ×4 blocos  [128→512]",          "#bcbd22"),
        (5.85, 0.85, "Layer 1  ×3 blocos  [64→256]",           "#2ca02c"),
        (6.85, 0.75, "MaxPool  3×3, s=2",                       "#9467bd"),
        (7.75, 0.70, "BN + ReLU",                               "#8c564b"),
        (8.60, 0.85, "Conv1  7×7, s=2  (64 filtros)",           "#1f77b4"),
        (9.60, 0.75, "Entrada  224 × 224 × 3",                  "#aec7e8"),
    ]

    for y, h, label, color in blocks:
        rect = mpatches.FancyBboxPatch(
            (0.30, y), 3.40, h,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor="black", linewidth=0.7, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(
            2.0, y + h / 2, label,
            ha="center", va="center", fontsize=7.2,
            color="white", fontweight="bold",
        )

    # Setas de conexão entre blocos
    for i in range(len(blocks) - 1):
        y_top = blocks[i][0] + blocks[i][1]
        y_bot = blocks[i + 1][0] + blocks[i + 1][1]
        ax.annotate(
            "", xy=(2.0, y_top), xytext=(2.0, y_bot - 0.02),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        )

    # Destaque tracejado em Layer4 (onde o GradCAM++ se conecta)
    highlight = mpatches.FancyBboxPatch(
        (0.22, 1.78), 3.56, 1.42,
        boxstyle="round,pad=0.08",
        facecolor="none", edgecolor="#d62728", linewidth=2.2, linestyle="--",
    )
    ax.add_patch(highlight)
    ax.annotate(
        "GradCAM++\ncaptura aqui",
        xy=(3.78, 2.49), fontsize=7.5, color="#d62728", fontweight="bold",
        ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="#d62728", boxstyle="round,pad=0.2", lw=1.2),
    )
    ax.annotate(
        "", xy=(3.66, 2.49), xytext=(3.88, 2.49),
        arrowprops=dict(arrowstyle="<-", color="#d62728", lw=1.4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Painel de imagem com overlay GradCAM++
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_panel(
    ax: plt.Axes,
    overlay_path: str,
    title: str,
    is_error: bool,
    overlap_score: float = 0.0,
    conf: float = 0.0,
) -> None:
    """Mostra overlay GradCAM++ com anotações de status."""
    try:
        img = np.array(Image.open(overlay_path).convert("RGB"))
    except Exception:
        ax.text(0.5, 0.5, "imagem\nnão encontrada",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axis("off")
        return

    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")

    fg_color = "#d62728" if is_error else "#2ca02c"
    prefix = "✗ ERRO" if is_error else "✓ CORRETO"
    ax.set_title(f"{prefix} — {title}", fontsize=8, color=fg_color,
                 fontweight="bold", pad=3)

    # Confiança da predição
    ax.text(
        0.5, 1.01, f"confiança: {conf:.1%}",
        transform=ax.transAxes, fontsize=7,
        ha="center", va="bottom", color="#333333",
    )

    # Barra de atenção fora dos pulmões (só para erros)
    if is_error and overlap_score > 0:
        ax.text(
            0.5, 0.03,
            f"atenção fora dos pulmões: {overlap_score:.0%}",
            transform=ax.transAxes, fontsize=7,
            ha="center", va="bottom", color="white",
            bbox=dict(facecolor="#d62728", alpha=0.80,
                      boxstyle="round,pad=0.25", edgecolor="none"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Legenda do heatmap JET
# ─────────────────────────────────────────────────────────────────────────────

def draw_colorbar_legend(fig: plt.Figure, left: float, bottom: float,
                          width: float, height: float) -> None:
    """Barra de cores manual simulando o colormap JET."""
    ax = fig.add_axes([left, bottom, width, height])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="jet", extent=[0, 1, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Baixa\nativação", "Média", "Alta ativação\n(GradCAM++)"],
                       fontsize=7.5)
    ax.set_title("Mapa de calor GradCAM++", fontsize=8, pad=3)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Figura principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    error_json = ROOT / "results" / "error_analysis.json"
    if not error_json.is_file():
        raise FileNotFoundError(f"Não encontrado: {error_json}\n"
                                "Execute gradcam_error_analysis.py primeiro.")

    data = json.loads(error_json.read_text(encoding="utf-8"))
    preds = data.get("predictions", [])
    errors = data.get("errors", [])

    # Seleciona exemplos representativos
    tp = next(
        iter(sorted([p for p in preds if p["bucket"] == "TP"],
                    key=lambda p: p["pred_confidence"], reverse=True)),
        None,
    )
    tn = next(
        iter(sorted([p for p in preds if p["bucket"] == "TN"],
                    key=lambda p: p["pred_confidence"], reverse=True)),
        None,
    )
    # FP e FN com maior overlap (atenção mais equivocada)
    fp = next(
        iter(sorted([e for e in errors if e["bucket"] == "FP"],
                    key=lambda e: e["overlap_score"], reverse=True)),
        None,
    )
    fn = next(
        iter(sorted([e for e in errors if e["bucket"] == "FN"],
                    key=lambda e: e["overlap_score"], reverse=True)),
        None,
    )
    # fallback: FP com menor confiança se não houver FN
    fn_or_fp2 = fn or next(
        iter(sorted([e for e in errors if e["bucket"] == "FP"],
                    key=lambda e: e["overlap_score"])),
        None,
    )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle(
        "ResNet-50 com GradCAM++: ativações e análise de erros  "
        "(Chest X-Ray — Pneumonia vs Normal)",
        fontsize=11, fontweight="bold", y=0.99, color="#111111",
    )

    # GridSpec: coluna esquerda (diagrama) | 2×2 imagens
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.15, 1, 1],
        hspace=0.50, wspace=0.22,
        left=0.03, right=0.97,
        top=0.92, bottom=0.10,
    )

    # Diagrama da rede (ocupa as 2 linhas)
    ax_arch = fig.add_subplot(gs[:, 0])
    ax_arch.set_facecolor("white")
    draw_resnet_diagram(ax_arch)

    # Painéis de imagem
    panels = [
        (gs[0, 1], tp,         "TP — Pneumonia detectada",   False),
        (gs[0, 2], tn,         "TN — Normal descartado",     False),
        (gs[1, 1], fp,         "FP — Normal→Pneumonia",      True),
        (gs[1, 2], fn_or_fp2,  "FN — Pneumonia→Normal",      True),
    ]

    for spec, item, title, is_error in panels:
        ax = fig.add_subplot(spec)
        if item:
            plot_gradcam_panel(
                ax,
                overlay_path=item["overlay_file"],
                title=title,
                is_error=is_error,
                overlap_score=float(item.get("overlap_score", 0)),
                conf=float(item.get("pred_confidence", 0)),
            )
        else:
            ax.text(0.5, 0.5, "sem exemplo\ndisponível",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.axis("off")

    # Barra de cores (legenda do heatmap)
    draw_colorbar_legend(fig, left=0.38, bottom=0.025, width=0.57, height=0.038)

    out_path = ROOT / "figures" / "network_gradcam_analysis.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Figura salva em: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
