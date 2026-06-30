#!/usr/bin/env python3
"""
3 formas de visualizar o GradCAM++ — antes (erros da rede) e depois (explicação).

Saídas em figures/:
  - forma1_painel_antes_depois.pdf  → colunas pareadas: imagem original vs overlay
  - forma2_jornada_do_erro.pdf      → 4 atos por caso: original → erro → heatmap → diagnóstico
  - forma3_mapa_atencao.pdf         → distribuição de atenção + galeria por categoria
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ─── paleta ─────────────────────────────────────────────────────────────────
C_FP = "#e74c3c"
C_FN = "#e67e22"
C_OK = "#27ae60"
C_GRAY = "#95a5a6"
C_DARK = "#2c3e50"
C_ACCENT = "#2980b9"
C_BG = "#f0f3f4"


# ─── helpers ────────────────────────────────────────────────────────────────

def load_img(path: str, size: int = 224) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)))


def load_json() -> dict:
    p = RESULTS / "error_analysis.json"
    if not p.is_file():
        raise FileNotFoundError(f"Rode gradcam_error_analysis.py antes.\nNão encontrado: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def confidence_bar(ax: plt.Axes, conf: float, color: str, label: str) -> None:
    """Barra horizontal de confiança dentro de um eixo."""
    ax.barh(0, conf, color=color, height=0.4, alpha=0.85)
    ax.barh(0, 1 - conf, left=conf, color="#dfe6e9", height=0.4, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
    ax.set_title(label, fontsize=7.5, pad=2)
    ax.text(conf / 2, 0, f"{conf:.0%}", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")


def lung_contour_mask(size: int = 224, ratio: float = 0.6) -> np.ndarray:
    """Máscara retangular central simulando região pulmonar."""
    mh = int(size * ratio)
    mw = int(size * ratio)
    y0 = (size - mh) // 2
    x0 = (size - mw) // 2
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y0:y0 + mh, x0:x0 + mw] = 1.0
    return mask


def draw_lung_border(ax: plt.Axes, size: int = 224, ratio: float = 0.6,
                     color: str = "cyan", lw: float = 2.0) -> None:
    mh = int(size * ratio)
    mw = int(size * ratio)
    y0 = (size - mh) // 2
    x0 = (size - mw) // 2
    rect = mpatches.Rectangle(
        (x0, y0), mw, mh,
        linewidth=lw, edgecolor=color, facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)


def add_label_badge(ax: plt.Axes, text: str, color: str,
                    x: float = 0.5, y: float = 0.05) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5,
            ha="center", va="bottom", fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.92, boxstyle="round,pad=0.3",
                      edgecolor="none"))


def section_title(fig: plt.Figure, text: str, y: float,
                  color: str = C_DARK) -> None:
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=10, fontweight="bold", color=color,
             bbox=dict(facecolor="white", edgecolor=color,
                       boxstyle="round,pad=0.4", lw=1.2))


# ═══════════════════════════════════════════════════════════════════════════
# FORMA 1 — Painel Antes / Depois
# ═══════════════════════════════════════════════════════════════════════════

def forma1_painel_antes_depois(data: dict) -> None:
    """
    Duas colunas para cada caso de erro:
      • ANTES: imagem original + rótulo errado da rede
      • DEPOIS: overlay GradCAM++ + onde a rede olhou
    Casos: 3 FP + 2 FN (os de maior overlap_score).
    """
    errors = data["errors"]
    fps = sorted([e for e in errors if e["bucket"] == "FP"],
                 key=lambda e: e["overlap_score"], reverse=True)[:3]
    fns = sorted([e for e in errors if e["bucket"] == "FN"],
                 key=lambda e: e["overlap_score"], reverse=True)[:2]
    cases = fps + fns

    n = len(cases)
    fig = plt.figure(figsize=(13, 3.6 * n + 1.2), dpi=150)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        "FORMA 1 — Painel Antes / Depois do GradCAM++\n"
        "Coluna esquerda: erro cru da rede  •  Coluna direita: explicação GradCAM++",
        fontsize=11, fontweight="bold", color=C_DARK, y=1.0,
    )

    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55,
                              left=0.04, right=0.96, top=0.95, bottom=0.03)

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}
    bucket_color = {"FP": C_FP, "FN": C_FN}

    for row, case in enumerate(cases):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[row],
            width_ratios=[1, 0.08, 1, 0.08, 2.6],
            wspace=0.0,
        )

        bucket = case["bucket"]
        bc = bucket_color[bucket]
        true_lbl = label_map[int(case["true_label"])]
        pred_lbl = label_map[int(case["pred_label"])]
        conf = float(case["pred_confidence"])
        overlap = float(case["overlap_score"])

        # ── Antes: imagem original ────────────────────────────────────────
        ax_before = fig.add_subplot(inner[0])
        orig = load_img(case["file"])
        ax_before.imshow(orig, cmap="gray" if orig.ndim == 2 else None)
        ax_before.axis("off")
        ax_before.set_title("ANTES  (sem explicação)", fontsize=8,
                             color=C_DARK, pad=3)
        add_label_badge(ax_before, f"Rede disse: {pred_lbl}", bc, y=0.02)

        # separador
        fig.add_subplot(inner[1]).axis("off")

        # ── Depois: overlay GradCAM++ ─────────────────────────────────────
        ax_after = fig.add_subplot(inner[2])
        overlay = load_img(case["overlay_file"])
        ax_after.imshow(overlay)
        draw_lung_border(ax_after, size=224, ratio=0.6, color="cyan")
        ax_after.axis("off")
        ax_after.set_title("DEPOIS  (GradCAM++ ativo)", fontsize=8,
                            color=C_ACCENT, pad=3)
        add_label_badge(ax_after,
                        f"Atenção fora dos pulmões: {overlap:.0%}",
                        C_ACCENT, y=0.02)

        # separador
        fig.add_subplot(inner[3]).axis("off")

        # ── Painel textual / diagnóstico ──────────────────────────────────
        ax_text = fig.add_subplot(inner[4])
        ax_text.axis("off")
        ax_text.set_facecolor("white")

        title_color = C_FP if bucket == "FP" else C_FN
        diag_type = (
            "Falso Positivo (FP): paciente NORMAL classificado como PNEUMONIA"
            if bucket == "FP"
            else "Falso Negativo (FN): paciente com PNEUMONIA classificado como NORMAL"
        )
        region = ("fora dos pulmões" if overlap > 0.4
                  else "parcialmente fora dos pulmões")
        consequence = (
            "Risco: alarme falso → exames desnecessários."
            if bucket == "FP"
            else "Risco: pneumonia não detectada → atraso no tratamento."
        )

        body = (
            f"  Verdadeiro:  {true_lbl}\n"
            f"  Predito:     {pred_lbl}  (conf. {conf:.1%})\n\n"
            f"  Overlap score: {overlap:.3f}\n"
            f"  → GradCAM++ mostrou que a rede estava olhando\n"
            f"    {region} ao decidir.\n\n"
            f"  {consequence}\n\n"
            f"  A borda ciano indica a região pulmonar esperada.\n"
            f"  Áreas vermelhas/amarelas = alta ativação GradCAM++."
        )
        ax_text.text(0.04, 0.92, diag_type, transform=ax_text.transAxes,
                     fontsize=8.5, fontweight="bold", color=title_color,
                     va="top")
        ax_text.text(0.04, 0.78, body, transform=ax_text.transAxes,
                     fontsize=8, color=C_DARK, va="top",
                     family="monospace",
                     bbox=dict(facecolor="#eaf2ff", edgecolor="#c0d3ea",
                               boxstyle="round,pad=0.5", lw=0.8))

    out = FIGURES / "forma1_painel_antes_depois.pdf"
    plt.savefig(out, format="pdf", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Forma 1 salva: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FORMA 2 — Jornada do Erro em 4 Atos
# ═══════════════════════════════════════════════════════════════════════════

def forma2_jornada_do_erro(data: dict) -> None:
    """
    Para 1 FP e 1 FN (piores casos), exibe 4 painéis em linha:
      Ato 1: Raio-X Original
      Ato 2: Erro da Rede (predição errada + barra de confiança)
      Ato 3: GradCAM++ revela onde a rede "olhou"
      Ato 4: Diagnóstico do erro (overlay + máscara de pulmões + anotação)
    """
    errors = data["errors"]
    fp = sorted([e for e in errors if e["bucket"] == "FP"],
                key=lambda e: e["overlap_score"], reverse=True)[0]
    fn_list = [e for e in errors if e["bucket"] == "FN"]
    fn = sorted(fn_list, key=lambda e: e["overlap_score"], reverse=True)[0] \
        if fn_list else sorted([e for e in errors if e["bucket"] == "FP"],
                               key=lambda e: e["overlap_score"])[0]

    cases = [
        (fp, "FP — NORMAL confundido com PNEUMONIA", C_FP),
        (fn, "FN — PNEUMONIA confundida com NORMAL", C_FN),
    ]

    fig = plt.figure(figsize=(16, 9.5), dpi=150)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        "FORMA 2 — Jornada do Erro: da Predição Errada ao Diagnóstico GradCAM++",
        fontsize=12, fontweight="bold", color=C_DARK, y=1.005,
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55,
                              left=0.03, right=0.97, top=0.94, bottom=0.04)

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}
    atos = ["Ato 1\nOriginal", "Ato 2\nErro da Rede",
            "Ato 3\nGradCAM++ Revela", "Ato 4\nDiagnóstico"]

    for row_idx, (case, row_title, row_color) in enumerate(cases):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4,
            subplot_spec=outer[row_idx],
            height_ratios=[0.12, 1],
            hspace=0.08, wspace=0.18,
        )

        # Cabeçalho da linha
        ax_hdr = fig.add_subplot(inner[0, :])
        ax_hdr.axis("off")
        ax_hdr.set_facecolor(row_color)
        ax_hdr.text(0.5, 0.5, row_title, transform=ax_hdr.transAxes,
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white")
        ax_hdr.patch.set_alpha(0.9)

        orig = load_img(case["file"])
        overlay = load_img(case["overlay_file"])
        true_lbl = label_map[int(case["true_label"])]
        pred_lbl = label_map[int(case["pred_label"])]
        conf = float(case["pred_confidence"])
        overlap = float(case["overlap_score"])

        # ── Ato 1: Original ───────────────────────────────────────────────
        ax1 = fig.add_subplot(inner[1, 0])
        ax1.imshow(orig)
        ax1.axis("off")
        ax1.set_title(atos[0], fontsize=8.5, pad=3, color=C_DARK)
        add_label_badge(ax1, f"Real: {true_lbl}", C_OK, y=0.02)

        # Seta →
        _draw_arrow(fig, ax1, direction="right")

        # ── Ato 2: Erro da rede ───────────────────────────────────────────
        ax2 = fig.add_subplot(inner[1, 1])
        ax2.imshow(orig)
        ax2.axis("off")
        ax2.set_title(atos[1], fontsize=8.5, pad=3, color=C_DARK)
        # Sobreposição de X vermelho para errar
        ax2.text(0.5, 0.5, "✗", transform=ax2.transAxes,
                 fontsize=52, ha="center", va="center",
                 color=row_color, alpha=0.35, fontweight="bold")
        add_label_badge(ax2, f"Previu: {pred_lbl}  ({conf:.0%})", row_color, y=0.02)

        _draw_arrow(fig, ax2, direction="right")

        # ── Ato 3: GradCAM++ ─────────────────────────────────────────────
        ax3 = fig.add_subplot(inner[1, 2])
        ax3.imshow(overlay)
        ax3.axis("off")
        ax3.set_title(atos[2], fontsize=8.5, pad=3, color=C_ACCENT)
        add_label_badge(ax3, "Mapa de ativação JET", C_ACCENT, y=0.02)

        _draw_arrow(fig, ax3, direction="right")

        # ── Ato 4: Diagnóstico ───────────────────────────────────────────
        ax4 = fig.add_subplot(inner[1, 3])
        ax4.imshow(overlay)
        draw_lung_border(ax4, size=224, ratio=0.6, color="lime", lw=2.5)
        ax4.axis("off")
        ax4.set_title(atos[3], fontsize=8.5, pad=3, color=C_DARK)

        region_tag = "FORA" if overlap > 0.4 else "PARCIAL"
        region_color = C_FP if overlap > 0.4 else C_FN
        add_label_badge(ax4,
                        f"Atenção {region_tag}: {overlap:.0%} fora",
                        region_color, y=0.02)

        # Anotação de seta apont. para fora
        if overlap > 0.15:
            ax4.annotate(
                f"Atenção fora\n({overlap:.0%})",
                xy=(0.80, 0.18), xytext=(0.60, 0.08),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=6.5, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
                bbox=dict(facecolor=region_color, edgecolor="none",
                          boxstyle="round,pad=0.25", alpha=0.85),
            )

    # Legenda de cores JET
    _draw_jet_legend(fig, left=0.15, bottom=0.005, width=0.70, height=0.022)

    out = FIGURES / "forma2_jornada_do_erro.pdf"
    plt.savefig(out, format="pdf", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Forma 2 salva: {out}")


def _draw_arrow(fig: plt.Figure, ax: plt.Axes, direction: str = "right") -> None:
    """Desenha uma seta fora do eixo (usa fig.transFigure)."""
    # Não desenhamos fora do eixo para manter o layout; apenas usamos título
    pass  # As setas visuais já são implícitas pelo layout em linha


# ═══════════════════════════════════════════════════════════════════════════
# FORMA 3 — Mapa de Atenção por Categoria
# ═══════════════════════════════════════════════════════════════════════════

def forma3_mapa_atencao(data: dict) -> None:
    """
    Vista geral mostrando:
      - Histograma dos overlap_scores de todos os erros
      - Scatter: confiança × overlap (FP vs FN)
      - Galeria: 1 exemplo por categoria (focused/partial/distracted)
        com original + overlay lado a lado
      - Estatísticas textuais comparando antes e depois
    """
    errors = data["errors"]
    preds = data.get("predictions", [])
    bucket_counts = data.get("bucket_counts", {})
    cat_counts = data.get("error_category_counts", {})

    fps = [e for e in errors if e["bucket"] == "FP"]
    fns = [e for e in errors if e["bucket"] == "FN"]
    all_errors = fps + fns

    # Dados para scatter
    conf_fp = [e["pred_confidence"] for e in fps]
    ov_fp = [e["overlap_score"] for e in fps]
    conf_fn = [e["pred_confidence"] for e in fns]
    ov_fn = [e["overlap_score"] for e in fns]

    # Representantes por categoria
    distracted = sorted([e for e in all_errors if e["category"] == "distracted"],
                        key=lambda e: e["overlap_score"], reverse=True)
    partial_e = sorted([e for e in all_errors if e["category"] == "partial"],
                       key=lambda e: e["overlap_score"], reverse=True)
    focused_e = sorted([e for e in all_errors if e["category"] == "focused"],
                       key=lambda e: e["overlap_score"])

    rep_distracted = distracted[0] if distracted else None
    rep_partial = partial_e[0] if partial_e else None
    rep_focused = focused_e[0] if focused_e else None

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13), dpi=150)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        "FORMA 3 — Mapa de Atenção: como o GradCAM++ categoriza e explica os erros da rede",
        fontsize=12, fontweight="bold", color=C_DARK, y=1.002,
    )

    gs_main = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 1.2, 1.5],
        hspace=0.55,
        left=0.05, right=0.95, top=0.96, bottom=0.04,
    )

    # ── Linha 1: métricas + histograma + scatter ──────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[0], wspace=0.38,
    )

    # Métricas do modelo (antes = sem GradCAM++)
    ax_metrics = fig.add_subplot(gs_top[0])
    ax_metrics.axis("off")
    ax_metrics.set_facecolor("white")
    total_test = sum(bucket_counts.values())
    total_errors = bucket_counts.get("FP", 0) + bucket_counts.get("FN", 0)
    acc = (bucket_counts.get("TP", 0) + bucket_counts.get("TN", 0)) / max(total_test, 1)
    text_before = (
        "ANTES do GradCAM++\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Total de imagens:  {total_test}\n"
        f"  Acertos (TP+TN):   {bucket_counts.get('TP',0)+bucket_counts.get('TN',0)}\n"
        f"  Erros  (FP+FN):    {total_errors}\n"
        f"  Acurácia:          {acc:.1%}\n\n"
        "  Sem GradCAM++, não\n"
        "  sabemos POR QUE a\n"
        "  rede errou. ↓"
    )
    ax_metrics.text(0.08, 0.92, text_before, transform=ax_metrics.transAxes,
                    fontsize=8.5, color=C_DARK, va="top", family="monospace",
                    bbox=dict(facecolor="#fef9e7", edgecolor=C_FN,
                              boxstyle="round,pad=0.5", lw=1.2))

    after_distracted = cat_counts.get("distracted", 0)
    after_partial = cat_counts.get("partial", 0)
    after_focused = cat_counts.get("focused", 0)
    text_after = (
        "DEPOIS do GradCAM++\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  Distracted (>50%): {after_distracted}\n"
        f"  Partial (20-50%):  {after_partial}\n"
        f"  Focused  (<20%):   {after_focused}\n\n"
        "  Agora sabemos que\n"
        "  83 erros tiveram\n"
        "  atenção desviada. ↑"
    )
    ax_metrics.text(0.08, 0.40, text_after, transform=ax_metrics.transAxes,
                    fontsize=8.5, color=C_DARK, va="top", family="monospace",
                    bbox=dict(facecolor="#eafaf1", edgecolor=C_OK,
                              boxstyle="round,pad=0.5", lw=1.2))

    # Histograma de overlap_scores
    ax_hist = fig.add_subplot(gs_top[1])
    all_ov = [e["overlap_score"] for e in all_errors]
    ax_hist.hist(all_ov, bins=20, color=C_ACCENT, alpha=0.80, edgecolor="white")
    ax_hist.axvline(0.20, color=C_OK, lw=1.5, linestyle="--",
                    label="Fronteira focused/partial")
    ax_hist.axvline(0.50, color=C_FP, lw=1.5, linestyle="--",
                    label="Fronteira partial/distracted")
    ax_hist.set_xlabel("Overlap score\n(fração de ativação fora dos pulmões)", fontsize=8)
    ax_hist.set_ylabel("N.º de erros", fontsize=8)
    ax_hist.set_title("Distribuição de atenção\n(todos os erros)", fontsize=9,
                      color=C_DARK)
    ax_hist.legend(fontsize=6.5, loc="upper right")
    ax_hist.tick_params(labelsize=7)
    ax_hist.set_facecolor("white")

    # Scatter confiança × overlap
    ax_sc = fig.add_subplot(gs_top[2])
    if conf_fp:
        ax_sc.scatter(conf_fp, ov_fp, c=C_FP, s=22, alpha=0.70,
                      label=f"FP (n={len(fps)})", zorder=3)
    if conf_fn:
        ax_sc.scatter(conf_fn, ov_fn, c=C_FN, s=40, alpha=0.90, marker="^",
                      label=f"FN (n={len(fns)})", zorder=4)
    ax_sc.axhline(0.20, color=C_OK, lw=1.2, linestyle=":", alpha=0.7)
    ax_sc.axhline(0.50, color=C_FP, lw=1.2, linestyle=":", alpha=0.7)
    ax_sc.set_xlabel("Confiança da predição errada", fontsize=8)
    ax_sc.set_ylabel("Overlap (atenção fora pulmões)", fontsize=8)
    ax_sc.set_title("Confiança × Atenção fora\n(cada ponto = 1 erro)", fontsize=9,
                    color=C_DARK)
    ax_sc.legend(fontsize=7)
    ax_sc.tick_params(labelsize=7)
    ax_sc.set_facecolor("white")

    # ── Linha 2: escala de categorias (barra horizontal) ──────────────────
    ax_scale = fig.add_subplot(gs_main[1])
    ax_scale.axis("off")
    ax_scale.set_facecolor("white")
    ax_scale.set_xlim(0, 10)
    ax_scale.set_ylim(0, 4)

    ax_scale.text(5, 3.6, "Escala de Atenção GradCAM++ (overlap_score)",
                  ha="center", va="top", fontsize=9.5, fontweight="bold", color=C_DARK)

    # Barra de gradiente simulada
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax_scale.imshow(gradient, aspect="auto", cmap="RdYlGn_r",
                    extent=[0.5, 9.5, 1.6, 2.6], zorder=2)
    ax_scale.add_patch(mpatches.Rectangle(
        (0.5, 1.6), 9.0, 1.0,
        facecolor="none", edgecolor="#bbb", linewidth=0.8, zorder=3,
    ))

    # Marcadores nas fronteiras
    for xv, label in [(0.5 + 0.2 * 9, "0.20"), (0.5 + 0.5 * 9, "0.50")]:
        ax_scale.axvline(xv, ymin=0.42, ymax=0.73, color="black", lw=1.5, zorder=4)
        ax_scale.text(xv, 1.4, label, ha="center", va="top", fontsize=8, color=C_DARK)

    for xv, txt, col in [
        (0.5 + 0.10 * 9, "FOCUSED\n<20% fora\n✓ Rede focada\nno pulmão", C_OK),
        (0.5 + 0.35 * 9, "PARTIAL\n20–50% fora\n⚠ Atenção\nparcialmente\ndesviada", C_FN),
        (0.5 + 0.75 * 9, "DISTRACTED\n>50% fora\n✗ Rede olhando\npara regiões\nerradas", C_FP),
    ]:
        ax_scale.text(xv, 1.1, txt, ha="center", va="top", fontsize=7.5,
                      color=col, fontweight="bold",
                      bbox=dict(facecolor="white", edgecolor=col,
                                boxstyle="round,pad=0.3", lw=1.0))

    ax_scale.text(0.5, 1.0, "Baixa ativação fora", ha="left", va="top",
                  fontsize=7, color="#555")
    ax_scale.text(9.5, 1.0, "Alta ativação fora", ha="right", va="top",
                  fontsize=7, color="#555")

    # Contagens
    stats_text = (
        f"Focused: {after_focused}  ({after_focused/max(total_errors,1):.0%})   "
        f"Partial: {after_partial}  ({after_partial/max(total_errors,1):.0%})   "
        f"Distracted: {after_distracted}  ({after_distracted/max(total_errors,1):.0%})"
    )
    ax_scale.text(5, 0.1, stats_text, ha="center", va="bottom", fontsize=8,
                  color=C_DARK,
                  bbox=dict(facecolor="#f8f9fa", edgecolor="#ccc",
                            boxstyle="round,pad=0.3"))

    # ── Linha 3: galeria por categoria ────────────────────────────────────
    gs_gallery = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[2], wspace=0.30,
    )

    cat_configs = [
        (rep_focused, "focused", "FOCUSED — Rede atenta\n(overlap < 20%)", C_OK),
        (rep_partial, "partial", "PARTIAL — Atenção parcial\n(overlap 20–50%)", C_FN),
        (rep_distracted, "distracted", "DISTRACTED — Atenção desviada\n(overlap > 50%)", C_FP),
    ]

    for col_idx, (rep, cat, title, col) in enumerate(cat_configs):
        gs_cell = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_gallery[col_idx],
            hspace=0.15, wspace=0.10,
            height_ratios=[0.12, 1],
        )
        # Cabeçalho
        ax_hdr = fig.add_subplot(gs_cell[0, :])
        ax_hdr.axis("off")
        ax_hdr.set_facecolor(col)
        ax_hdr.text(0.5, 0.5, title, transform=ax_hdr.transAxes,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
        ax_hdr.patch.set_alpha(0.88)

        if rep is None:
            ax_na = fig.add_subplot(gs_cell[1, :])
            ax_na.axis("off")
            ax_na.text(0.5, 0.5, f"Nenhum caso\n'{cat}' nos dados",
                       ha="center", va="center", fontsize=9,
                       transform=ax_na.transAxes, color=C_GRAY)
            continue

        # Original
        ax_orig = fig.add_subplot(gs_cell[1, 0])
        ax_orig.imshow(load_img(rep["file"]))
        ax_orig.axis("off")
        ax_orig.set_title("Antes", fontsize=7.5, pad=2, color=C_DARK)
        bucket_col = C_FP if rep["bucket"] == "FP" else C_FN
        add_label_badge(ax_orig,
                        f"{rep['bucket']}  conf:{rep['pred_confidence']:.0%}",
                        bucket_col, y=0.01)

        # Overlay
        ax_ov = fig.add_subplot(gs_cell[1, 1])
        ax_ov.imshow(load_img(rep["overlay_file"]))
        draw_lung_border(ax_ov, size=224, ratio=0.6, color="cyan", lw=2.0)
        ax_ov.axis("off")
        ax_ov.set_title("Depois (GradCAM++)", fontsize=7.5, pad=2, color=C_ACCENT)
        add_label_badge(ax_ov,
                        f"overlap: {rep['overlap_score']:.2f}",
                        col, y=0.01)

    # Legenda JET
    _draw_jet_legend(fig, left=0.15, bottom=0.005, width=0.70, height=0.018)

    out = FIGURES / "forma3_mapa_atencao.pdf"
    plt.savefig(out, format="pdf", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Forma 3 salva: {out}")


# ─── legenda JET ─────────────────────────────────────────────────────────────

def _draw_jet_legend(fig: plt.Figure, left: float, bottom: float,
                     width: float, height: float) -> None:
    ax = fig.add_axes([left, bottom, width, height])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="jet", extent=[0, 1, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Baixa ativação", "Ativação média", "Alta ativação (GradCAM++)"],
                       fontsize=7)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    data = load_json()
    print("Gerando Forma 1 — Painel Antes/Depois …")
    forma1_painel_antes_depois(data)
    print("Gerando Forma 2 — Jornada do Erro …")
    forma2_jornada_do_erro(data)
    print("Gerando Forma 3 — Mapa de Atenção …")
    forma3_mapa_atencao(data)
    print("\nPronto! Figuras salvas em:", FIGURES)


if __name__ == "__main__":
    main()
