"""PNG charts for experiment reports (matplotlib, headless).

Colours follow a validated categorical palette in fixed order: a configuration always gets the
same colour, whatever subset is plotted. Grid and axes are recessive; marks are thin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.evaluation.metrics import auroc, risk_coverage_curve, roc_points  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONFIG_ORDER = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def config_color(name: str) -> str:
    """Fixed colour per configuration (custom names take the remaining slots)."""
    if name in CONFIG_ORDER:
        return SERIES[CONFIG_ORDER.index(name)]
    return SERIES[7]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _figure(n_panels: int, width_per: float = 5.0, height: float = 4.0) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, n_panels, figsize=(width_per * n_panels, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    axs = list(axes[0])
    for ax in axs:
        _style(ax)
    return fig, axs


def _ordered(configs) -> list[str]:  # type: ignore[no-untyped-def]
    cfgs = list(dict.fromkeys(configs))
    return sorted(cfgs, key=lambda c: (CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 99, c))


def hallucination_rate_chart(summary: pd.DataFrame, path: Path) -> Path:
    """Grouped bars: hallucination rate per configuration, one group per model, 95% CI whiskers."""
    models = list(dict.fromkeys(summary["model"]))
    configs = _ordered(summary["config"])
    fig, (ax,) = _figure(1, width_per=max(6.0, 1.6 * len(models) * len(configs) / 2), height=4.2)
    width = 0.8 / max(len(configs), 1)
    x = np.arange(len(models))
    for j, cfg in enumerate(configs):
        rows = [summary[(summary.model == m) & (summary.config == cfg)] for m in models]
        vals = np.array([r["hallucination_rate"].iloc[0] if len(r) else np.nan for r in rows])
        lo = np.array([r["hallucination_rate_ci_low"].iloc[0] if len(r) else np.nan for r in rows])
        hi = np.array([r["hallucination_rate_ci_high"].iloc[0] if len(r) else np.nan for r in rows])
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, vals, width=width * 0.9, color=config_color(cfg), label=cfg, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Hallucination rate (share of all questions)")
    ax.set_ylim(0, 1)
    ax.set_title("Hallucination rate by configuration (95% bootstrap CI)", fontsize=11, loc="left")
    ax.legend(frameon=False, ncol=min(len(configs), 6), fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def accuracy_coverage_chart(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    """Per model: risk-coverage curves (from the confidence score) and each config's operating point."""
    models = list(dict.fromkeys(summary["model"]))
    fig, axs = _figure(len(models))
    for ax, model in zip(axs, models):
        sub = results[results.model == model]
        for cfg in _ordered(sub["config"]):
            rows = sub[(sub.config == cfg) & sub.candidate_label.isin(["correct", "hallucinated"])]
            conf = rows["confidence"].astype(float)
            if conf.notna().sum() >= 2:
                cov, acc = risk_coverage_curve(conf, (rows["candidate_label"] == "correct").astype(float))
                ax.plot(cov, acc, color=config_color(cfg), linewidth=2, alpha=0.9, label=f"{cfg} curve")
            srow = summary[(summary.model == model) & (summary.config == cfg)]
            if len(srow) and not np.isnan(srow["selective_accuracy"].iloc[0]):
                ax.scatter(srow["coverage"], srow["selective_accuracy"], s=60, color=config_color(cfg),
                           edgecolor=SURFACE, linewidth=2, zorder=3, label=f"{cfg} operating point")
                ax.annotate(cfg, (srow["coverage"].iloc[0], srow["selective_accuracy"].iloc[0]),
                            textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK_2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Coverage (share of questions answered)")
        ax.set_ylabel("Accuracy on answered questions")
        ax.set_title(f"Accuracy vs coverage: {model}", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def gain_latency_chart(summary: pd.DataFrame, path: Path) -> Path:
    """Accuracy gain and hallucination reduction over C0 versus mean latency per question."""
    fig, axs = _figure(2)
    models = list(dict.fromkeys(summary["model"]))
    for m_i, model in enumerate(models):
        sub = summary[summary.model == model]
        base = sub[sub.config == "C0"]
        if base.empty:
            continue
        acc0 = base["accuracy"].iloc[0]
        hal0 = base["hallucination_rate"].iloc[0]
        for _, row in sub.iterrows():
            lat = row["latency_mean_s"]
            for ax, y in ((axs[0], row["accuracy"] - acc0), (axs[1], hal0 - row["hallucination_rate"])):
                ax.scatter(lat, y, s=64, color=config_color(row["config"]), marker=MARKERS[m_i % len(MARKERS)],
                           edgecolor=SURFACE, linewidth=2, zorder=3)
                ax.annotate(row["config"], (lat, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK_2)
    for ax, ylabel, title in (
        (axs[0], "Accuracy minus C0 accuracy", "Accuracy gain vs latency"),
        (axs[1], "C0 hallucination rate minus config rate", "Hallucination reduction vs latency"),
    ):
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Mean latency per question (s, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], linestyle="", color=MUTED, label=m)
               for i, m in enumerate(models)]
    axs[1].legend(handles=handles, frameon=False, fontsize=8, title="model (marker)", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def roc_chart(results: pd.DataFrame, path: Path) -> Path:
    """ROC curves of semantic entropy and SelfCheck as detectors of hallucinated initial answers."""
    models = list(dict.fromkeys(results["model"]))
    fig, axs = _figure(len(models))
    for ax, model in zip(axs, models):
        sub = results[(results.model == model) & results.initial_label.isin(["correct", "hallucinated"])]
        plotted = 0
        for cfg in _ordered(sub["config"]):
            rows = sub[sub.config == cfg]
            labels = (rows["initial_label"] == "hallucinated").astype(float)
            for col, name, style in (("semantic_entropy_norm", "SE", "-"), ("selfcheck_score", "SelfCheck", "--")):
                scores = rows[col].astype(float)
                if scores.notna().sum() < 2:
                    continue
                fpr, tpr = roc_points(scores, labels)
                if fpr.size == 0:
                    continue
                ax.plot(fpr, tpr, linestyle=style, linewidth=2, color=config_color(cfg),
                        label=f"{cfg} {name} (AUROC={auroc(scores, labels):.2f}, n={int(scores.notna().sum())})")
                plotted += 1
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"Detector ROC: {model}", fontsize=11, loc="left")
        if plotted:
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        else:
            ax.text(0.5, 0.5, "No ROC available\n(needs SE/SelfCheck configs\nand both label classes)",
                    ha="center", va="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
