"""
src/reporting/charts.py
============================
Trách nhiệm DUY NHẤT: Vẽ biểu đồ so sánh từ reports dict.

Charts:
  1. anls_comparison.png   – bar chart ANLS theo model
  2. latency_comparison.png – bar chart latency mean theo model

Nếu matplotlib chưa cài → in warning và bỏ qua (không crash pipeline).
"""

from __future__ import annotations

import logging

from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive: safe trên server/SSH không có display
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Color palette ────────────────────────────────────────────────────
_COLORS = ["#4C8BF5", "#34A853", "#FBBC05", "#EA4335", "#9B59B6"]


def _require_matplotlib(func):
    """Decorator: bỏ qua hàm nếu matplotlib chưa cài."""
    def wrapper(*args, **kwargs):
        if not _HAS_MPL:
            logging.getLogger("poc1").info("[charts] matplotlib not installed → skip chart generation")
            logging.getLogger("poc1").info("[charts] Install: pip install matplotlib")
            return
        return func(*args, **kwargs)
    return wrapper


@_require_matplotlib
def plot_anls_comparison(reports: dict[str, dict], out_dir: str) -> None:
    """
    Bar chart: ANLS (%) theo model.

    Args:
        reports: {model_key: report dict}
        out_dir: thư mục lưu ảnh
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    models = sorted(reports.keys())
    scores = [reports[m]["overall"].get("anls", 0) for m in models]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    bars = ax.bar(models, scores, color=colors, width=0.5, edgecolor="white", linewidth=0.5)

    # Thêm nhãn trên mỗi cột
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("ANLS (%)", fontsize=12)
    ax.set_title("ANLS Score Comparison — POC-1", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(100, max(scores) * 1.2 + 5))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = p / "anls_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("poc1").info(f"[charts] ✔ Saved → {out_path}")


@_require_matplotlib
def plot_latency_comparison(reports: dict[str, dict], out_dir: str) -> None:
    """
    Bar chart: Latency mean (s) theo model.

    Args:
        reports: {model_key: report dict}
        out_dir: thư mục lưu ảnh
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    models = sorted(reports.keys())
    means  = [reports[m]["latency"].get("mean", 0) for m in models]
    p95s   = [reports[m]["latency"].get("p95",  0) for m in models]
    x      = range(len(models))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    ax.bar([xi - w/2 for xi in x], means, width=w, label="Mean",  color="#4C8BF5", alpha=0.85)
    ax.bar([xi + w/2 for xi in x], p95s,  width=w, label="P95",   color="#EA4335", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Latency (s)", fontsize=12)
    ax.set_title("Inference Latency Comparison — POC-1", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = p / "latency_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger("poc1").info(f"[charts] ✔ Saved → {out_path}")


def generate_all(reports: dict[str, dict], out_dir: str) -> None:
    """Gọi tất cả chart generators."""
    plot_anls_comparison(reports, out_dir)
    plot_latency_comparison(reports, out_dir)
