"""Biểu đồ cho báo cáo (tuỳ chọn - cần matplotlib; nếu thiếu thì bỏ qua và báo một dòng)."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from .metrics import roc_auc, roc_curve


@functools.lru_cache(maxsize=None)
def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")  # vẽ ra file, không cần cửa sổ
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib - bỏ qua vẽ biểu đồ (pip install matplotlib).")
        return None
    return plt


def _save(fig, plt, path: str | Path) -> Path:
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> Path | None:
    """Loss và các độ đo theo epoch (train và val)."""
    names = [n for n in ("loss", "accuracy", "precision", "recall", "auc") if n in history]
    plt = _pyplot() if names else None
    if plt is None:
        return None
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        epochs = np.arange(1, len(history[name]) + 1)
        ax.plot(epochs, history[name], marker=".", label="train")
        if f"val_{name}" in history:
            ax.plot(epochs, history[f"val_{name}"], marker=".", label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, plt, path)


def plot_roc(labels, scores, path: str | Path) -> Path | None:
    """Đường ROC: TAR (= 1 − FRR) theo FAR khi ngưỡng thay đổi."""
    plt = _pyplot()
    if plt is None:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc(labels, scores):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Đoán ngẫu nhiên")
    ax.set_xlabel("FAR - tỉ lệ chấp nhận nhầm")
    ax.set_ylabel("TAR = 1 − FRR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return _save(fig, plt, path)


def plot_score_distribution(labels, scores, threshold: float, path: str | Path) -> Path | None:
    """Histogram điểm tương đồng của cặp cùng người / khác người và ngưỡng quyết định."""
    plt = _pyplot()
    if plt is None:
        return None
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Cặp cùng người")
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Cặp khác người")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Ngưỡng = {threshold:.2f}")
    ax.set_xlabel("Điểm tương đồng p(x₁, x₂)")
    ax.set_ylabel("Số cặp")
    ax.legend()
    return _save(fig, plt, path)
