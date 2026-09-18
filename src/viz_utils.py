"""viz_utils.py — shared plotting helpers so every notebook produces
visually consistent figures, saved under outputs/figures/ for reuse in the
summary report notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_utils import FIGURES_DIR

sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)

# Matplotlib's default font (DejaVu Sans) has no Thai glyphs, so Thai text in
# titles/labels/legends renders as tofu boxes. Tahoma ships with Windows and
# covers both Thai and Latin scripts cleanly — set it globally, after
# sns.set_theme() so seaborn's style reset doesn't clobber it.
plt.rcParams["font.family"] = "Tahoma"
plt.rcParams["axes.unicode_minus"] = False  # Tahoma's minus glyph can mis-render otherwise

PALETTE = "viridis"


def savefig(fig, name: str, dpi: int = 150):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure -> {path}")
    return path


def plot_class_distribution(counts_df, name="00_class_distribution.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=counts_df, x="class_id", y="count", order=counts_df["class_id"], palette=PALETTE, ax=ax)
    ax.set_xlabel("Class ID (folder name in round2/)")
    ax.set_ylabel("Number of images")
    ax.set_title("จำนวนภาพต่อคลาส เรียงจากมากไปน้อย (Class Imbalance)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    savefig(fig, name)
    plt.show()
    return fig


def plot_training_history(history_dict, name="03_training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_dict["accuracy"], label="Train")
    axes[0].plot(history_dict["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history_dict["loss"], label="Train")
    axes[1].plot(history_dict["val_loss"], label="Validation")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    savefig(fig, name)
    plt.show()
    return fig


def plot_confusion_matrix(cm, class_names, name="04_confusion_matrix.png", figsize=(16, 14)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="viridis", ax=ax, cbar=True, square=True,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Validation Set)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    savefig(fig, name)
    plt.show()
    return fig


def show_image_grid(images, titles=None, ncols=6, name=None, figsize_scale=2.0):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_scale, nrows * figsize_scale))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    if name:
        savefig(fig, name)
    plt.show()
    return fig
