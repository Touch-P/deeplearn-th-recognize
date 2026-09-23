"""
torch_pipeline/viz.py
=====================
กราฟทุกใบของ pipeline นี้ เซฟเป็น .png ความละเอียดสูงไว้ใช้ในรายงาน/สไลด์

ทุกฟังก์ชันเซฟไฟล์แล้วคืน path กลับมา และไม่เรียก plt.show() เพราะใช้ backend
"Agg" (เขียนไฟล์เท่านั้น) จึงรันบนเครื่องที่ไม่มีหน้าจอได้
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from torch_pipeline import config as cfg

# สีประจำ pipeline: น้ำเงิน = ปกติ, แดงอิฐ = ต้องสนใจ/อ่อนแอ, เทา = อ้างอิง
BLUE = "#2563EB"
RED = "#C2410C"
GREEN = "#15803D"
GREY = "#9CA3AF"
GRID = "#E5E7EB"


def setup_thai_font() -> str:
    """เลือกฟอนต์ที่มี glyph ภาษาไทยจริง

    ฟอนต์เริ่มต้นของ matplotlib (DejaVu Sans) ไม่มีอักขระไทย ป้ายกำกับไทย
    ทั้งหมดจะกลายเป็นกล่องสี่เหลี่ยม Tahoma มาพร้อม Windows และครอบคลุมทั้ง
    ไทยและละติน ตัวที่เหลือเป็น fallback สำหรับ macOS/Linux
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (
        "Tahoma",
        "Leelawadee UI",
        "Leelawadee",
        "Noto Sans Thai",
        "Sarabun",
        "TH Sarabun New",
        "Angsana New",
        "Thonburi",
    ):
        if candidate in available:
            sns.set_theme(style="white", context="notebook", font=candidate, font_scale=0.9)
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    print("[warn] ไม่พบฟอนต์ภาษาไทย - ป้ายกำกับไทยอาจเป็นกล่องสี่เหลี่ยม")
    return str(plt.rcParams["font.family"][0])


def _save(fig, name: str, dpi: int = 300) -> Path:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"บันทึกรูป -> {path}")
    return path


# ---------------------------------------------------------------------------
# ขั้นที่ 1: สำรวจข้อมูล
# ---------------------------------------------------------------------------
def plot_class_distribution(
    counts: dict[str, int], labels: dict[str, str], name: str = "01_class_distribution.png"
) -> Path:
    """bar chart จำนวนภาพต่อคลาส เรียงจากมากไปน้อย"""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [labels.get(c, c) for c, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(range(len(values)), values, color=BLUE, width=0.72)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_xlabel("คลาส (รหัส + อักษรไทย)")
    ax.set_ylabel("จำนวนภาพ")
    ax.set_title(
        f"จำนวนภาพต่อคลาส เรียงจากมากไปน้อย - {len(values)} คลาส, "
        f"รวม {sum(values):,} ภาพ (มากสุด {max(values):,} / น้อยสุด {min(values):,} = "
        f"{max(values) / max(1, min(values)):.0f} เท่า)"
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


def plot_sample_grid(
    df: pd.DataFrame,
    labels: dict[str, str],
    per_class: int = 4,
    max_classes: int | None = None,
    seed: int = cfg.SEED,
    name: str = "01_sample_images.png",
) -> Path:
    """สุ่มภาพตัวอย่าง per_class ภาพต่อคลาส แสดงเป็นตาราง (1 คลาส = 1 แถว)"""
    from torch_pipeline.data import load_rgb

    rng = np.random.RandomState(seed)
    class_ids = sorted(df["class_id"].unique(), key=lambda c: int(c))
    if max_classes:
        class_ids = class_ids[:max_classes]

    nrows, ncols = len(class_ids), per_class
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.45))
    axes = np.atleast_2d(axes)

    for r, class_id in enumerate(class_ids):
        pool = df[df["class_id"] == class_id]["filepath"].tolist()
        picks = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                ax.imshow(load_rgb(picks[c]))
            if c == 0:
                ax.set_title(
                    f"{labels.get(class_id, class_id)}  (n={len(pool)})",
                    fontsize=8,
                    loc="left",
                    color="#404040",
                )

    fig.suptitle(f"ตัวอย่างภาพ {per_class} ใบต่อคลาส", y=1.001, fontsize=13)
    fig.tight_layout()
    return _save(fig, name, dpi=200)


def plot_split_summary(
    summary: pd.DataFrame, labels: dict[str, str], name: str = "01_split_per_class.png"
) -> Path:
    """แท่งซ้อนแสดง train/val/test ต่อคลาส - ใช้ยืนยันว่า stratify ได้จริง"""
    df = summary.sort_values("total", ascending=False)
    names = [labels.get(str(c), str(c)) for c in df.index]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x, df["train"], color=BLUE, width=0.72, label="train")
    ax.bar(x, df["val"], bottom=df["train"], color=GREEN, width=0.72, label="val")
    ax.bar(
        x,
        df["test"],
        bottom=df["train"] + df["val"],
        color=RED,
        width=0.72,
        label="test 5% (กันไว้ให้อาจารย์)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("จำนวนภาพ")
    ax.set_title("การแบ่ง train / val / test ต่อคลาส (stratified)")
    ax.legend(frameon=True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# ขั้นที่ 3: augmentation
# ---------------------------------------------------------------------------
def plot_before_after_counts(
    before: dict[str, int],
    after: dict[str, int],
    weak_classes: set[str],
    labels: dict[str, str],
    target: int | None = None,
    name: str = "03_counts_before_after_augmentation.png",
) -> Path:
    """เทียบจำนวนภาพต่อคลาส ก่อน-หลัง augmentation

    เรียงตามจำนวนก่อน augment เพื่อให้เห็นว่าเดิม imbalance แค่ไหน และหลัง
    augment ทุกคลาสถูกยกขึ้นมาเท่ากันที่เส้น target
    """
    order = sorted(before, key=lambda c: before[c], reverse=True)
    x = np.arange(len(order))
    before_v = [before[c] for c in order]
    after_v = [after.get(c, 0) for c in order]
    names = [labels.get(c, c) for c in order]
    tick_colors = [RED if c in weak_classes else "#404040" for c in order]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x, after_v, color="#BFDBFE", width=0.78, label="หลัง augment (จำนวนที่โมเดลเห็น/epoch)")
    ax.bar(x, before_v, color=BLUE, width=0.78, label="ก่อน augment (ภาพจริง)")
    target = int(target if target else (max(after_v) if after_v else 0))
    ax.axhline(target, color=GREY, linestyle="--", linewidth=1)
    ax.text(
        len(order) - 0.5, target, f" target = {target:,} ภาพ/คลาส",
        va="bottom", ha="right", fontsize=9, color="#404040",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.set_ylabel("จำนวนภาพ")
    ax.set_title(
        "จำนวนภาพต่อคลาสใน training set ก่อน-หลัง Error-driven Balanced Augmentation\n"
        f"(ชื่อคลาสสีแดง = คลาสที่ทายผิดบ่อยจาก baseline, augment เข้มข้นกว่า - {len(weak_classes)} คลาส)",
        fontsize=12,
    )
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


def plot_augmentation_examples(
    manifest: pd.DataFrame,
    labels: dict[str, str],
    n_sources: int = 6,
    n_variants: int = 5,
    seed: int = cfg.SEED,
    name: str = "03_augmentation_examples.png",
) -> Path:
    """ต้นฉบับ (คอลัมน์ซ้าย) vs ภาพหลัง augment หลายแบบ (คอลัมน์ถัด ๆ ไป)

    เลือกภาพต้นแบบจากคลาสที่ถูก augment แบบ intense ก่อน เพราะเป็นตัวที่
    น่าสนใจที่สุดในรายงาน แล้วเติมด้วยคลาสแบบ normal
    """
    from torch_pipeline.augment import augment_preview
    from torch_pipeline.data import load_rgb

    rng = np.random.RandomState(seed)
    picked: list[tuple[str, str, str]] = []  # (filepath, class_id, kind)
    for kind in ("intense", "normal"):
        sub = manifest[manifest["aug_kind"] == kind]
        if sub.empty:
            continue
        classes = sorted(sub["class_id"].unique(), key=lambda c: int(c))
        rng.shuffle(classes)
        for class_id in classes:
            rows = sub[sub["class_id"] == class_id]
            row = rows.iloc[0]
            picked.append((row["filepath"], class_id, kind))
            if len(picked) >= n_sources:
                break
        if len(picked) >= n_sources:
            break

    nrows, ncols = len(picked), n_variants + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 1.85))
    axes = np.atleast_2d(axes)

    for r, (path, class_id, kind) in enumerate(picked):
        axes[r, 0].imshow(load_rgb(path))
        axes[r, 0].set_ylabel(f"{labels.get(class_id, class_id)}\n({kind})", fontsize=8, rotation=0, ha="right", va="center")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for s in axes[r, 0].spines.values():
            s.set_edgecolor(GRID)
        if r == 0:
            axes[r, 0].set_title("ต้นฉบับ", fontsize=9)

        for c in range(1, ncols):
            ax = axes[r, c]
            ax.imshow(augment_preview(path, kind, seed=1000 * r + c))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"augment #{c}", fontsize=9)

    fig.suptitle(
        "ภาพต้นฉบับ vs ภาพหลัง augmentation (ไม่มีการ flip - ตัวอักษรไทยมีทิศทาง)",
        y=1.002,
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, name, dpi=200)


# ---------------------------------------------------------------------------
# ขั้นที่ 4-5: learning curve, confusion matrix, รายงานต่อคลาส
# ---------------------------------------------------------------------------
def plot_learning_curves(
    history: list[dict], name: str = "04_learning_curves.png", best_metric: str = "macro_f1"
) -> Path:
    """loss / accuracy / macro F1 ของ train-val ตาม epoch

    สามแผง: loss, Accuracy Rate (ที่โจทย์กำหนดให้แสดง) และ macro F1 ซึ่งเป็น
    เกณฑ์ที่ใช้เลือก best checkpoint จริง วงกลมเขียวทำเครื่องหมาย epoch ที่ดี
    ที่สุดตาม best_metric
    """
    epochs = [h["epoch"] for h in history]
    has_f1 = all("val_macro_f1" in h for h in history)
    ncols = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.6 * ncols, 5))

    def series(ax, train_key, val_key, title, ylabel, mark_best=False):
        ax.plot(epochs, [h[train_key] for h in history], color=BLUE, marker="o", markersize=4, label="train")
        ax.plot(epochs, [h[val_key] for h in history], color=RED, marker="o", markersize=4, label="validation")
        if mark_best:
            best = max(history, key=lambda h: h[val_key])
            ax.scatter([best["epoch"]], [best[val_key]], s=90, facecolor="none", edgecolor=GREEN, linewidth=2, zorder=5)
            ax.annotate(
                f"best = {best[val_key]:.4f}\n(epoch {best['epoch']})",
                (best["epoch"], best[val_key]),
                textcoords="offset points",
                xytext=(-10, -34),
                fontsize=9,
                color=GREEN,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

    series(axes[0], "train_loss", "val_loss", "Loss ต่อ epoch", "Loss (CrossEntropy + label smoothing 0.1)")
    series(
        axes[1], "train_acc", "val_acc", "Accuracy Rate ต่อ epoch", "Accuracy",
        mark_best=(best_metric == "acc"),
    )
    if has_f1:
        series(
            axes[2], "train_macro_f1", "val_macro_f1",
            "Macro F1 ต่อ epoch (เกณฑ์เลือก best checkpoint)", "Macro F1",
            mark_best=(best_metric == "macro_f1"),
        )

    for ax in axes:
        ax.legend(frameon=True)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


def plot_confusion_matrix(
    cm: np.ndarray,
    tick_labels: list[str],
    name: str,
    title: str,
    normalized: bool = False,
    counts_scale: str = "log",
    figsize=(24, 24),
    dpi: int = 300,
) -> Path:
    """heatmap 72x72 สีเดี่ยว (Blues) อ่อน = น้อย เข้ม = มาก

    เวอร์ชันตัวเลขดิบใช้ log scale เป็นค่าเริ่มต้น เพราะบน dataset ที่ไม่
    balance ค่าบนเส้นทแยงมุมสูงกว่านอกเส้นทแยง 2-3 order of magnitude
    ถ้าใช้สเกลเชิงเส้น จุดที่โมเดลสับสนจะจางเป็นสีขาวจนมองไม่เห็นทั้งหมด
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalized:
        kwargs = dict(vmin=0.0, vmax=100.0)
        cbar_label = "% ของภาพจริงในคลาสนั้น (แต่ละแถวรวมได้ 100%)"
        data = cm
    elif counts_scale == "log":
        kwargs = dict(norm=LogNorm(vmin=1, vmax=max(2, int(cm.max()))))
        cbar_label = "จำนวนภาพ (log scale)"
        data = np.ma.masked_where(cm == 0, cm)
    else:
        kwargs = dict(vmin=0, vmax=int(cm.max()))
        cbar_label = "จำนวนภาพ"
        data = np.ma.masked_where(cm == 0, cm)

    sns.heatmap(
        data,
        cmap="Blues",
        square=True,
        linewidths=0.15,
        linecolor="#f2f2f2",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"shrink": 0.45, "pad": 0.01},
        ax=ax,
        **kwargs,
    )
    ax.set_facecolor("white")
    ax.collections[0].colorbar.set_label(cbar_label, rotation=90, labelpad=14)
    ax.set_xlabel("Predicted class (ทำนาย)", labelpad=14)
    ax.set_ylabel("True class (จริง)", labelpad=14)
    ax.set_title(title, pad=20)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, name, dpi=dpi)


def plot_per_class_metric(
    df: pd.DataFrame,
    metric: str,
    name: str,
    title: str,
    threshold: float = 0.80,
    label_col: str = "label",
) -> Path:
    """แท่งนอนของ metric รายคลาส เรียงจากต่ำไปสูง

    แท่งนอนเพราะชื่อคลาส 72 อันวางเป็นแกน x แล้วอ่านไม่ออก สีบอก "สถานะ"
    (ต่ำกว่าเกณฑ์) ไม่ใช่บอกอันดับ คลาสหนึ่งจึงมีสีเดิมเสมอ และพิมพ์ support
    ท้ายแท่ง เพราะค่า metric ต่ำที่ n=8 กับที่ n=1,000 เป็นปัญหาคนละแบบ
    """
    plot_df = df[df["support"] > 0].sort_values(metric, ascending=True).reset_index(drop=True)
    absent = df[df["support"] == 0]
    colors = np.where(plot_df[metric] < threshold, RED, BLUE)

    fig, ax = plt.subplots(figsize=(11, max(8, 0.26 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df[metric], height=0.62, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[label_col], fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_title(title, pad=14)

    for yi, (value, support) in enumerate(zip(plot_df[metric], plot_df["support"])):
        ax.text(min(value, 1.0) + 0.012, yi, f"{value:.2f}  (n={int(support)})", va="center", fontsize=7, color="#404040")

    ax.axvline(threshold, color=GREY, linestyle="--", linewidth=1)
    ax.legend(
        handles=[
            Patch(color=RED, label=f"{metric} < {threshold:.2f}"),
            Patch(color=BLUE, label=f"{metric} >= {threshold:.2f}"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=9,
    )
    xlabel = metric
    if not absent.empty:
        xlabel += (
            f"\nไม่แสดง {len(absent)} คลาสที่ไม่มีภาพใน split นี้ (support = 0): "
            + ", ".join(absent[label_col].astype(str))
        )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)


def plot_before_after_metric(
    comparison: pd.DataFrame,
    metric: str = "recall",
    name: str = "05_recall_before_after_augmentation.png",
    top_n: int | None = None,
) -> Path:
    """เทียบ per-class recall ก่อน (baseline) vs หลัง (โมเดลสุดท้าย)

    เรียงตามส่วนต่าง เพื่อให้เห็นว่าคลาสไหนดีขึ้น/แย่ลงมากที่สุด และทำ
    เครื่องหมายคลาสที่เคยอยู่ในลิสต์ "ทายผิดบ่อย" เพราะนั่นคือกลุ่มที่
    error-driven augmentation ตั้งใจจะแก้
    """
    # คำนวณส่วนต่างเองจากคอลัมน์ก่อน/หลัง ไม่พึ่งคอลัมน์ delta ที่ผู้เรียกตั้งชื่อมา
    # (ก่อนหน้านี้เดาชื่อเป็น f"delta_{metric}" ซึ่งได้ "delta_f1-score" แต่ผู้เรียก
    # ตั้งชื่อว่า "delta_f1" ทำให้ KeyError)
    base_col, final_col = f"{metric}_baseline", f"{metric}_final"
    for col in (base_col, final_col, "label", "was_weak"):
        if col not in comparison.columns:
            raise KeyError(f"ตาราง comparison ไม่มีคอลัมน์ '{col}'")

    df = comparison.copy()
    df["_delta"] = df[final_col] - df[base_col]
    df = df.sort_values("_delta")
    if top_n:
        df = pd.concat([df.head(top_n), df.tail(top_n)])
    y = np.arange(len(df))

    base_v = df[base_col].to_numpy(dtype=float)
    final_v = df[final_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, max(8, 0.3 * len(df))))
    # เส้นเชื่อมค่าก่อน->หลังของแต่ละคลาส (ใช้ numpy array ไม่ใช่ itertuples
    # เพราะชื่อคอลัมน์มีขีดกลาง itertuples จะเปลี่ยนเป็นชื่อ positional)
    for yi, (b, f) in enumerate(zip(base_v, final_v)):
        ax.plot([b, f], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter(base_v, y, s=42, color=GREY, zorder=2, label="baseline (ก่อน augment)")
    ax.scatter(
        final_v,
        y,
        s=42,
        color=np.where(df["_delta"].to_numpy() >= 0, GREEN, RED),
        zorder=3,
        label="โมเดลสุดท้าย (หลัง augment)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{'★ ' if w else ''}{lab}" for lab, w in zip(df["label"], df["was_weak"])], fontsize=8
    )
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel(f"{metric} ต่อคลาส (validation)")
    ax.set_title(
        f"Per-class {metric} ก่อน-หลัง Error-driven Balanced Augmentation\n"
        "★ = คลาสที่ถูกจัดเป็น 'ทายผิดบ่อย' จาก baseline (เขียว = ดีขึ้น, แดง = แย่ลง)",
        fontsize=12,
    )
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, name)
