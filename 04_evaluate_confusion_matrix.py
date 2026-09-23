"""
04_evaluate_confusion_matrix.py
================================
Evaluation + Confusion Matrix report for the 72-class Thai character/digit
classifier (MobileNetV2 transfer-learning model trained in notebook 03).

What it produces (all under outputs/figures/ and outputs/reports/):
  04_confusion_matrix_counts.png       72x72 confusion matrix, raw counts
  04_confusion_matrix_normalized.png   72x72 confusion matrix, row-normalized (%)
  04_f1_per_class.png                  per-class F1 bar chart, sorted low -> high
  04_classification_report.csv         precision / recall / f1 / support per class
  04_per_class_accuracy.csv            per-class accuracy (= recall) + support
  04_top_confused_pairs.csv            top-N most-confused (true, pred) pairs
  04_metrics_summary.json              overall / macro / weighted headline metrics

Framework: TensorFlow 2.15 / tf.keras (this project is Keras, not PyTorch).

Usage
-----
    uv run python 04_evaluate_confusion_matrix.py                  # evaluate the val split
    uv run python 04_evaluate_confusion_matrix.py --split test     # if split.csv has a test split
    uv run python 04_evaluate_confusion_matrix.py --holdout-frac 0.25 --tag test5
    uv run python 04_evaluate_confusion_matrix.py --max-images 500 # quick smoke test

Note on the "test set": outputs/split.csv currently contains only `train` and
`val` (80/20, see src/data_utils.stratified_split) - there is no untouched 5%
test split, and carving one out of `train` now would leak data the model was
fit on. So the default here is the `val` split (12,545 images the weights were
never fit on). `--holdout-frac 0.25` deterministically takes a stratified 25%
of val (= 5% of the whole dataset) if the report needs a "5% test set" figure;
it is still data that early-stopping/checkpointing saw, so it is reported as a
holdout subset, not as a clean test set.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Thai text is printed to stdout below; Windows' default cp1252 console codec
# would raise UnicodeEncodeError on it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # file output only; no interactive display needed
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from tqdm import tqdm

from src.augmentation import load_and_resize
from src.data_utils import (
    CACHE_DIR,
    CLASS_ID_TO_THAI,
    FIGURES_DIR,
    MODELS_DIR,
    SPLIT_CSV,
    load_split,
)
from src.pipeline import IMG_SIZE, build_label_maps

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


# ---------------------------------------------------------------------------
# 0. Thai-capable font for the plots
# ---------------------------------------------------------------------------
def setup_thai_font() -> str:
    """Pick a font that actually has Thai glyphs.

    Matplotlib's default (DejaVu Sans) has none, so every Thai label would
    render as a tofu box. Tahoma ships with Windows and covers Thai + Latin;
    the rest are common fallbacks on macOS/Linux.
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
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    print("[warn] ไม่พบฟอนต์ที่รองรับภาษาไทย - label ภาษาไทยอาจแสดงเป็นกล่องสี่เหลี่ยม")
    return plt.rcParams["font.family"][0]


def class_display_label(class_id: str) -> str:
    """'161 ก' / '209 ◌ั' - the tick label for one class.

    Thai vowel signs and tone marks (e.g. ั, ิ, ่) are combining characters:
    on their own they attach to whatever precedes them and look broken, so
    they get the conventional U+25CC DOTTED CIRCLE carrier, exactly as Thai
    dictionaries print them.
    """
    thai = CLASS_ID_TO_THAI.get(str(class_id))
    if not thai:
        return str(class_id)
    if unicodedata.combining(thai[0]):
        thai = "◌" + thai
    return f"{class_id} {thai}"


# ---------------------------------------------------------------------------
# 1. Model + class mapping
# ---------------------------------------------------------------------------
def load_trained_model(model_path: Path, num_classes: int):
    """Load best_model.keras.

    compile=False: the model was trained with a custom focal-loss closure
    (src.model_utils.sparse_categorical_focal_loss) that is not serialized by
    name - and inference needs no loss anyway.

    safe_mode=False: the architecture contains a Lambda(preprocess_input)
    layer, and Keras refuses to deserialize Lambda layers by default since
    they can carry arbitrary pickled code. This checkpoint is our own
    (written by notebook 03), so allowing it is safe here.

    If whole-model deserialization still fails, we fall back to rebuilding the
    identical graph via build_transfer_model() and loading the weights in.
    """
    from tensorflow import keras

    try:
        try:
            model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        except TypeError:  # safe_mode kwarg does not exist on older Keras
            model = keras.models.load_model(model_path, compile=False)
        print(f"โหลดโมเดลสำเร็จ (load_model): {model_path}")
    except Exception as exc:  # noqa: BLE001 - any deserialization failure -> rebuild
        print(f"[warn] load_model ล้มเหลว ({type(exc).__name__}: {exc})")
        print("       กำลัง rebuild สถาปัตยกรรมเดิมแล้ว load_weights แทน...")
        from src.model_utils import build_transfer_model

        model, _ = build_transfer_model(
            "mobilenetv2",
            num_classes=num_classes,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            freeze_backbone=True,
        )
        model.load_weights(model_path)
        print(f"โหลด weights สำเร็จ: {model_path}")
    return model


def resolve_class_order(split_df: pd.DataFrame) -> list[str]:
    """Class index -> class_id, in EXACTLY the order used during training.

    Training read cache/class_maps.json (written by notebook 01); that file is
    gitignored and may be absent, but it was produced by
    src.pipeline.build_label_maps(), which sorts class_ids as strings - a
    deterministic order we can reproduce from split.csv alone.
    """
    maps_path = CACHE_DIR / "class_maps.json"
    if maps_path.exists():
        with open(maps_path, encoding="utf-8") as f:
            idx_to_class = json.load(f)["idx_to_class"]
        classes = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
        print(f"ลำดับคลาสอ่านจาก {maps_path}")
        return [str(c) for c in classes]

    class_to_idx, _ = build_label_maps(split_df)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    print("ลำดับคลาสสร้างใหม่จาก split.csv ด้วย build_label_maps() (เหมือนตอนเทรน)")
    return [str(c) for c in classes]


# ---------------------------------------------------------------------------
# 2. Evaluation data
# ---------------------------------------------------------------------------
def select_eval_rows(
    split_df: pd.DataFrame, split: str, holdout_frac: float, seed: int = 42
) -> pd.DataFrame:
    """Rows to evaluate on, optionally a stratified subset of the split."""
    available = sorted(split_df["split"].dropna().unique())
    if split not in available:
        raise SystemExit(
            f"split '{split}' ไม่มีใน {SPLIT_CSV} (มีเฉพาะ: {available}).\n"
            f"ดูหมายเหตุเรื่อง test set ที่หัวไฟล์นี้ - ใช้ --split val "
            f"หรือ --split val --holdout-frac 0.25 แทน"
        )
    rows = split_df[split_df["split"] == split].reset_index(drop=True)
    if holdout_frac <= 0:
        return rows

    # Stratified per class, so every class keeps representation in the subset.
    rng = np.random.RandomState(seed)
    keep: list[int] = []
    for _, group in rows.groupby("class_id"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_keep = max(1, int(round(len(idx) * holdout_frac)))
        keep.extend(idx[:n_keep])
    subset = rows.loc[sorted(keep)].reset_index(drop=True)
    print(
        f"ตัด holdout แบบ stratified {holdout_frac:.0%} ของ '{split}': "
        f"{len(subset):,} / {len(rows):,} ภาพ (seed={seed})"
    )
    return subset


def load_images(rows: pd.DataFrame, img_size: int) -> np.ndarray:
    """Decode + resize every image exactly as the training pipeline did
    (src.augmentation.load_and_resize; no augmentation at evaluation time)."""
    images = np.zeros((len(rows), img_size, img_size, 3), dtype=np.uint8)
    for i, fp in enumerate(tqdm(rows["filepath"].tolist(), desc="Loading images")):
        images[i] = load_and_resize(fp, size=img_size)
    return images


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    *,
    normalized: bool,
    out_path: Path,
    title: str,
    counts_scale: str = "log",
    dpi: int = 300,
    figsize=(24, 24),
) -> None:
    """One 72x72 heatmap. Sequential single hue (Blues): light = few, dark = many.

    counts_scale='log' for the raw-count version, because on an imbalanced
    dataset the diagonal sits 2-3 orders of magnitude above the off-diagonal
    cells - on a linear ramp every actual confusion washes out to white. Pass
    'linear' for a literally count-proportional ramp.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalized:
        kwargs = dict(vmin=0.0, vmax=100.0)
        cbar_label = "% ของภาพจริงในคลาสนั้น (แต่ละแถวรวมได้ 100%)"
    elif counts_scale == "log":
        kwargs = dict(norm=LogNorm(vmin=1, vmax=max(2, int(cm.max()))))
        cbar_label = "จำนวนภาพ (log scale)"
    else:
        kwargs = dict(vmin=0, vmax=int(cm.max()))
        cbar_label = "จำนวนภาพ"

    # Empty cells read as blank rather than as pale blue, so the eye only lands
    # on real predictions - essential when 72x72 = 5,184 cells are mostly zero.
    data = cm if normalized else np.ma.masked_where(cm == 0, cm)

    sns.heatmap(
        data,
        cmap="Blues",
        square=True,
        linewidths=0.15,
        linecolor="#f2f2f2",  # hairline grid so individual cells stay countable
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.45, "label": cbar_label, "pad": 0.01},
        ax=ax,
        **kwargs,
    )
    ax.set_facecolor("white")
    # seaborn hands the colorbar label in at rotation=270 (reads upside down)
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, rotation=90, labelpad=14)
    ax.set_xlabel("Predicted class (ทำนาย)", labelpad=14)
    ax.set_ylabel("True class (จริง)", labelpad=14)
    ax.set_title(title, pad=20)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"บันทึกรูป -> {out_path}")


def plot_f1_per_class(
    report_df: pd.DataFrame, out_path: Path, *, weak_threshold: float = 0.80, dpi: int = 300
) -> None:
    """Horizontal per-class F1 bars, worst class at the bottom.

    Horizontal because 72 class names never fit as readable x tick labels.
    Color encodes *status* (below threshold), not rank, so a class keeps its
    color whatever the sort order. Support is printed at the end of each bar:
    on an imbalanced dataset a low F1 at n=8 and a low F1 at n=600 are very
    different problems.

    Classes with zero support in this split are left out rather than drawn as
    F1=0 bars: they were never *tested*, so a zero there is an absence of
    evidence, not a failure. (stratified_split() sends single-image classes to
    train only, so val genuinely has a couple of them.) They stay in the CSVs
    and are named in the figure's footnote.
    """
    absent = report_df[report_df["support"] == 0]
    report_df = report_df[report_df["support"] > 0]
    df = report_df.sort_values("f1-score", ascending=True).reset_index(drop=True)
    weak = df["f1-score"] < weak_threshold
    colors = np.where(weak, "#C2410C", "#2563EB")  # status amber-red vs. neutral blue

    fig, ax = plt.subplots(figsize=(11, max(8, 0.26 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["f1-score"], height=0.62, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-score")
    ax.set_title(f"F1-score รายคลาส เรียงจากต่ำไปสูง (n = {len(df)} คลาส)", pad=14)
    if not absent.empty:
        ax.set_xlabel(
            "F1-score\n"
            f"ไม่แสดง {len(absent)} คลาสที่ไม่มีภาพใน split นี้ (support = 0): "
            + ", ".join(absent["label"]),
            fontsize=8,
        )

    for yi, (f1, support) in enumerate(zip(df["f1-score"], df["support"])):
        ax.text(
            min(f1, 1.0) + 0.012,
            yi,
            f"{f1:.2f}  (n={int(support)})",
            va="center",
            fontsize=7,
            color="#404040",
        )

    ax.axvline(weak_threshold, color="#9CA3AF", linestyle="--", linewidth=1)
    ax.legend(
        handles=[
            Patch(color="#C2410C", label=f"F1 < {weak_threshold:.2f} (คลาสที่ควรแก้)"),
            Patch(color="#2563EB", label=f"F1 >= {weak_threshold:.2f}"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)  # recessive grid, x only
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"บันทึกรูป -> {out_path}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    p.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    p.add_argument("--split", default="val", help="split ที่จะประเมิน (default: val)")
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help="ตัด stratified subset จาก split นี้ เช่น 0.25 = 25%% ของ val = 5%% ของทั้ง dataset",
    )
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--counts-scale",
        choices=("log", "linear"),
        default="log",
        help="สเกลสีของ confusion matrix เวอร์ชันตัวเลขดิบ (default: log)",
    )
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--weak-threshold", type=float, default=0.80)
    p.add_argument(
        "--max-images", type=int, default=0, help="จำกัดจำนวนภาพ (0 = ทั้งหมด) สำหรับทดลองรันเร็ว"
    )
    p.add_argument("--tag", default="", help="ต่อท้ายชื่อไฟล์ output เช่น --tag test5")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"ไม่พบไฟล์โมเดล: {args.model}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    font = setup_thai_font()
    sns.set_theme(style="white", context="notebook", font=font, font_scale=0.9)
    plt.rcParams["font.family"] = font  # sns.set_theme() resets it; re-assert

    # --- 4.1 data + class order ------------------------------------------
    split_df = load_split(args.split_csv)
    classes = resolve_class_order(split_df)
    num_classes = len(classes)
    class_labels = [class_display_label(c) for c in classes]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"จำนวนคลาส: {num_classes}")

    rows = select_eval_rows(split_df, args.split, args.holdout_frac)
    if args.max_images:
        rows = (
            rows.sample(n=min(args.max_images, len(rows)), random_state=42)
            .sort_index()
            .reset_index(drop=True)
        )
        print(f"[smoke test] ใช้เพียง {len(rows):,} ภาพ")

    unknown = set(rows["class_id"].astype(str)) - set(class_to_idx)
    if unknown:
        raise SystemExit(f"split มี class_id ที่ไม่อยู่ใน label mapping: {sorted(unknown)}")

    y_true = rows["class_id"].astype(str).map(class_to_idx).to_numpy()
    images = load_images(rows, args.img_size)
    print(f"ชุดประเมิน '{args.split}': {images.shape[0]:,} ภาพ, {images.shape[1:]} ต่อภาพ")

    # --- 4.2 predict ------------------------------------------------------
    model = load_trained_model(args.model, num_classes)
    out_units = model.output_shape[-1]
    if out_units != num_classes:
        raise SystemExit(
            f"โมเดลมี output {out_units} คลาส แต่ label mapping มี {num_classes} คลาส"
        )

    # The preprocess_input step lives inside the model (Lambda layer), so raw
    # uint8 images are exactly what it expects - do not rescale here.
    probs = model.predict(images, batch_size=args.batch_size, verbose=1)
    y_pred = probs.argmax(axis=1)

    # --- 4.3 confusion matrix --------------------------------------------
    label_idx = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=label_idx)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm * 100.0, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0
    )  # classes with 0 support stay an all-zero row instead of NaN

    np.save(REPORTS_DIR / f"04_confusion_matrix{suffix}.npy", cm)
    pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(
        REPORTS_DIR / f"04_confusion_matrix{suffix}.csv", encoding="utf-8-sig"
    )

    subtitle = f"{args.split} split · {len(y_true):,} ภาพ · {num_classes} คลาส"
    plot_confusion_matrix(
        cm,
        class_labels,
        normalized=False,
        out_path=FIGURES_DIR / f"04_confusion_matrix_counts{suffix}.png",
        title=f"Confusion Matrix - จำนวนภาพ ({subtitle})",
        counts_scale=args.counts_scale,
        dpi=args.dpi,
    )
    plot_confusion_matrix(
        cm_norm,
        class_labels,
        normalized=True,
        out_path=FIGURES_DIR / f"04_confusion_matrix_normalized{suffix}.png",
        title=f"Confusion Matrix - normalized ต่อแถว (%) ({subtitle})",
        dpi=args.dpi,
    )

    # --- 4.4 classification report ---------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        labels=label_idx,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        [
            {
                "class_id": cid,
                "thai_char": CLASS_ID_TO_THAI.get(cid, ""),
                "label": lab,
                **{k: report[lab][k] for k in ("precision", "recall", "f1-score", "support")},
            }
            for cid, lab in zip(classes, class_labels)
        ]
    )
    averages = pd.DataFrame(
        [
            {"class_id": name, "thai_char": "", "label": name, **report[name]}
            for name in ("macro avg", "weighted avg")
            if name in report
        ]
    )
    report_path = REPORTS_DIR / f"04_classification_report{suffix}.csv"
    # utf-8-sig so Excel on Windows opens the Thai characters correctly
    pd.concat([per_class, averages], ignore_index=True).to_csv(
        report_path, index=False, encoding="utf-8-sig"
    )
    print(f"บันทึก classification report -> {report_path}")

    # --- 4.5 accuracy: overall + per class -------------------------------
    overall_acc = accuracy_score(y_true, y_pred)
    # per-class accuracy = diagonal / row total = that class's recall
    supports = row_sums.ravel()
    per_class_acc = np.divide(
        np.diag(cm).astype(float),
        supports,
        out=np.full(num_classes, np.nan),
        where=supports > 0,
    )
    acc_df = pd.DataFrame(
        {
            "class_id": classes,
            "thai_char": [CLASS_ID_TO_THAI.get(c, "") for c in classes],
            "label": class_labels,
            "support": supports,
            "correct": np.diag(cm),
            "accuracy": per_class_acc,
        }
    ).sort_values("accuracy", na_position="last")
    acc_path = REPORTS_DIR / f"04_per_class_accuracy{suffix}.csv"
    acc_df.to_csv(acc_path, index=False, encoding="utf-8-sig")
    print(f"บันทึก per-class accuracy -> {acc_path}")

    # --- 4.6 most-confused pairs -----------------------------------------
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    flat = np.argsort(off_diag.ravel())[::-1][: args.top_pairs]
    pairs = []
    for f in flat:
        i, j = divmod(int(f), num_classes)
        count = int(off_diag[i, j])
        if count == 0:
            continue
        pairs.append(
            {
                "true_class": classes[i],
                "true_char": CLASS_ID_TO_THAI.get(classes[i], ""),
                "pred_class": classes[j],
                "pred_char": CLASS_ID_TO_THAI.get(classes[j], ""),
                "count": count,
                "true_support": int(supports[i]),
                "pct_of_true_class": round(count * 100.0 / max(1, int(supports[i])), 2),
            }
        )
    pairs_df = pd.DataFrame(pairs)
    pairs_path = REPORTS_DIR / f"04_top_confused_pairs{suffix}.csv"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    print(f"บันทึก top-{args.top_pairs} confused pairs -> {pairs_path}")

    # --- 4.7 F1 bar chart -------------------------------------------------
    plot_f1_per_class(
        per_class,
        FIGURES_DIR / f"04_f1_per_class{suffix}.png",
        weak_threshold=args.weak_threshold,
        dpi=args.dpi,
    )

    # --- 4.8 headline metrics --------------------------------------------
    measured = per_class[per_class["support"] > 0]
    summary = {
        "model": str(args.model),
        "split": args.split,
        "holdout_frac": args.holdout_frac,
        "n_images": int(len(y_true)),
        "n_classes": num_classes,
        "accuracy": float(overall_acc),
        # balanced accuracy = mean per-class recall: on a dataset ranging from
        # ~30 to ~4,800 images per class, this is the number that does not let
        # majority-class success hide minority-class failure
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "top5_accuracy": float(top_k_accuracy_score(y_true, probs, k=5, labels=label_idx)),
        # only over classes that actually appear in this split - a zero-support
        # class has no measurable F1 and must not be counted as a failure
        "n_classes_measured": int(measured.shape[0]),
        "classes_with_zero_support": [
            c for c, s in zip(classes, supports) if s == 0
        ],
        "classes_below_threshold": int((measured["f1-score"] < args.weak_threshold).sum()),
        "weak_threshold": args.weak_threshold,
    }
    summary_path = REPORTS_DIR / f"04_metrics_summary{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- 4.9 console summary ---------------------------------------------
    print("\n" + "=" * 66)
    print(f"Accuracy รวม           = {summary['accuracy']:.4f}")
    print(f"Balanced accuracy      = {summary['balanced_accuracy']:.4f}  (เฉลี่ย recall ต่อคลาส)")
    print(f"Macro F1               = {summary['macro_f1']:.4f}")
    print(f"Weighted F1            = {summary['weighted_f1']:.4f}")
    print(f"Top-5 accuracy         = {summary['top5_accuracy']:.4f}")
    print(
        f"คลาสที่ F1 < {args.weak_threshold:.2f}       = "
        f"{summary['classes_below_threshold']} / {summary['n_classes_measured']} คลาสที่วัดได้"
    )
    if summary["classes_with_zero_support"]:
        print(
            f"คลาสที่ไม่มีภาพใน split นี้ = {summary['classes_with_zero_support']} "
            f"(ไม่ถูกนับในสถิติข้างบน)"
        )
    print("=" * 66)

    print("\n5 คลาสที่แย่ที่สุด (F1 ต่ำสุด):")
    worst = measured.nsmallest(5, "f1-score")[
        ["label", "precision", "recall", "f1-score", "support"]
    ]
    print(worst.to_string(index=False))

    if not pairs_df.empty:
        print(f"\nTop-{len(pairs_df)} คู่คลาสที่สับสนกันมากที่สุด:")
        for r in pairs_df.itertuples():
            print(
                f"  {r.true_class} ({r.true_char}) -> {r.pred_class} ({r.pred_char}): "
                f"{r.count} ครั้ง ({r.pct_of_true_class}% ของคลาสจริง, support={r.true_support})"
            )

    print(f"\nสรุปตัวเลข -> {summary_path}")
    if args.split != "test":
        print(f"\n[หมายเหตุ] ประเมินบน split '{args.split}' - ดูหัวไฟล์เรื่องการไม่มี test set แยก")


if __name__ == "__main__":
    main()
