"""
run_step5_evaluate.py  -  ขั้นที่ 5 (ส่วนแรก): Evaluation & Reporting บน validation
====================================================================================
- Learning curve (วาดซ้ำจาก training history เพื่อให้ได้ไฟล์ล่าสุด)
- Confusion Matrix (ตัวเลขดิบ + normalized) ด้วยโมเดลที่ดีที่สุด
- Classification Report ต่อคลาส (precision/recall/f1/support) -> CSV
- เทียบ per-class accuracy/recall ก่อน-หลัง augmentation โดยเฉพาะคลาสที่เคย
  อยู่ในลิสต์ "ทายผิดบ่อย"
- Top-10 คู่คลาสที่สับสนกันมากที่สุด

การทำนายชุด test 5% แยกไปอยู่ run_step6_test_inference.py

รัน:  uv run python run_step5_evaluate.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_pipeline import augment as A
from torch_pipeline import config as cfg
from torch_pipeline import data as D
from torch_pipeline import model as M
from torch_pipeline import reporting as R
from torch_pipeline import viz

BEST_PATH = cfg.MODELS_DIR / "best_model.pth"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=str(BEST_PATH))
    p.add_argument("--split", default="val", choices=("val", "train", "test"))
    p.add_argument("--batch-size", type=int, default=cfg.EVAL_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=cfg.EVAL_NUM_WORKERS)
    p.add_argument("--weak-threshold", type=float, default=0.80)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    cfg.ensure_dirs()
    cfg.set_seed()
    device = cfg.get_device()
    print(cfg.describe_device())

    from pathlib import Path

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"ไม่พบ checkpoint: {ckpt_path} - รัน run_step4_train.py ก่อน")

    # --- 5.1 โหลดโมเดลที่ดีที่สุด ---------------------------------------
    label_map = D.load_label_map()
    split_df = D.load_splits()
    class_to_idx, _ = D.load_class_index()
    num_classes = len(class_to_idx)

    # inference ล้วน ๆ: channels_last เร็วกว่า contiguous จริงบนการ์ดนี้
    model = M.prepare_model(
        M.build_resnet50(num_classes, pretrained=False), device, cfg.CHANNELS_LAST_EVAL
    )
    ckpt = M.load_checkpoint(ckpt_path, model, device)
    history = list(ckpt.get("history", []))
    print(
        f"โหลด {ckpt_path.name}: epoch {ckpt.get('epoch')} "
        f"best val acc {ckpt.get('best_val_acc', float('nan')):.4f}"
    )

    rows = split_df[split_df["split"] == args.split]
    ds = D.ThaiCharDataset(rows, class_to_idx, A.build_eval_transform())
    loader = D.make_loader(ds, args.batch_size, False, args.num_workers, device)
    print(f"ประเมินบน split '{args.split}': {len(ds):,} ภาพ")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"
    out = M.evaluate(
        model, loader, criterion, device, use_amp, return_probs=True, desc=args.split,
        channels_last=cfg.CHANNELS_LAST_EVAL, num_classes=num_classes,
    )
    y_true, y_pred, probs = out["y_true"], out["y_pred"], out["probs"]

    # --- 5.2 learning curve ----------------------------------------------
    viz.setup_thai_font()
    if history and all("val_macro_f1" in h for h in history):
        viz.plot_learning_curves(history, best_metric=cfg.BEST_METRIC)
    elif history:
        viz.plot_learning_curves(history)

    # --- 5.3 confusion matrix --------------------------------------------
    cm, cm_norm = R.build_confusion(y_true, y_pred, num_classes)
    ticks = R.class_tick_labels(class_to_idx, label_map)
    np.save(cfg.REPORTS_DIR / f"05_confusion_matrix_{args.split}.npy", cm)
    R.save_csv(
        pd.DataFrame(cm, index=ticks, columns=ticks),
        cfg.REPORTS_DIR / f"05_confusion_matrix_{args.split}.csv",
        index=True,
    )
    subtitle = f"{args.split} · {len(y_true):,} ภาพ · {num_classes} คลาส"
    viz.plot_confusion_matrix(
        cm,
        ticks,
        f"05_confusion_matrix_{args.split}.png",
        f"Confusion Matrix - ResNet50 + Error-driven Balanced Augmentation ({subtitle})",
    )
    viz.plot_confusion_matrix(
        cm_norm,
        ticks,
        f"05_confusion_matrix_{args.split}_normalized.png",
        f"Confusion Matrix - normalized ต่อแถว % ({subtitle})",
        normalized=True,
    )

    # --- 5.4 classification report ---------------------------------------
    per_class = R.per_class_table(y_true, y_pred, class_to_idx, label_map)
    R.save_csv(per_class, cfg.REPORTS_DIR / f"05_classification_report_{args.split}.csv")

    metrics = R.overall_metrics(y_true, y_pred, probs, num_classes)
    measured = per_class[per_class["support"] > 0]
    metrics.update(
        {
            "split": args.split,
            "n_images": int(len(y_true)),
            "n_classes": num_classes,
            "n_classes_measured": int(len(measured)),
            "classes_with_zero_support": per_class.loc[
                per_class["support"] == 0, "class_id"
            ].tolist(),
            "classes_below_threshold": int((measured["f1-score"] < args.weak_threshold).sum()),
            "weak_threshold": args.weak_threshold,
            "checkpoint": str(ckpt_path),
        }
    )
    R.print_overall(metrics, f"ResNet50 + augmentation - {args.split}")
    print(
        f"คลาสที่ F1 < {args.weak_threshold:.2f} = "
        f"{metrics['classes_below_threshold']} / {metrics['n_classes_measured']} คลาสที่วัดได้"
    )
    print("\n10 คลาสที่ F1 ต่ำสุด (weak classes ของโมเดลสุดท้าย):")
    print(
        measured.nsmallest(10, "f1-score")[
            ["label", "precision", "recall", "f1-score", "support"]
        ].to_string(index=False)
    )
    R.save_json(metrics, cfg.REPORTS_DIR / f"05_metrics_{args.split}.json")

    viz.plot_per_class_metric(
        per_class,
        "f1-score",
        f"05_f1_per_class_{args.split}.png",
        f"F1-score รายคลาส เรียงจากต่ำไปสูง ({args.split})",
        threshold=args.weak_threshold,
    )

    pairs = R.top_confused_pairs(cm, class_to_idx, label_map, top_n=10)
    R.save_csv(pairs, cfg.REPORTS_DIR / f"05_top_confused_pairs_{args.split}.csv")
    if not pairs.empty:
        print("\nTop-10 คู่คลาสที่สับสนกันมากที่สุด:")
        for r in pairs.itertuples():
            print(
                f"  {r.true_class} ({r.true_char}) -> {r.pred_class} ({r.pred_char}): "
                f"{r.count} ครั้ง ({r.pct_of_true_class}% ของคลาสจริง, support={r.true_support})"
            )

    print("\n5 คลาสที่แย่ที่สุด (F1 ต่ำสุด):")
    print(
        measured.nsmallest(5, "f1-score")[
            ["label", "precision", "recall", "f1-score", "support"]
        ].to_string(index=False)
    )

    # --- 5.5 เทียบ per-class ก่อน-หลัง augmentation ---------------------
    baseline_csv = cfg.REPORTS_DIR / "02_baseline_per_class.csv"
    weak_json = cfg.REPORTS_DIR / "02_weak_classes.json"
    if args.split == "val" and baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv, dtype={"class_id": str})
        weak_classes: set[str] = set()
        if weak_json.exists():
            with open(weak_json, encoding="utf-8") as f:
                weak_classes = {str(c) for c in json.load(f)["weak_classes"]}

        comparison = (
            baseline[["class_id", "label", "recall", "f1-score", "support"]]
            .rename(
                columns={"recall": "recall_baseline", "f1-score": "f1-score_baseline"}
            )
            .merge(
                per_class[["class_id", "recall", "f1-score"]].rename(
                    columns={"recall": "recall_final", "f1-score": "f1-score_final"}
                ),
                on="class_id",
            )
        )
        comparison["delta_recall"] = comparison["recall_final"] - comparison["recall_baseline"]
        comparison["delta_f1"] = comparison["f1-score_final"] - comparison["f1-score_baseline"]
        comparison["was_weak"] = comparison["class_id"].isin(weak_classes)
        comparison = comparison.sort_values("delta_f1", ascending=False)
        R.save_csv(comparison, cfg.REPORTS_DIR / "05_recall_before_after.csv")

        measured_cmp = comparison[comparison["support"] > 0]
        weak_cmp = measured_cmp[measured_cmp["was_weak"]]
        strong_cmp = measured_cmp[~measured_cmp["was_weak"]]

        summary = {
            "macro_f1_baseline": float(measured_cmp["f1-score_baseline"].mean()),
            "macro_f1_final": float(measured_cmp["f1-score_final"].mean()),
            "macro_f1_delta": float(measured_cmp["delta_f1"].mean()),
            "mean_recall_baseline": float(measured_cmp["recall_baseline"].mean()),
            "mean_recall_final": float(measured_cmp["recall_final"].mean()),
            "mean_recall_delta": float(measured_cmp["delta_recall"].mean()),
            "weak_classes_mean_f1_baseline": float(weak_cmp["f1-score_baseline"].mean())
            if len(weak_cmp)
            else None,
            "weak_classes_mean_f1_final": float(weak_cmp["f1-score_final"].mean())
            if len(weak_cmp)
            else None,
            "weak_classes_mean_f1_delta": float(weak_cmp["delta_f1"].mean())
            if len(weak_cmp)
            else None,
            "other_classes_mean_f1_delta": float(strong_cmp["delta_f1"].mean())
            if len(strong_cmp)
            else None,
            "n_classes_f1_improved": int((measured_cmp["delta_f1"] > 0).sum()),
            "n_classes_f1_worse": int((measured_cmp["delta_f1"] < 0).sum()),
            "n_weak_classes_f1_improved": int((weak_cmp["delta_f1"] > 0).sum())
            if len(weak_cmp)
            else 0,
            "n_weak_classes": int(len(weak_cmp)),
        }
        R.save_json(summary, cfg.REPORTS_DIR / "05_before_after_summary.json")

        print("\n" + "=" * 66)
        print("เทียบ per-class F1 ก่อน-หลัง Error-driven Balanced Augmentation")
        print("-" * 66)
        print(
            f"F1 เฉลี่ยทุกคลาส           {summary['macro_f1_baseline']:.4f} -> "
            f"{summary['macro_f1_final']:.4f}  ({summary['macro_f1_delta']:+.4f})"
        )
        print(
            f"recall เฉลี่ยทุกคลาส        {summary['mean_recall_baseline']:.4f} -> "
            f"{summary['mean_recall_final']:.4f}  ({summary['mean_recall_delta']:+.4f})"
        )
        if summary["weak_classes_mean_f1_delta"] is not None:
            print(
                f"F1 คลาสที่ทายผิดบ่อย       {summary['weak_classes_mean_f1_baseline']:.4f} -> "
                f"{summary['weak_classes_mean_f1_final']:.4f}  "
                f"({summary['weak_classes_mean_f1_delta']:+.4f})  "
                f"[{summary['n_weak_classes']} คลาส]"
            )
        if summary["other_classes_mean_f1_delta"] is not None:
            print(f"F1 คลาสอื่น ๆ              {summary['other_classes_mean_f1_delta']:+.4f}")
        print(
            f"F1 ดีขึ้น {summary['n_classes_f1_improved']} คลาส / "
            f"แย่ลง {summary['n_classes_f1_worse']} คลาส"
        )
        print("=" * 66)

        def print_f1_rows(frame) -> None:
            # ใช้ iterrows เพราะชื่อคอลัมน์มีขีดกลาง ("f1-score_final") ซึ่ง
            # itertuples จะเปลี่ยนเป็นชื่อ positional (_4, _6) ที่เลื่อนตำแหน่ง
            # ทุกครั้งที่เพิ่มหรือลดคอลัมน์
            for _, row in frame.iterrows():
                star = "★" if row["was_weak"] else " "
                print(
                    f"  {star} {row['label']:10s} F1 {row['f1-score_baseline']:.3f} -> "
                    f"{row['f1-score_final']:.3f} ({row['delta_f1']:+.3f})"
                )

        print("\n5 คลาสที่ F1 ดีขึ้นมากที่สุด:")
        print_f1_rows(comparison.head(5))
        worse = comparison.tail(5).iloc[::-1]
        if (worse["delta_f1"] < 0).any():
            print("\n5 คลาสที่ F1 แย่ลงมากที่สุด:")
            print_f1_rows(worse)

        viz.plot_before_after_metric(
            comparison, metric="f1-score", name="05_f1_before_after_augmentation.png"
        )
        viz.plot_before_after_metric(
            comparison, metric="recall", name="05_recall_before_after_augmentation.png"
        )
    elif args.split == "val":
        print(f"\n[warn] ไม่พบ {baseline_csv} - ข้ามการเทียบก่อน-หลัง augmentation")

    print("\nขั้นที่ 5 เสร็จแล้ว - ต่อไปรัน run_step6_test_inference.py")


if __name__ == "__main__":
    main()
