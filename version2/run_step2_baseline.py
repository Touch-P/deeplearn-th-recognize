"""
run_step2_baseline.py  -  ขั้นที่ 2: Baseline model เพื่อวิเคราะห์ error
=========================================================================
- ResNet50 pretrained ImageNet (IMAGENET1K_V2), เปลี่ยน fc -> 72 คลาส
- เทรนเร็ว ๆ 5 epochs บน train เดิม (ยังไม่ augment) เพื่อดู error pattern
- Confusion Matrix + ตาราง per-class recall/accuracy บน validation
- ระบุ "คลาสที่ทายผิดบ่อย" (recall < mean-1SD หรือ bottom 20%) เก็บเป็น JSON
  ให้ขั้นที่ 3 ใช้

โมเดล baseline นี้ใช้เพื่อ "วิเคราะห์" เท่านั้น ขั้นที่ 4 จะสร้าง ResNet50
ตัวใหม่จาก pretrained weights ใหม่ ไม่ได้ต่อจาก weight ของ baseline

รัน:  uv run python run_step2_baseline.py
      uv run python run_step2_baseline.py --epochs 3 --limit-per-class 60   # ทดสอบเร็ว
"""

from __future__ import annotations

import argparse

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=cfg.BASELINE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--rule", default=cfg.WEAK_CLASS_RULE, choices=("both", "mean-sd", "bottom20"))
    p.add_argument(
        "--weak-metric",
        default=cfg.WEAK_CLASS_METRIC,
        choices=("f1-score", "recall", "precision"),
        help="metric ที่ใช้ตัดสินว่าคลาสไหนอ่อนแอ (ค่าเริ่มต้น: per-class F1)",
    )
    p.add_argument(
        "--best-metric",
        default=cfg.BEST_METRIC,
        choices=("macro_f1", "acc"),
        help="เกณฑ์เลือก checkpoint ที่ดีที่สุด (ค่าเริ่มต้น: macro F1)",
    )
    p.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="จำกัดจำนวนภาพต่อคลาส (0 = ใช้ทั้งหมด) สำหรับ smoke test",
    )
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    cfg.ensure_dirs()
    cfg.set_seed(args.seed)
    device = cfg.get_device()
    print(cfg.describe_device())

    # --- 2.1 ข้อมูล: train เดิม (ไม่ augment) + val --------------------
    label_map = D.load_label_map()
    split_df = D.load_splits()
    class_to_idx, _ = D.load_class_index()
    num_classes = len(class_to_idx)

    train_rows = split_df[split_df["split"] == "train"]
    val_rows = split_df[split_df["split"] == "val"]
    if args.limit_per_class:
        train_rows = train_rows.groupby("class_id", group_keys=False).head(args.limit_per_class)
        val_rows = val_rows.groupby("class_id", group_keys=False).head(
            max(2, args.limit_per_class // 4)
        )
    print(f"baseline train = {len(train_rows):,} ภาพ | val = {len(val_rows):,} ภาพ")

    # ทั้ง train และ val ใช้ transform แบบไม่สุ่ม เพราะขั้นนี้ต้องการเห็น
    # error pattern ของข้อมูลดิบ ไม่ใช่ผลของ augmentation
    eval_tf = A.build_eval_transform()
    train_ds = D.ThaiCharDataset(train_rows, class_to_idx, eval_tf)
    val_ds = D.ThaiCharDataset(val_rows, class_to_idx, eval_tf)
    train_loader = D.make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = D.make_loader(val_ds, cfg.EVAL_BATCH_SIZE, False, args.num_workers, device)

    # --- 2.2 โมเดล -------------------------------------------------------
    model = M.prepare_model(
        M.build_resnet50(num_classes, pretrained=True), device, cfg.CHANNELS_LAST_TRAIN
    )
    total, trainable = M.count_parameters(model)
    print(f"ResNet50: พารามิเตอร์ {total:,} ตัว (เทรนได้ {trainable:,} ตัว)")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- 2.3 เทรน --------------------------------------------------------
    log_path = M.start_log(cfg.LOGS_DIR / "02_baseline_log.jsonl")
    timer = M.EpochTimer(args.epochs)
    best_score, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        timer.start()
        tr = M.train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs,
            num_classes=num_classes,
        )
        val_out = M.evaluate(
            model, val_loader, criterion, device, use_amp, desc=f"val {epoch}",
            num_classes=num_classes,
        )
        elapsed = timer.stop()

        record = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": val_out["loss"],
            "val_acc": val_out["acc"],
            "val_macro_f1": val_out["macro_f1"],
            "seconds": round(elapsed, 1),
        }
        history.append(record)
        M.append_log(log_path, record)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} F1 {tr['macro_f1']:.4f}  |  "
            f"val loss {val_out['loss']:.4f} acc {val_out['acc']:.4f} F1 {val_out['macro_f1']:.4f}  |  "
            f"{M.format_eta(elapsed)} (เหลือ ~{timer.eta(epoch)})"
        )

        # เลือก checkpoint ด้วย macro F1 ไม่ใช่ accuracy (ดู config.BEST_METRIC)
        score = val_out["macro_f1"] if args.best_metric == "macro_f1" else val_out["acc"]
        if score > best_score:
            best_score = score
            M.save_checkpoint(
                cfg.MODELS_DIR / "baseline_model.pth",
                model,
                epoch=epoch,
                best_val_acc=val_out["acc"],
                history=history,
                extra={
                    "stage": "baseline",
                    "class_to_idx": class_to_idx,
                    "best_metric": args.best_metric,
                    "best_score": best_score,
                    "best_val_macro_f1": val_out["macro_f1"],
                },
            )
            print(f"  -> {args.best_metric} ดีที่สุดใหม่ = {best_score:.4f} (เซฟ baseline_model.pth)")

    # --- 2.4 ประเมินครั้งสุดท้ายบน val + confusion matrix ---------------
    val_out = M.evaluate(
        model, val_loader, criterion, device, use_amp, return_probs=True, desc="val final",
        num_classes=num_classes,
    )
    y_true, y_pred, probs = val_out["y_true"], val_out["y_pred"], val_out["probs"]

    per_class = R.per_class_table(y_true, y_pred, class_to_idx, label_map)
    R.save_csv(per_class, cfg.REPORTS_DIR / "02_baseline_per_class.csv")

    metrics = R.overall_metrics(y_true, y_pred, probs, num_classes)
    metrics.update(
        {
            "epochs": args.epochs,
            "n_val_images": int(len(y_true)),
            "stage": "baseline",
            "best_metric": args.best_metric,
            "best_score": best_score,
            "weak_metric": args.weak_metric,
        }
    )
    R.print_overall(metrics, f"Baseline ResNet50 ({args.epochs} epochs, ยังไม่ augment) - validation")
    R.save_json(metrics, cfg.REPORTS_DIR / "02_baseline_metrics.json")

    cm, cm_norm = R.build_confusion(y_true, y_pred, num_classes)
    np.save(cfg.REPORTS_DIR / "02_baseline_confusion_matrix.npy", cm)
    ticks = R.class_tick_labels(class_to_idx, label_map)
    R.save_csv(
        pd.DataFrame(cm, index=ticks, columns=ticks),
        cfg.REPORTS_DIR / "02_baseline_confusion_matrix.csv",
        index=True,
    )

    pairs = R.top_confused_pairs(cm, class_to_idx, label_map, top_n=10)
    R.save_csv(pairs, cfg.REPORTS_DIR / "02_baseline_top_confused_pairs.csv")

    # --- 2.5 เลือก "คลาสที่ทายผิดบ่อย" --------------------------------
    weak, info = R.select_weak_classes(per_class, rule=args.rule, metric=args.weak_metric)
    weak_payload = {
        **info,
        "weak_classes": weak,
        "weak_classes_labeled": [D.display_label(c, label_map) for c in weak],
        "baseline_recall": dict(
            zip(per_class["class_id"], per_class["recall"].round(4).astype(float))
        ),
        "baseline_f1": dict(
            zip(per_class["class_id"], per_class["f1-score"].round(4).astype(float))
        ),
    }
    R.save_json(weak_payload, cfg.REPORTS_DIR / "02_weak_classes.json")

    print(
        f"\nเกณฑ์ ({info['metric']}): เฉลี่ย {info['mean_metric']:.3f}, "
        f"SD {info['sd_metric']:.3f} -> mean-1SD = {info['threshold_mean_minus_1sd']:.3f}, "
        f"bottom20 = {info['threshold_bottom20']:.3f}"
    )
    print(f"คลาสที่ทายผิดบ่อย ({len(weak)} คลาส, rule={args.rule}):")
    for c in weak:
        row = per_class[per_class["class_id"] == c].iloc[0]
        print(
            f"  {D.display_label(c, label_map):10s} F1={row['f1-score']:.3f} "
            f"recall={row['recall']:.3f} precision={row['precision']:.3f} "
            f"support={int(row['support'])}"
        )

    if not pairs.empty:
        print("\nคู่คลาสที่สับสนกันมากที่สุด (baseline):")
        for r in pairs.itertuples():
            print(
                f"  {r.true_class} ({r.true_char}) -> {r.pred_class} ({r.pred_char}): "
                f"{r.count} ครั้ง ({r.pct_of_true_class}% ของคลาสจริง)"
            )

    # --- 2.6 กราฟ --------------------------------------------------------
    viz.setup_thai_font()
    viz.plot_confusion_matrix(
        cm,
        ticks,
        "02_baseline_confusion_matrix.png",
        f"Confusion Matrix - Baseline ResNet50 ({args.epochs} epochs, ไม่ augment) "
        f"· validation {len(y_true):,} ภาพ",
    )
    viz.plot_confusion_matrix(
        cm_norm,
        ticks,
        "02_baseline_confusion_matrix_normalized.png",
        f"Confusion Matrix - Baseline (normalized ต่อแถว %) · validation {len(y_true):,} ภาพ",
        normalized=True,
    )
    viz.plot_per_class_metric(
        per_class,
        "f1-score",
        "02_baseline_f1_per_class.png",
        "Baseline: F1 รายคลาส เรียงจากต่ำไปสูง (แดง = คลาสที่ทายผิดบ่อย)",
        threshold=info["threshold_bottom20"],
    )
    viz.plot_per_class_metric(
        per_class,
        "recall",
        "02_baseline_recall_per_class.png",
        "Baseline: recall รายคลาส เรียงจากต่ำไปสูง",
        threshold=info["threshold_bottom20"],
    )
    if len(history) > 1:
        viz.plot_learning_curves(
            history, name="02_baseline_learning_curves.png", best_metric=args.best_metric
        )

    print("\nขั้นที่ 2 เสร็จแล้ว - ต่อไปรัน run_step3_augment_plan.py")


if __name__ == "__main__":
    main()
