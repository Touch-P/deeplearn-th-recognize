"""
run_step4_train.py  -  ขั้นที่ 4: Final Training
=================================================
- สร้าง ResNet50 pretrained ImageNet ตัวใหม่ (ไม่ใช้ weight จาก baseline)
- Resize 224x224 + normalize ด้วย ImageNet mean/std
- Fine-tune ทั้งโมเดล (unfreeze ทุก layer) เหตุผลอยู่ใน
  torch_pipeline/model.py -> build_resnet50()
- Loss: CrossEntropyLoss(label_smoothing=0.1)
- Optimizer: AdamW lr 1e-4 + CosineAnnealingLR
- เทรนบน train ที่ augment แล้ว (manifest จากขั้นที่ 3) 10 epochs
  validate ทุก epoch บน val (ไม่ augment)
- เก็บ train/val loss + accuracy ทุก epoch เป็น .jsonl และ .csv
- เซฟ checkpoint ที่ val accuracy ดีที่สุด -> outputs/models/best_model.pth

รัน:  uv run python run_step4_train.py
      uv run python run_step4_train.py --resume              # ต่อจาก checkpoint ล่าสุด
      uv run python run_step4_train.py --epochs 1 --max-steps 30   # smoke test
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn as nn

from torch_pipeline import augment as A
from torch_pipeline import config as cfg
from torch_pipeline import data as D
from torch_pipeline import model as M
from torch_pipeline import reporting as R
from torch_pipeline import viz

MANIFEST_PATH = cfg.SPLITS_DIR / "train_manifest.csv.gz"
BEST_PATH = cfg.MODELS_DIR / "best_model.pth"
LAST_PATH = cfg.MODELS_DIR / "last_checkpoint.pth"
LOG_PATH = cfg.LOGS_DIR / "04_training_log.jsonl"
HISTORY_CSV = cfg.REPORTS_DIR / "04_training_history.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=cfg.FINAL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--no-amp", action="store_true", help="ปิด mixed precision")
    p.add_argument(
        "--best-metric",
        default=cfg.BEST_METRIC,
        choices=("macro_f1", "acc"),
        help="เกณฑ์เลือก best_model.pth (ค่าเริ่มต้น: macro F1 เพราะข้อมูลไม่สมดุล)",
    )
    p.add_argument("--resume", action="store_true", help="ต่อจาก last_checkpoint.pth")
    p.add_argument(
        "--fresh-aug-each-epoch",
        action="store_true",
        help="สุ่ม augment ใหม่ทุก epoch (ค่าเริ่มต้นคือใช้ชุดเดิมทุก epoch)",
    )
    p.add_argument(
        "--max-steps", type=int, default=0, help="จำกัด step ต่อ epoch (0 = ไม่จำกัด) สำหรับ smoke test"
    )
    args = p.parse_args()

    cfg.ensure_dirs()
    cfg.set_seed(args.seed)
    device = cfg.get_device()
    print(cfg.describe_device())

    # --- 4.1 ข้อมูล ------------------------------------------------------
    label_map = D.load_label_map()
    split_df = D.load_splits()
    class_to_idx, _ = D.load_class_index()
    num_classes = len(class_to_idx)

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"ไม่พบ {MANIFEST_PATH} - รัน run_step3_augment_plan.py ก่อน")
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"class_id": str}, compression="gzip")
    val_rows = split_df[split_df["split"] == "val"]

    transforms_by_kind = A.build_transforms()
    train_ds = D.AugmentedManifestDataset(
        manifest,
        class_to_idx,
        transforms_by_kind,
        fixed=not args.fresh_aug_each_epoch,
    )
    val_ds = D.ThaiCharDataset(val_rows, class_to_idx, A.build_eval_transform())

    train_loader = D.make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = D.make_loader(val_ds, cfg.EVAL_BATCH_SIZE, False, args.num_workers, device)

    steps_per_epoch = len(train_loader) if not args.max_steps else min(args.max_steps, len(train_loader))
    print(
        f"train (augment แล้ว) = {len(train_ds):,} ภาพ / {steps_per_epoch:,} step ต่อ epoch "
        f"(batch {args.batch_size})"
    )
    print(f"val (ไม่ augment)    = {len(val_ds):,} ภาพ")

    # --- 4.2 โมเดล / loss / optimizer / scheduler -----------------------
    model = M.prepare_model(
        M.build_resnet50(num_classes, pretrained=True), device, cfg.CHANNELS_LAST_TRAIN
    )
    total, trainable = M.count_parameters(model)
    print(f"ResNet50 fine-tune ทั้งโมเดล: เทรนได้ {trainable:,} / {total:,} พารามิเตอร์")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # CosineAnnealingLR: lr ลดจาก 1e-4 ลงเป็นเกือบ 0 แบบโคไซน์ภายใน T_max epoch
    # ช่วงท้ายที่ lr ต่ำมากทำให้โมเดลลงหลุมที่แคบกว่าเดิมได้ - ได้ผลดีกับงาน
    # fine-tune ที่จำนวน epoch กำหนดไว้ล่วงหน้าแบบนี้
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP (mixed precision) = {use_amp}")

    start_epoch, best_score, history = 1, -1.0, []
    if args.resume and LAST_PATH.exists():
        ckpt = M.load_checkpoint(LAST_PATH, model, device, optimizer, scheduler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        history = list(ckpt.get("history", []))
        print(
            f"ต่อจาก checkpoint: เริ่ม epoch {start_epoch}, "
            f"best {args.best_metric} = {best_score:.4f}"
        )

    # --- 4.3 วนเทรน ------------------------------------------------------
    if not args.resume:
        M.start_log(LOG_PATH)  # เริ่ม log ใหม่ ไม่ให้ปนกับรอบก่อน
    timer = M.EpochTimer(args.epochs)
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)  # มีผลเฉพาะโหมด --fresh-aug-each-epoch
        timer.start()

        if args.max_steps:
            # smoke test: ตัดจำนวน step ต่อ epoch โดยห่อ loader ด้วย islice
            from itertools import islice

            class _Limited:
                def __init__(self, loader, n):
                    self.loader, self.n = loader, n

                def __iter__(self):
                    return islice(iter(self.loader), self.n)

                def __len__(self):
                    return self.n

            loader = _Limited(train_loader, args.max_steps)
        else:
            loader = train_loader

        tr = M.train_one_epoch(
            model, loader, criterion, optimizer, device, scaler, epoch, args.epochs,
            num_classes=num_classes,
        )
        val_out = M.evaluate(
            model, val_loader, criterion, device, use_amp, desc=f"val {epoch}",
            num_classes=num_classes,
        )
        scheduler.step()
        elapsed = timer.stop()

        record = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": val_out["loss"],
            "val_acc": val_out["acc"],
            "val_macro_f1": val_out["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(elapsed, 1),
        }
        history.append(record)
        M.append_log(LOG_PATH, record)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} F1 {tr['macro_f1']:.4f}  |  "
            f"val loss {val_out['loss']:.4f} acc {val_out['acc']:.4f} F1 {val_out['macro_f1']:.4f}  |  "
            f"{M.format_eta(elapsed)} (เหลือ ~{timer.eta(epoch - start_epoch + 1)})"
        )

        # เกณฑ์ best = macro F1 (ไม่ใช่ accuracy) เพราะคลาสใหญ่สุดมีภาพมากกว่า
        # คลาสเล็กสุดหลายพันเท่า accuracy จะเลือก checkpoint ที่เก่งแต่คลาสใหญ่
        score = val_out["macro_f1"] if args.best_metric == "macro_f1" else val_out["acc"]
        ckpt_extra = {
            "stage": "final",
            "class_to_idx": class_to_idx,
            "best_metric": args.best_metric,
            "best_score": max(best_score, score),
            "val_macro_f1": val_out["macro_f1"],
            "val_acc": val_out["acc"],
        }
        # เซฟทุก epoch เพื่อให้ --resume ได้ถ้าเทรนหลุด
        M.save_checkpoint(
            LAST_PATH, model, optimizer, scheduler, epoch, val_out["acc"], history,
            extra=ckpt_extra,
        )
        if score > best_score:
            best_score = score
            M.save_checkpoint(
                BEST_PATH, model, optimizer, scheduler, epoch, val_out["acc"], history,
                extra=ckpt_extra,
            )
            print(
                f"  -> val {args.best_metric} ดีที่สุดใหม่ {best_score:.4f} "
                f"(acc {val_out['acc']:.4f}) เซฟไว้ที่ {BEST_PATH.name}"
            )

        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

    # --- 4.4 สรุป + learning curve --------------------------------------
    R.save_csv(pd.DataFrame(history), HISTORY_CSV)
    best_record = max(history, key=lambda h: h["val_macro_f1"]) if history else None
    R.save_json(
        {
            "epochs_run": len(history),
            "best_metric": args.best_metric,
            "best_score": best_score,
            "best_epoch": best_record["epoch"] if best_record else None,
            "best_val_macro_f1": best_record["val_macro_f1"] if best_record else None,
            "best_val_acc": best_record["val_acc"] if best_record else None,
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_train_macro_f1": history[-1]["train_macro_f1"] if history else None,
            "final_val_acc": history[-1]["val_acc"] if history else None,
            "final_val_macro_f1": history[-1]["val_macro_f1"] if history else None,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "amp": use_amp,
            "train_images_per_epoch": len(train_ds),
            "best_model_path": str(BEST_PATH),
        },
        cfg.REPORTS_DIR / "04_training_summary.json",
    )

    if history:
        viz.setup_thai_font()
        viz.plot_learning_curves(history, best_metric=args.best_metric)
        print(f"\nval {args.best_metric} ดีที่สุด = {best_score:.4f}")
        if best_record:
            print(
                f"  epoch {best_record['epoch']}: macro F1 {best_record['val_macro_f1']:.4f}, "
                f"accuracy {best_record['val_acc']:.4f}"
            )
        print(f"โมเดลที่ดีที่สุดอยู่ที่ {BEST_PATH}")

    print("\nขั้นที่ 4 เสร็จแล้ว - ต่อไปรัน run_step5_evaluate.py")


if __name__ == "__main__":
    main()
