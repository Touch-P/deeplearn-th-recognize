"""
run_step7_finetune_v3.py  -  version 3: fine-tune ต่อด้วย manual class-weighted loss
=====================================================================================
ต่อยอดจาก checkpoint ของ version 2 (ไม่เทรนใหม่จากศูนย์) อีก 5 epochs โดยใส่
น้ำหนักต่อคลาสใน CrossEntropyLoss เจาะจงคลาสที่ confusion matrix ของ v2 ชี้ว่า
ยังสับสนกันเป็นระบบ

ที่มาของน้ำหนัก (วิเคราะห์จาก outputs/reports/05_confusion_matrix_val.csv จริง
โดยนับ error ของคลาส = จำนวนครั้งที่ถูกทายผิดตอนเป็น true label บวกกับจำนวนครั้ง
ที่ถูกทายมาผิด ๆ ตอนเป็น predicted label)

    ระดับ 3.0  า ๅ ว          - error 81 / 34 / 30 ครั้ง สามอันดับแรก
    ระดับ 2.5  ั ้            - วรรณยุกต์/สระลอยสับสนกันเอง (10 และ 6 ครั้ง)
    ระดับ 2.0  ด ต ช ซ ใ      - รูปทรงใกล้กัน (ด->ต 7, ช->ซ 6, ซ->ช 4 ครั้ง)
    ระดับ 1.5  ค น บ ท พ จ ร ข ล ส อ
    ระดับ 1.0  คลาสอื่นทั้งหมด (ค่าเริ่มต้น)

21 คลาสที่ได้น้ำหนัก > 1 ครอบคลุม 89.6% ของ error ทั้งหมดบน validation ของ v2

ข้อควรรู้ก่อนอ่านผล
-------------------
ชุด train ของ v2 balance มาแล้ว (ทุกคลาส 3,819 ภาพเท่ากัน) การใส่ class weight
ทับลงไปจึงเป็นการถ่วงน้ำหนักซ้ำรอบที่สอง ไม่ใช่การแก้ความไม่สมดุลของข้อมูล
จุดประสงค์คือ "ให้ความสำคัญกับคลาสที่ยังสับสน" เท่านั้น ส่วน error ที่เหลือบน
validation มีเพียง 168 ครั้งจาก 11,913 ภาพ (1.4%) จึงมีที่ให้ดีขึ้นไม่มาก และ
อาจเกิดการย้าย error ไปคลาสที่ไม่ได้ถ่วงน้ำหนักแทน - ตารางเทียบ before/after
ท้ายสคริปต์คือคำตอบว่าเกิดอะไรขึ้นจริง

รัน:
    python run_step7_finetune_v3.py                 # 5 epochs (~2.7 ชม. บน RTX 3050)
    python run_step7_finetune_v3.py --epochs 1 --max-steps 30   # smoke test
    python run_step7_finetune_v3.py --eval-only     # ประเมิน model_v3.pth ที่มีอยู่
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

VERSION = "v3"

MANIFEST_PATH = cfg.SPLITS_DIR / "train_manifest.csv.gz"
V2_BEST = cfg.MODELS_DIR / "best_model.pth"
V2_LAST = cfg.MODELS_DIR / "last_checkpoint.pth"
V3_BEST = cfg.MODELS_DIR / f"model_{VERSION}.pth"
V3_LAST = cfg.MODELS_DIR / f"last_checkpoint_{VERSION}.pth"
WEIGHTS_JSON = cfg.MODELS_DIR / f"class_weights_{VERSION}.json"
LOG_PATH = cfg.LOGS_DIR / f"07_finetune_{VERSION}_log.jsonl"
HISTORY_CSV = cfg.REPORTS_DIR / f"07_finetune_{VERSION}_history.csv"

# ---------------------------------------------------------------------------
# น้ำหนักต่อคลาส เขียนด้วยตัวอักษรไทยเพื่อให้อ่านรู้เรื่อง
# แล้วแปลงเป็น index จริงด้วย label.json + class_to_idx.json ของโปรเจกต์
# ---------------------------------------------------------------------------
MANUAL_CLASS_WEIGHTS: dict[str, float] = {
    # กลุ่มสับสนหนักสุด
    "า": 3.0, "ๅ": 3.0, "ว": 3.0,
    # วรรณยุกต์/สระลอยสับสนกันเอง
    "ั": 2.5, "้": 2.5,
    # รูปทรงคล้ายกัน
    "ด": 2.0, "ต": 2.0, "ช": 2.0, "ซ": 2.0, "ใ": 2.0,
    # กลุ่มที่ยังพลาดประปราย
    "ค": 1.5, "น": 1.5, "บ": 1.5, "ท": 1.5, "พ": 1.5,
    "จ": 1.5, "ร": 1.5, "ข": 1.5, "ล": 1.5, "ส": 1.5, "อ": 1.5,
}
DEFAULT_WEIGHT = 1.0


def build_weight_tensor(
    class_to_idx: dict[str, int], label_map: dict[str, str], device
) -> tuple[torch.Tensor, dict]:
    """สร้าง weight tensor ขนาด [72] ตามลำดับ index จริงของโมเดล

    ถ้ามีตัวอักษรใน MANUAL_CLASS_WEIGHTS ที่ map เป็นคลาสไม่ได้ จะ raise ทันที
    ไม่ปล่อยให้เงียบ ๆ แล้วได้ weight ผิดคลาส
    """
    char_to_class: dict[str, str] = {}
    for class_id, char in label_map.items():
        char_to_class.setdefault(char, class_id)

    weights = np.full(len(class_to_idx), DEFAULT_WEIGHT, dtype=np.float32)
    resolved: dict[str, dict] = {}
    unresolved: list[str] = []

    for char, w in MANUAL_CLASS_WEIGHTS.items():
        class_id = char_to_class.get(char)
        if class_id is None or class_id not in class_to_idx:
            unresolved.append(f"{char} (U+{ord(char[0]):04X})")
            continue
        idx = class_to_idx[class_id]
        weights[idx] = w
        resolved[char] = {"class_id": class_id, "index": idx, "weight": w}

    if unresolved:
        raise SystemExit(
            "map ตัวอักษรเป็นคลาสไม่ได้: " + ", ".join(unresolved) +
            "\nตรวจว่าตัวอักษรตรงกับใน label.json (ระวังอักขระ combining ที่หน้าตาเหมือนกัน)"
        )

    payload = {
        "version": VERSION,
        "default_weight": DEFAULT_WEIGHT,
        "n_classes": len(class_to_idx),
        "n_weighted_classes": len(resolved),
        "source": "วิเคราะห์จาก outputs/reports/05_confusion_matrix_val.csv ของ version 2",
        "rule": "error ของคลาส = ถูกทายผิดตอนเป็น true label + ถูกทายมาผิดตอนเป็น predicted label",
        "manual_class_weights_by_char": MANUAL_CLASS_WEIGHTS,
        "resolved": resolved,
        "weight_vector_by_index": {
            str(i): float(weights[i]) for i in range(len(weights))
        },
    }
    return torch.tensor(weights, device=device), payload


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    return np.divide(
        np.diag(cm).astype(float), support, out=np.full(cm.shape[0], np.nan), where=support > 0
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5, help="จำนวน epoch ที่เทรนเพิ่ม (ไม่ใช่ยอดรวม)")
    p.add_argument(
        "--from-checkpoint",
        type=str,
        default=str(V2_BEST),
        help="checkpoint ตั้งต้น (ค่าเริ่มต้น = best_model.pth ของ v2 ซึ่งเป็นตัวที่ตัวเลขใน "
        "รายงาน v2 อ้างถึง จึงเทียบ before/after กันได้ตรง ๆ)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=cfg.LEARNING_RATE / 10,
        help="ค่าเริ่มต้น = 1/10 ของ lr ตอนเทรนหลัก (1e-4 -> 1e-5)",
    )
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true", help=f"ต่อจาก {V3_LAST.name}")
    p.add_argument("--eval-only", action="store_true", help="ข้ามการเทรน ประเมิน model_v3.pth ที่มีอยู่")
    p.add_argument("--max-steps", type=int, default=0, help="จำกัด step ต่อ epoch (smoke test)")
    args = p.parse_args()

    cfg.ensure_dirs()
    cfg.set_seed(args.seed)
    device = cfg.get_device()
    print(cfg.describe_device())

    # --- 7.1 ข้อมูล + label mapping -------------------------------------
    label_map = D.load_label_map()
    split_df = D.load_splits()
    class_to_idx, _ = D.load_class_index()
    num_classes = len(class_to_idx)
    ticks = R.class_tick_labels(class_to_idx, label_map)

    val_rows = split_df[split_df["split"] == "val"]
    val_ds = D.ThaiCharDataset(val_rows, class_to_idx, A.build_eval_transform())
    val_loader = D.make_loader(val_ds, cfg.EVAL_BATCH_SIZE, False, args.num_workers, device)
    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"

    # loss สำหรับ "วัดผล" ไม่ใส่ weight เพื่อให้ val loss เทียบกับ v2 ได้ตรง ๆ
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    # --- 7.2 น้ำหนักต่อคลาส -----------------------------------------------
    weight_tensor, weight_payload = build_weight_tensor(class_to_idx, label_map, device)
    R.save_json(weight_payload, WEIGHTS_JSON)
    print(f"\nน้ำหนักต่อคลาส {weight_payload['n_weighted_classes']} คลาสที่ถูกถ่วง "
          f"(อีก {num_classes - weight_payload['n_weighted_classes']} คลาสใช้ {DEFAULT_WEIGHT})")
    by_tier: dict[float, list[str]] = {}
    for char, info in weight_payload["resolved"].items():
        by_tier.setdefault(info["weight"], []).append(f"{info['class_id']} {char}")
    for tier in sorted(by_tier, reverse=True):
        print(f"  weight {tier:>4.1f} : {'  '.join(by_tier[tier])}")

    # --- 7.3 ประเมิน "ก่อน" ด้วย checkpoint ของ v2 -----------------------
    from_path = Path(args.from_checkpoint)
    if not from_path.exists():
        raise SystemExit(f"ไม่พบ checkpoint ตั้งต้น: {from_path}")

    model = M.prepare_model(
        M.build_resnet50(num_classes, pretrained=False), device, cfg.CHANNELS_LAST_TRAIN
    )
    v2_ckpt = M.load_checkpoint(from_path, model, device)
    print(f"\nโหลด checkpoint ตั้งต้น {from_path.name} "
          f"(epoch {v2_ckpt.get('epoch')}, val macro F1 {v2_ckpt.get('val_macro_f1', float('nan')):.4f})")

    print("\nประเมิน 'ก่อน fine-tune' บน validation...")
    before = M.evaluate(
        model, val_loader, eval_criterion, device, use_amp, return_probs=True,
        desc="before", num_classes=num_classes,
    )
    cm_before, _ = R.build_confusion(before["y_true"], before["y_pred"], num_classes)
    recall_before = per_class_recall(cm_before)
    metrics_before = R.overall_metrics(before["y_true"], before["y_pred"], before["probs"], num_classes)
    print(f"  accuracy {metrics_before['accuracy']:.4f} · macro F1 {metrics_before['macro_f1']:.4f}")

    # --- 7.4 fine-tune ----------------------------------------------------
    history: list[dict] = []
    if not args.eval_only:
        if not MANIFEST_PATH.exists():
            raise SystemExit(f"ไม่พบ {MANIFEST_PATH} - รัน run_step3_augment_plan.py ก่อน")
        manifest = pd.read_csv(MANIFEST_PATH, dtype={"class_id": str}, compression="gzip")
        train_ds = D.AugmentedManifestDataset(
            manifest, class_to_idx, A.build_transforms(), fixed=cfg.FIXED_AUGMENTATION
        )
        train_loader = D.make_loader(train_ds, args.batch_size, True, args.num_workers, device)
        print(f"\ntrain (pipeline เดิมของ v2, augment แล้ว) = {len(train_ds):,} ภาพ "
              f"/ {len(train_loader):,} step ต่อ epoch")

        # loss ที่ใช้ "เทรน" ใส่ weight + คง label smoothing เดิมไว้
        train_criterion = nn.CrossEntropyLoss(
            weight=weight_tensor, label_smoothing=cfg.LABEL_SMOOTHING
        )
        # optimizer เริ่มใหม่ ไม่โหลด state ของ v2 เพราะ state นั้นผูกกับตาราง lr
        # ของรอบก่อน (cosine ที่ลงไปจนเกือบ 0 แล้ว) การเริ่มใหม่ที่ lr ต่ำเป็น
        # วิธีมาตรฐานของการ fine-tune ต่อเป็นสเตจใหม่
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        print(f"lr = {args.lr:g} (1/10 ของ {cfg.LEARNING_RATE:g}) · CosineAnnealingLR "
              f"T_max={args.epochs} · label_smoothing = {cfg.LABEL_SMOOTHING} · AMP = {use_amp}")

        start_epoch, best_score = 1, metrics_before["macro_f1"]
        print(f"เกณฑ์เซฟ: macro F1 ต้องดีกว่าจุดเริ่ม ({best_score:.4f}) จึงจะเขียน {V3_BEST.name}")
        if args.resume and V3_LAST.exists():
            rc = M.load_checkpoint(V3_LAST, model, device, optimizer, scheduler)
            start_epoch = int(rc.get("epoch", 0)) + 1
            best_score = float(rc.get("best_score", best_score))
            history = list(rc.get("history", []))
            print(f"ต่อจาก {V3_LAST.name}: เริ่ม epoch {start_epoch}")
        else:
            M.start_log(LOG_PATH)

        timer = M.EpochTimer(args.epochs)
        for epoch in range(start_epoch, args.epochs + 1):
            train_ds.set_epoch(epoch)
            timer.start()

            loader = train_loader
            if args.max_steps:
                from itertools import islice

                class _Limited:
                    def __init__(self, inner, n):
                        self.inner, self.n = inner, n

                    def __iter__(self):
                        return islice(iter(self.inner), self.n)

                    def __len__(self):
                        return self.n

                loader = _Limited(train_loader, args.max_steps)

            tr = M.train_one_epoch(
                model, loader, train_criterion, optimizer, device, scaler, epoch, args.epochs,
                num_classes=num_classes,
            )
            val_out = M.evaluate(
                model, val_loader, eval_criterion, device, use_amp, desc=f"val {epoch}",
                num_classes=num_classes,
            )
            scheduler.step()
            elapsed = timer.stop()

            record = {
                "epoch": epoch,
                "train_loss_weighted": tr["loss"],
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
                f"train loss(w) {tr['loss']:.4f} acc {tr['acc']:.4f} F1 {tr['macro_f1']:.4f}  |  "
                f"val loss {val_out['loss']:.4f} acc {val_out['acc']:.4f} F1 {val_out['macro_f1']:.4f}  |  "
                f"{M.format_eta(elapsed)} (เหลือ ~{timer.eta(epoch - start_epoch + 1)})"
            )

            extra = {
                "stage": f"finetune_{VERSION}",
                "class_to_idx": class_to_idx,
                "best_metric": "macro_f1",
                "best_score": max(best_score, val_out["macro_f1"]),
                "val_macro_f1": val_out["macro_f1"],
                "val_acc": val_out["acc"],
                "from_checkpoint": str(from_path),
                "class_weights_file": str(WEIGHTS_JSON),
                "lr": args.lr,
            }
            M.save_checkpoint(V3_LAST, model, optimizer, scheduler, epoch, val_out["acc"], history, extra=extra)
            if val_out["macro_f1"] > best_score:
                best_score = val_out["macro_f1"]
                M.save_checkpoint(V3_BEST, model, optimizer, scheduler, epoch, val_out["acc"], history, extra=extra)
                print(f"  -> val macro F1 ดีที่สุดใหม่ {best_score:.4f} เซฟไว้ที่ {V3_BEST.name}")

            pd.DataFrame(history).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

        if history:
            R.save_csv(pd.DataFrame(history), HISTORY_CSV)
            viz.setup_thai_font()
            curve_history = [
                {**h, "train_loss": h["train_loss_weighted"]} for h in history
            ]
            viz.plot_learning_curves(
                curve_history, name=f"07_finetune_{VERSION}_learning_curves.png"
            )

    # model_v3.pth ต้องมีเสมอในฐานะ "โมเดลของ version 3" แม้จะไม่ชนะ baseline
    # ถ้าไม่มี epoch ไหนทำ macro F1 ดีกว่าจุดเริ่มต้น ให้ใช้ epoch สุดท้ายเป็น
    # โมเดลของเวอร์ชันนี้ พร้อมทำเครื่องหมาย beat_baseline = False ไว้ใน
    # checkpoint เพื่อไม่ให้เข้าใจผิดว่าเป็น best checkpoint
    if not V3_BEST.exists() and V3_LAST.exists():
        ckpt = torch.load(V3_LAST, map_location="cpu", weights_only=False)
        ckpt["beat_baseline"] = False
        ckpt["baseline_macro_f1"] = float(metrics_before["macro_f1"])
        ckpt["note"] = (
            "ไม่มี epoch ใดทำ val macro F1 ได้ดีกว่า checkpoint ตั้งต้นของ v2 "
            "ไฟล์นี้คือ epoch สุดท้ายของการ fine-tune"
        )
        torch.save(ckpt, V3_BEST)
        print(f"\n[หมายเหตุ] ไม่มี epoch ใดทำ macro F1 ดีกว่าจุดเริ่มต้น "
              f"({metrics_before['macro_f1']:.4f})")
        print(f"           เซฟ epoch สุดท้ายเป็น {V3_BEST.name} "
              f"พร้อมทำเครื่องหมาย beat_baseline = False")

    # โหมด --eval-only ไม่ได้เทรนในรอบนี้ จึงต้องอ่าน history ของการ fine-tune
    # ที่เคยรันไว้กลับมา ไม่งั้นรายงานจะบอกว่าเทรน 0 epoch
    if not history and HISTORY_CSV.exists():
        history = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig").to_dict("records")
        print(f"อ่าน history ของการ fine-tune เดิมกลับมา: {len(history)} epochs")

    # --- 7.5 ประเมิน "หลัง" ------------------------------------------------
    after_path = V3_BEST if V3_BEST.exists() else V3_LAST
    if not after_path.exists():
        raise SystemExit("ไม่พบ checkpoint ของ v3 - ต้องเทรนก่อน (อย่าใช้ --eval-only ในการรันครั้งแรก)")

    eval_model = M.prepare_model(
        M.build_resnet50(num_classes, pretrained=False), device, cfg.CHANNELS_LAST_EVAL
    )
    v3_ckpt = M.load_checkpoint(after_path, eval_model, device)
    print(f"\nประเมิน 'หลัง fine-tune' ด้วย {after_path.name} (epoch {v3_ckpt.get('epoch')})...")
    after = M.evaluate(
        eval_model, val_loader, eval_criterion, device, use_amp, return_probs=True,
        desc="after", channels_last=cfg.CHANNELS_LAST_EVAL, num_classes=num_classes,
    )
    cm_after, cm_after_norm = R.build_confusion(after["y_true"], after["y_pred"], num_classes)
    recall_after = per_class_recall(cm_after)
    metrics_after = R.overall_metrics(after["y_true"], after["y_pred"], after["probs"], num_classes)

    # --- 7.6 บันทึกผล ------------------------------------------------------
    pd.DataFrame(cm_after, index=ticks, columns=ticks).to_csv(
        cfg.REPORTS_DIR / f"05_confusion_matrix_val_{VERSION}.csv", encoding="utf-8-sig"
    )
    np.save(cfg.REPORTS_DIR / f"05_confusion_matrix_val_{VERSION}.npy", cm_after)
    print(f"บันทึก confusion matrix -> 05_confusion_matrix_val_{VERSION}.csv")

    per_class_after = R.per_class_table(after["y_true"], after["y_pred"], class_to_idx, label_map)
    R.save_csv(per_class_after, cfg.REPORTS_DIR / f"05_classification_report_val_{VERSION}.csv")

    # ตารางเทียบ recall ต่อคลาส ก่อน-หลัง
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    compare = pd.DataFrame(
        {
            "class_id": [idx_to_class[i] for i in range(num_classes)],
            "char": [label_map.get(idx_to_class[i], "") for i in range(num_classes)],
            "label": ticks,
            "manual_weight": [
                float(weight_payload["weight_vector_by_index"][str(i)]) for i in range(num_classes)
            ],
            "support": cm_after.sum(axis=1),
            "recall_v2": recall_before,
            "recall_v3": recall_after,
            "errors_v2": (cm_before.sum(axis=1) - np.diag(cm_before)),
            "errors_v3": (cm_after.sum(axis=1) - np.diag(cm_after)),
        }
    )
    compare["delta_recall"] = compare["recall_v3"] - compare["recall_v2"]
    compare["is_weighted"] = compare["manual_weight"] > 1.0
    R.save_csv(
        compare.sort_values(["is_weighted", "delta_recall"], ascending=[False, True]),
        cfg.REPORTS_DIR / f"07_before_after_{VERSION}.csv",
    )

    summary = {
        "version": VERSION,
        "from_checkpoint": str(from_path),
        "v3_checkpoint": str(after_path),
        "class_weights_file": str(WEIGHTS_JSON),
        "epochs_finetuned": len(history),
        "lr": args.lr,
        "n_weighted_classes": weight_payload["n_weighted_classes"],
        "before": {k: v for k, v in metrics_before.items()},
        "after": {k: v for k, v in metrics_after.items()},
        "delta": {
            k: float(metrics_after[k] - metrics_before[k])
            for k in metrics_before
            if isinstance(metrics_before[k], float)
        },
        "weighted_classes": {
            "mean_recall_before": float(compare.loc[compare["is_weighted"], "recall_v2"].mean()),
            "mean_recall_after": float(compare.loc[compare["is_weighted"], "recall_v3"].mean()),
            "errors_before": int(compare.loc[compare["is_weighted"], "errors_v2"].sum()),
            "errors_after": int(compare.loc[compare["is_weighted"], "errors_v3"].sum()),
            "n_improved": int((compare.loc[compare["is_weighted"], "delta_recall"] > 0).sum()),
            "n_worse": int((compare.loc[compare["is_weighted"], "delta_recall"] < 0).sum()),
        },
        "other_classes": {
            "mean_recall_before": float(compare.loc[~compare["is_weighted"], "recall_v2"].mean()),
            "mean_recall_after": float(compare.loc[~compare["is_weighted"], "recall_v3"].mean()),
            "errors_before": int(compare.loc[~compare["is_weighted"], "errors_v2"].sum()),
            "errors_after": int(compare.loc[~compare["is_weighted"], "errors_v3"].sum()),
            "n_improved": int((compare.loc[~compare["is_weighted"], "delta_recall"] > 0).sum()),
            "n_worse": int((compare.loc[~compare["is_weighted"], "delta_recall"] < 0).sum()),
        },
    }
    R.save_json(summary, cfg.REPORTS_DIR / f"07_metrics_{VERSION}.json")

    # --- 7.7 กราฟ ----------------------------------------------------------
    viz.setup_thai_font()
    viz.plot_confusion_matrix(
        cm_after,
        ticks,
        f"05_confusion_matrix_val_{VERSION}.png",
        f"Confusion Matrix - version 3 (fine-tune ด้วย class-weighted loss) · "
        f"validation {int(cm_after.sum()):,} ภาพ",
    )
    viz.plot_confusion_matrix(
        cm_after_norm,
        ticks,
        f"05_confusion_matrix_val_{VERSION}_normalized.png",
        f"Confusion Matrix - version 3 (normalized ต่อแถว %) · validation {int(cm_after.sum()):,} ภาพ",
        normalized=True,
    )

    weighted_cmp = compare[compare["is_weighted"]].rename(
        columns={"recall_v2": "recall_baseline", "recall_v3": "recall_final"}
    )
    weighted_cmp["was_weak"] = True
    viz.plot_before_after_metric(
        weighted_cmp,
        metric="recall",
        name=f"07_recall_before_after_{VERSION}.png",
        title=(
            "Per-class recall ของ 21 คลาสที่ถ่วงน้ำหนัก: version 2 -> version 3\n"
            "(fine-tune 5 epochs ด้วย manual class-weighted loss · เขียว = ดีขึ้น, แดง = แย่ลง)"
        ),
        before_label="version 2 (ก่อน fine-tune)",
        after_label="version 3 (หลัง fine-tune)",
        mark_prefix="",
    )

    # --- 7.8 สรุปบนหน้าจอ ---------------------------------------------------
    print("\n" + "=" * 78)
    print("เทียบผลรวมบน validation: version 2 -> version 3")
    print("-" * 78)
    for key, name in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("top5_accuracy", "Top-5 accuracy"),
    ):
        if key in metrics_before and key in metrics_after:
            b, a = metrics_before[key], metrics_after[key]
            print(f"  {name:<20} {b:.4f} -> {a:.4f}   ({a - b:+.4f})")
    print("=" * 78)

    w = summary["weighted_classes"]
    o = summary["other_classes"]
    print(f"\n21 คลาสที่ถ่วงน้ำหนัก: recall เฉลี่ย {w['mean_recall_before']:.4f} -> "
          f"{w['mean_recall_after']:.4f} ({w['mean_recall_after'] - w['mean_recall_before']:+.4f})")
    print(f"   error รวม {w['errors_before']} -> {w['errors_after']} ครั้ง · "
          f"ดีขึ้น {w['n_improved']} คลาส / แย่ลง {w['n_worse']} คลาส")
    print(f"51 คลาสที่ไม่ถ่วง (weight 1.0): recall เฉลี่ย {o['mean_recall_before']:.4f} -> "
          f"{o['mean_recall_after']:.4f} ({o['mean_recall_after'] - o['mean_recall_before']:+.4f})")
    print(f"   error รวม {o['errors_before']} -> {o['errors_after']} ครั้ง · "
          f"ดีขึ้น {o['n_improved']} คลาส / แย่ลง {o['n_worse']} คลาส")

    print("\nคลาสที่ถ่วงน้ำหนัก เรียงตามน้ำหนัก (recall ก่อน -> หลัง)")
    print("-" * 78)
    print(f"{'คลาส':<10} {'weight':>7} {'support':>8} {'recall v2':>10} {'recall v3':>10} {'ส่วนต่าง':>10}")
    print("-" * 78)
    for _, r in compare[compare["is_weighted"]].sort_values(
        ["manual_weight", "delta_recall"], ascending=[False, False]
    ).iterrows():
        print(
            f"{r['label']:<10} {r['manual_weight']:>7.1f} {int(r['support']):>8,} "
            f"{r['recall_v2']:>10.4f} {r['recall_v3']:>10.4f} {r['delta_recall']:>+10.4f}"
        )

    print(f"\nไฟล์ที่บันทึก")
    for f in (
        after_path, WEIGHTS_JSON, HISTORY_CSV,
        cfg.REPORTS_DIR / f"05_confusion_matrix_val_{VERSION}.csv",
        cfg.REPORTS_DIR / f"05_classification_report_val_{VERSION}.csv",
        cfg.REPORTS_DIR / f"07_before_after_{VERSION}.csv",
        cfg.REPORTS_DIR / f"07_metrics_{VERSION}.json",
    ):
        print(f"  {Path(f).relative_to(cfg.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
