"""
run_step6_test_inference.py  -  ขั้นที่ 5 (ส่วนท้าย): inference บนชุด test 5%
==============================================================================
ชุด test ถูกกันไว้ตั้งแต่ขั้นที่ 1 และไม่เคยถูกใช้เทรนหรือ augment เลย
(manifest ของขั้นที่ 3 สร้างจากแถวที่ split == "train" เท่านั้น)

รูปแบบผลลัพธ์ (predictions.csv) - สเปกไม่ได้ระบุ format ของอาจารย์ไว้
จึงเขียนคอลัมน์ที่ใช้ได้กับทุกแบบที่พบทั่วไป แล้วแก้/ตัดคอลัมน์ได้ง่าย:

    filename, filepath, true_class_id, true_char,
    pred_class_id, pred_char, confidence, correct,
    top1..top3_class_id / top1..top3_char / top1..top3_prob

ถ้าอาจารย์ให้โฟลเดอร์ test มาแยก ใช้ --test-dir ชี้ไปที่โฟลเดอร์นั้นได้
รองรับทั้งแบบมีโฟลเดอร์ย่อยเป็นคลาส (จะคิด accuracy ให้) และแบบภาพกองเดียว
ไม่มี label (จะเขียนแค่ผลทำนาย)

รัน:  uv run python run_step6_test_inference.py
      uv run python run_step6_test_inference.py --test-dir "D:/test_set_อาจารย์"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_pipeline import augment as A
from torch_pipeline import config as cfg
from torch_pipeline import data as D
from torch_pipeline import model as M
from torch_pipeline import reporting as R
from torch_pipeline import viz

BEST_PATH = cfg.MODELS_DIR / "best_model.pth"


def scan_external_test_dir(test_dir: Path) -> tuple[pd.DataFrame, bool]:
    """อ่านโฟลเดอร์ test ภายนอก

    คืน (DataFrame, has_labels)
      - ถ้ามีโฟลเดอร์ย่อย -> ถือว่าชื่อโฟลเดอร์คือคลาส (has_labels=True)
      - ถ้าเป็นไฟล์ภาพกองเดียว -> ไม่มี label (has_labels=False)
    """
    subdirs = [d for d in sorted(test_dir.iterdir()) if d.is_dir()]
    rows = []
    if subdirs:
        for class_dir in subdirs:
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in cfg.VALID_EXTS:
                    rows.append(
                        {
                            "filepath": str(f.resolve()),
                            "class_id": class_dir.name,
                            "filename": f.name,
                        }
                    )
        return pd.DataFrame(rows), True

    for f in sorted(test_dir.iterdir()):
        if f.suffix.lower() in cfg.VALID_EXTS:
            rows.append(
                {"filepath": str(f.resolve()), "class_id": "unknown", "filename": f.name}
            )
    return pd.DataFrame(rows), False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=str(BEST_PATH))
    p.add_argument(
        "--test-dir",
        type=str,
        default="",
        help="โฟลเดอร์ test ภายนอก (ถ้าไม่ใส่ จะใช้ split test 5% จาก splits.csv)",
    )
    p.add_argument("--batch-size", type=int, default=cfg.EVAL_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=cfg.EVAL_NUM_WORKERS)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out-name", default="06_test_predictions.csv")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    cfg.ensure_dirs()
    cfg.set_seed()
    device = cfg.get_device()
    print(cfg.describe_device())

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"ไม่พบ checkpoint: {ckpt_path} - รัน run_step4_train.py ก่อน")

    label_map = D.load_label_map()
    class_to_idx, _ = D.load_class_index()
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # --- 6.1 ข้อมูล test -------------------------------------------------
    if args.test_dir:
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            raise SystemExit(f"ไม่พบโฟลเดอร์: {test_dir}")
        rows, has_labels = scan_external_test_dir(test_dir)
        if rows.empty:
            raise SystemExit(f"ไม่พบไฟล์ภาพใน {test_dir}")
        print(f"อ่านชุด test ภายนอก: {len(rows):,} ภาพ (มี label = {has_labels})")
        if has_labels:
            unknown = set(rows["class_id"]) - set(class_to_idx)
            if unknown:
                raise SystemExit(
                    f"ชื่อโฟลเดอร์คลาสที่ไม่รู้จัก: {sorted(unknown)}\n"
                    f"คลาสที่โมเดลรู้จักคือรหัส {min(class_to_idx, key=int)}-{max(class_to_idx, key=int)}"
                )
    else:
        split_df = D.load_splits()
        rows = split_df[split_df["split"] == "test"].reset_index(drop=True)
        has_labels = True
        print(f"ใช้ชุด test 5% จาก {cfg.SPLIT_CSV}: {len(rows):,} ภาพ")
        missing = sorted(set(class_to_idx) - set(rows["class_id"]), key=int)
        if missing:
            print(
                f"[หมายเหตุ] {len(missing)} คลาสไม่มีภาพในชุด test (ข้อมูลทั้งคลาสน้อยเกินไป): "
                + ", ".join(D.display_label(c, label_map) for c in missing)
            )

    ds = D.ThaiCharDataset(rows, class_to_idx, A.build_eval_transform(), has_labels=has_labels)
    loader = D.make_loader(ds, args.batch_size, False, args.num_workers, device)

    # --- 6.2 โหลดโมเดล + ทำนาย ------------------------------------------
    # inference ล้วน ๆ: channels_last เร็วกว่า contiguous จริงบนการ์ดนี้
    model = M.prepare_model(
        M.build_resnet50(num_classes, pretrained=False), device, cfg.CHANNELS_LAST_EVAL
    )
    ckpt = M.load_checkpoint(ckpt_path, model, device)
    print(f"โหลด {ckpt_path.name} (epoch {ckpt.get('epoch')}, val acc {ckpt.get('best_val_acc', 0):.4f})")

    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"
    out = M.evaluate(
        model, loader, None, device, use_amp, return_probs=True, desc="test",
        channels_last=cfg.CHANNELS_LAST_EVAL,
    )
    probs, row_index = out["probs"], out["row_index"]
    y_pred = probs.argmax(axis=1)

    # เรียงผลกลับตามลำดับแถวเดิมของ DataFrame (DataLoader ไม่ shuffle แต่
    # เรียงตาม row_index ให้ชัวร์ เผื่อเปลี่ยน loader ในอนาคต)
    order = np.argsort(row_index)
    probs, y_pred, row_index = probs[order], y_pred[order], row_index[order]
    ordered_rows = ds.rows.iloc[row_index].reset_index(drop=True)

    # --- 6.3 เขียนไฟล์ผลทำนาย -------------------------------------------
    topk = min(args.topk, num_classes)
    top_idx = np.argsort(probs, axis=1)[:, ::-1][:, :topk]

    result = pd.DataFrame(
        {
            "filename": ordered_rows["filename"],
            "filepath": ordered_rows["filepath"],
            "pred_class_id": [idx_to_class[int(i)] for i in y_pred],
            "pred_char": [label_map.get(idx_to_class[int(i)], "") for i in y_pred],
            "confidence": probs.max(axis=1).round(6),
        }
    )
    if has_labels:
        true_ids = ordered_rows["class_id"].astype(str)
        result.insert(2, "true_class_id", true_ids)
        result.insert(3, "true_char", [label_map.get(c, "") for c in true_ids])
        result["correct"] = result["true_class_id"] == result["pred_class_id"]

    for k in range(topk):
        ids = [idx_to_class[int(i)] for i in top_idx[:, k]]
        result[f"top{k + 1}_class_id"] = ids
        result[f"top{k + 1}_char"] = [label_map.get(c, "") for c in ids]
        result[f"top{k + 1}_prob"] = probs[np.arange(len(probs)), top_idx[:, k]].round(6)

    out_path = cfg.REPORTS_DIR / args.out_name
    R.save_csv(result, out_path)

    # --- 6.4 ถ้ามี label: คิด metric + confusion matrix ------------------
    if has_labels:
        y_true = ordered_rows["class_id"].astype(str).map(class_to_idx).to_numpy()
        metrics = R.overall_metrics(y_true, y_pred, probs, num_classes)
        per_class = R.per_class_table(y_true, y_pred, class_to_idx, label_map)
        measured = per_class[per_class["support"] > 0]
        metrics.update(
            {
                "split": "test",
                "source": args.test_dir or str(cfg.SPLIT_CSV),
                "n_images": int(len(y_true)),
                "n_classes_measured": int(len(measured)),
                "checkpoint": str(ckpt_path),
            }
        )
        R.print_overall(metrics, "ชุด test 5% (ไม่เคยถูกใช้เทรนหรือ augment)")
        R.save_json(metrics, cfg.REPORTS_DIR / "06_test_metrics.json")
        R.save_csv(per_class, cfg.REPORTS_DIR / "06_test_classification_report.csv")

        cm, cm_norm = R.build_confusion(y_true, y_pred, num_classes)
        ticks = R.class_tick_labels(class_to_idx, label_map)
        np.save(cfg.REPORTS_DIR / "06_test_confusion_matrix.npy", cm)
        R.save_csv(
            pd.DataFrame(cm, index=ticks, columns=ticks),
            cfg.REPORTS_DIR / "06_test_confusion_matrix.csv",
            index=True,
        )
        pairs = R.top_confused_pairs(cm, class_to_idx, label_map, top_n=10)
        R.save_csv(pairs, cfg.REPORTS_DIR / "06_test_top_confused_pairs.csv")

        viz.setup_thai_font()
        viz.plot_confusion_matrix(
            cm,
            ticks,
            "06_test_confusion_matrix.png",
            f"Confusion Matrix - ชุด test 5% · {len(y_true):,} ภาพ",
        )
        viz.plot_confusion_matrix(
            cm_norm,
            ticks,
            "06_test_confusion_matrix_normalized.png",
            f"Confusion Matrix - ชุด test 5% (normalized ต่อแถว %) · {len(y_true):,} ภาพ",
            normalized=True,
        )

        wrong = result[~result["correct"]]
        print(f"\nทายผิด {len(wrong):,} / {len(result):,} ภาพ")
        if not wrong.empty:
            R.save_csv(wrong, cfg.REPORTS_DIR / "06_test_misclassified.csv")
            top_wrong = (
                wrong.groupby(["true_class_id", "true_char", "pred_class_id", "pred_char"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            print("คู่ที่ทายผิดบ่อยที่สุดในชุด test:")
            for r in top_wrong.itertuples():
                print(
                    f"  {r.true_class_id} ({r.true_char}) -> {r.pred_class_id} ({r.pred_char}): "
                    f"{r.count} ภาพ"
                )
    else:
        print(f"\nชุด test ไม่มี label - เขียนเฉพาะผลทำนาย {len(result):,} แถว")
        print(f"ความเชื่อมั่นเฉลี่ย = {float(result['confidence'].mean()):.4f}")

    print(f"\nไฟล์ผลทำนาย: {out_path}")
    print("ขั้นที่ 6 เสร็จแล้ว - ครบทุกขั้นตอน")


if __name__ == "__main__":
    main()
