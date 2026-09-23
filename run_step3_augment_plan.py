"""
run_step3_augment_plan.py  -  ขั้นที่ 3: Error-driven + Balanced Augmentation
=============================================================================
สร้าง "แผน" การ augment ของ training set (เฉพาะ split train เท่านั้น)

- target count n = จำนวนภาพของคลาสที่เยอะสุด (ค่าเริ่มต้นตามสเปก = คลาส "า")
- คลาสที่ < n และอยู่ในลิสต์ "ทายผิดบ่อย" จากขั้นที่ 2 -> augment เข้มข้น
- คลาสที่ < n อื่น ๆ -> augment ปกติ
- คลาสที่ >= n -> augment เบา ๆ ตาม --light-extra-frac (ค่าเริ่มต้น 0 = ไม่เพิ่ม)
- กราฟเทียบจำนวนภาพต่อคลาส ก่อน-หลัง augment
- กราฟตัวอย่างภาพต้นฉบับ vs หลัง augment

ทำไมไม่เขียนไฟล์ภาพลงดิสก์: ดูคำอธิบายหัวไฟล์ torch_pipeline/data.py
manifest เก็บ (ภาพต้นฉบับ, ระดับความเข้ม, seed) ต่อหนึ่งภาพที่จะถูกสร้าง
จำนวนแถวใน manifest = จำนวนภาพจริงที่โมเดลเห็นต่อ epoch และ seed ถูกล็อก
ทำให้ได้ภาพชุดเดิมทุกครั้งที่รัน ต่างกับการ augment on-the-fly แบบสุ่มทั่วไป

รัน:  uv run python run_step3_augment_plan.py
      uv run python run_step3_augment_plan.py --target-count 1500   # ลดขนาด epoch
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from torch_pipeline import config as cfg
from torch_pipeline import data as D
from torch_pipeline import reporting as R
from torch_pipeline import viz

MANIFEST_PATH = cfg.SPLITS_DIR / "train_manifest.csv.gz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target-count",
        type=int,
        default=cfg.TARGET_COUNT if cfg.TARGET_COUNT else 0,
        help="0 = ใช้จำนวนของคลาสที่เยอะสุด (ตามสเปก)",
    )
    p.add_argument("--light-extra-frac", type=float, default=cfg.LIGHT_EXTRA_FRAC)
    p.add_argument(
        "--cap-majority",
        action="store_true",
        help="สุ่มลดคลาสที่มีภาพเกิน target ลงมาเท่ากับ target (มีผลเฉพาะเมื่อตั้ง "
        "--target-count ต่ำกว่าคลาสใหญ่สุด) ค่าเริ่มต้นคือไม่ทิ้งภาพจริงทิ้ง",
    )
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()

    cfg.ensure_dirs()
    label_map = D.load_label_map()
    split_df = D.load_splits()

    # --- 3.1 อ่านลิสต์คลาสที่ทายผิดบ่อยจากขั้นที่ 2 --------------------
    weak_path = cfg.REPORTS_DIR / "02_weak_classes.json"
    if weak_path.exists():
        with open(weak_path, encoding="utf-8") as f:
            weak_payload = json.load(f)
        weak_classes = {str(c) for c in weak_payload["weak_classes"]}
        print(
            f"อ่านคลาสที่ทายผิดบ่อยจาก baseline: {len(weak_classes)} คลาส "
            f"(rule = {weak_payload.get('rule')})"
        )
    else:
        weak_classes = set()
        print(
            f"[warn] ไม่พบ {weak_path} - จะ augment ทุกคลาสด้วยความเข้มปกติ\n"
            f"        รัน run_step2_baseline.py ก่อนเพื่อให้ได้ส่วน error-driven"
        )

    # --- 3.2 สร้าง manifest (เฉพาะ split train) -------------------------
    train_rows = split_df[split_df["split"] == "train"].reset_index(drop=True)
    assert "test" not in set(train_rows["split"]), "manifest ต้องไม่มีภาพจากชุด test"
    before = D.count_per_class(train_rows)

    target_used = D.resolve_target_count(train_rows, args.target_count or None)
    manifest = D.build_balanced_manifest(
        train_rows,
        weak_classes=weak_classes,
        target_count=target_used,
        light_extra_frac=args.light_extra_frac,
        seed=args.seed,
        cap_majority=args.cap_majority,
    )

    summary = D.manifest_summary(manifest)
    after = {str(c): int(n) for c, n in summary["total"].items()}
    over_target = {c: n for c, n in after.items() if n > target_used}
    if over_target:
        print(
            f"[หมายเหตุ] {len(over_target)} คลาสมีภาพจริงมากกว่า target ({target_used:,}) "
            f"และถูกเก็บไว้ทั้งหมด จึงยังไม่ balanced เป๊ะ - ใส่ --cap-majority "
            f"ถ้าต้องการสุ่มลดคลาสใหญ่ลงมาเท่ากับ target"
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False, compression="gzip", encoding="utf-8")
    print(f"บันทึก manifest ({len(manifest):,} แถว) -> {MANIFEST_PATH}")

    summary_out = summary.copy()
    summary_out.insert(0, "label", [D.display_label(str(c), label_map) for c in summary_out.index])
    summary_out.insert(1, "before", [before.get(str(c), 0) for c in summary_out.index])
    summary_out["is_weak_class"] = [str(c) in weak_classes for c in summary_out.index]
    R.save_csv(summary_out, cfg.REPORTS_DIR / "03_augmentation_plan.csv", index=True)

    # --- 3.3 สรุปตัวเลข --------------------------------------------------
    n_synth = int(len(manifest) - len(train_rows))
    stats = {
        "target_count": target_used,
        "cap_majority": args.cap_majority,
        "classes_above_target": {c: int(n) for c, n in sorted(over_target.items(), key=lambda kv: int(kv[0]))},
        "n_classes": len(after),
        "train_images_before": int(len(train_rows)),
        "train_images_after": int(len(manifest)),
        "synthesized_images": n_synth,
        "growth_factor": round(len(manifest) / max(1, len(train_rows)), 2),
        "n_weak_classes": len(weak_classes),
        "weak_classes": sorted(weak_classes, key=lambda c: int(c)),
        "aug_kind_counts": {k: int(v) for k, v in manifest["aug_kind"].value_counts().items()},
        "imbalance_before": round(max(before.values()) / max(1, min(before.values())), 1),
        "imbalance_after": round(max(after.values()) / max(1, min(after.values())), 2),
        "fixed_augmentation": cfg.FIXED_AUGMENTATION,
        "manifest_path": str(MANIFEST_PATH),
    }
    R.save_json(stats, cfg.REPORTS_DIR / "03_augmentation_stats.json")

    print("\n" + "=" * 66)
    print(f"target ต่อคลาส            = {target_used:,} ภาพ")
    print(f"train ก่อน augment        = {len(train_rows):,} ภาพ")
    print(f"train หลัง augment        = {len(manifest):,} ภาพ ({stats['growth_factor']}x)")
    print(f"ภาพที่สังเคราะห์เพิ่ม      = {n_synth:,} ภาพ")
    print(f"อัตราส่วน imbalance      = {stats['imbalance_before']} เท่า -> {stats['imbalance_after']} เท่า")
    print(f"แยกตามชนิด augment       = {stats['aug_kind_counts']}")
    print("=" * 66)

    biggest_gain = (
        summary_out.assign(added=summary_out["total"] - summary_out["before"])
        .sort_values("added", ascending=False)
        .head(5)
    )
    print("\n5 คลาสที่ถูกเติมภาพมากที่สุด:")
    for row in biggest_gain.itertuples():
        kind = "เข้มข้น" if row.is_weak_class else "ปกติ"
        print(f"  {row.label:10s} {row.before:5,d} -> {row.total:6,d} ภาพ (+{row.added:,} แบบ{kind})")

    # --- 3.4 กราฟ --------------------------------------------------------
    if not args.skip_figures:
        viz.setup_thai_font()
        labels = {c: D.display_label(c, label_map) for c in before}
        viz.plot_before_after_counts(before, after, weak_classes, labels, target=target_used)
        viz.plot_augmentation_examples(manifest, labels)

    print("\nขั้นที่ 3 เสร็จแล้ว - ต่อไปรัน run_step4_train.py")


if __name__ == "__main__":
    main()
