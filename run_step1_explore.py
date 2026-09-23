"""
run_step1_explore.py  -  ขั้นที่ 1: สำรวจและเตรียมข้อมูล
=========================================================
- ตรวจโครงสร้างโฟลเดอร์ dataset
- นับจำนวนภาพแต่ละคลาส เก็บเป็น dict {class_name: count} -> JSON
- bar chart จำนวนข้อมูลต่อคลาส เรียงจากมากไปน้อย
- สุ่มแสดงตัวอย่างภาพ 4 ภาพต่อคลาส (ไฟล์ .png ไว้ใช้ในรายงาน)
- แบ่ง test 5% ออกก่อน แล้วแบ่งที่เหลือเป็น train/val = 80/20 แบบ stratified

รัน:  uv run python run_step1_explore.py
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from torch_pipeline import config as cfg
from torch_pipeline import data as D
from torch_pipeline import reporting as R
from torch_pipeline import viz


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=str(cfg.DATA_ROOT))
    p.add_argument("--test-frac", type=float, default=cfg.TEST_FRACTION)
    p.add_argument("--val-frac", type=float, default=cfg.VAL_FRACTION)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--samples-per-class", type=int, default=4)
    p.add_argument("--skip-figures", action="store_true", help="ข้ามการวาดกราฟ (ทดสอบเร็ว)")
    args = p.parse_args()

    cfg.ensure_dirs()
    label_map = D.load_label_map()

    # --- 1.1 สำรวจโครงสร้างโฟลเดอร์ ------------------------------------
    from pathlib import Path

    data_root = Path(args.data_root)
    print(f"dataset root: {data_root}")
    df = D.scan_dataset(data_root)
    class_ids = sorted(df["class_id"].unique(), key=lambda c: int(c))
    print(f"พบ {len(class_ids)} คลาส, รวม {len(df):,} ภาพ")
    print(f"นามสกุลไฟล์ที่พบ: {sorted(set(Path(f).suffix.lower() for f in df['filepath'][:2000]))}")

    missing_labels = [c for c in class_ids if c not in label_map]
    if missing_labels:
        print(f"[warn] คลาสที่ไม่มีใน label.json: {missing_labels}")

    # --- 1.2 นับจำนวนภาพต่อคลาส ----------------------------------------
    counts = D.count_per_class(df)
    # ใช้ชื่อคลาสแบบอ่านง่าย (รหัส + อักษรไทย) เป็น key อีกชุดหนึ่งด้วย
    counts_named = {
        D.display_label(c, label_map): n for c, n in counts.items()
    }
    R.save_json(
        {
            "n_classes": len(counts),
            "n_images": int(sum(counts.values())),
            "counts_by_class_id": counts,
            "counts_by_thai_label": counts_named,
            "max_class": max(counts, key=counts.get),
            "max_count": max(counts.values()),
            "min_class": min(counts, key=counts.get),
            "min_count": min(counts.values()),
            "imbalance_ratio": round(max(counts.values()) / max(1, min(counts.values())), 1),
        },
        cfg.REPORTS_DIR / "01_class_counts.json",
    )
    print(
        f"คลาสใหญ่สุด: {D.display_label(max(counts, key=counts.get), label_map)} "
        f"= {max(counts.values()):,} ภาพ | "
        f"คลาสเล็กสุด: {D.display_label(min(counts, key=counts.get), label_map)} "
        f"= {min(counts.values()):,} ภาพ "
        f"(ต่างกัน {max(counts.values()) / max(1, min(counts.values())):.0f} เท่า)"
    )

    # --- 1.3 แบ่ง train / val / test ------------------------------------
    split_df = D.make_splits(df, args.test_frac, args.val_frac, args.seed)
    D.save_splits(split_df)
    print(f"บันทึก split -> {cfg.SPLIT_CSV}")

    summary = D.split_summary(split_df)
    R.save_csv(summary, cfg.REPORTS_DIR / "01_split_per_class.csv", index=True)

    counts_by_split = split_df["split"].value_counts()
    print("\nจำนวนภาพแต่ละ split:")
    for name in ("train", "val", "test"):
        n = int(counts_by_split.get(name, 0))
        print(f"  {name:5s} = {n:6,d} ภาพ ({n / len(split_df):.1%})")

    no_test = summary.index[summary["test"] == 0].tolist()
    if no_test:
        print(
            f"\n[หมายเหตุสำคัญสำหรับรายงาน] {len(no_test)} คลาสไม่มีภาพในชุด test "
            f"เพราะมีภาพทั้งหมดน้อยเกินไป: "
            + ", ".join(D.display_label(c, label_map) for c in no_test)
        )

    # class_to_idx ล็อกลำดับคลาสไว้ให้ทุกขั้นถัดไปใช้ชุดเดียวกัน
    class_to_idx = D.build_class_index(df)
    D.save_class_index(class_to_idx, label_map)
    print(f"บันทึก class_to_idx -> {cfg.CLASS_INDEX_JSON}")

    # --- 1.4 กราฟ --------------------------------------------------------
    if not args.skip_figures:
        viz.setup_thai_font()
        labels = {c: D.display_label(c, label_map) for c in class_ids}
        viz.plot_class_distribution(counts, labels)
        viz.plot_split_summary(summary, labels)
        viz.plot_sample_grid(df, labels, per_class=args.samples_per_class)

    print("\nขั้นที่ 1 เสร็จแล้ว - ต่อไปรัน run_step2_baseline.py")


if __name__ == "__main__":
    main()
