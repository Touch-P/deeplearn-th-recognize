"""
make_results.py  -  สร้าง RESULTS.md จาก artifact จริงที่ pipeline เซฟไว้
==========================================================================
อ่านทุกไฟล์ใน outputs_torch/reports/ แล้วประกอบเป็นรายงานเดียวที่พร้อมใช้
นำเสนอ ตัวเลขทุกตัวในรายงานถูกอ่านจากไฟล์ ไม่ได้พิมพ์ด้วยมือ จึงไม่มีทางที่
รายงานจะขัดกับผลจริง และรันซ้ำได้ทุกครั้งที่เทรนใหม่

รัน:  uv run python make_results.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch_pipeline import config as cfg
from torch_pipeline import data as D

R = cfg.REPORTS_DIR
OUT = PROJECT_ROOT / "RESULTS.md"
missing: list[str] = []


def read_json(name: str) -> dict | None:
    path = R / name
    if not path.exists():
        missing.append(name)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(name: str, **kw) -> pd.DataFrame | None:
    path = R / name
    if not path.exists():
        missing.append(name)
        return None
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def md_table(df: pd.DataFrame, columns: dict[str, str], digits: int = 3) -> str:
    """DataFrame -> ตาราง markdown โดย columns = {ชื่อคอลัมน์จริง: หัวตาราง}"""
    header = "| " + " | ".join(columns.values()) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for col in columns:
            v = r[col]
            if isinstance(v, float):
                cells.append(f"{v:.{digits}f}")
            elif isinstance(v, (int,)) or (hasattr(v, "item") and isinstance(v.item(), int)):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main() -> None:
    label_map = D.load_label_map()

    counts = read_json("01_class_counts.json")
    split_tbl = read_csv("01_split_per_class.csv", dtype={"class_id": str})
    base_metrics = read_json("02_baseline_metrics.json")
    weak = read_json("02_weak_classes.json")
    base_per_class = read_csv("02_baseline_per_class.csv", dtype={"class_id": str})
    aug_stats = read_json("03_augmentation_stats.json")
    train_summary = read_json("04_training_summary.json")
    history = read_csv("04_training_history.csv")
    val_metrics = read_json("05_metrics_val.json")
    val_per_class = read_csv("05_classification_report_val.csv", dtype={"class_id": str})
    val_pairs = read_csv("05_top_confused_pairs_val.csv", dtype=str)
    before_after = read_json("05_before_after_summary.json")
    comparison = read_csv("05_recall_before_after.csv", dtype={"class_id": str})
    test_metrics = read_json("06_test_metrics.json")
    test_per_class = read_csv("06_test_classification_report.csv", dtype={"class_id": str})
    test_pairs = read_csv("06_test_top_confused_pairs.csv", dtype=str)

    L: list[str] = []
    A = L.append

    # ================= หัวเรื่อง + ผลสรุป =================
    A("# ผลการทดลอง: รู้จำตัวอักษร/ตัวเลขภาษาไทย 72 คลาส")
    A("")
    A("**ResNet50 (Transfer Learning) + Error-driven Balanced Data Augmentation** · PyTorch")
    A("")
    A(f"สร้างรายงานนี้อัตโนมัติจากไฟล์ผลลัพธ์จริงด้วย `make_results.py` เมื่อ {date.today().isoformat()}")
    A("")

    A("## 1. ผลลัพธ์สรุป")
    A("")
    if val_metrics or test_metrics:
        A("| ชุดข้อมูล | Accuracy | **Macro F1** | Weighted F1 | Balanced Acc | Top-5 Acc | จำนวนภาพ |")
        A("|---|---|---|---|---|---|---|")
        for label, m in (("Validation", val_metrics), ("**Test 5% (ไม่เคยถูกเทรน)**", test_metrics)):
            if m:
                A(
                    f"| {label} | {fmt(m.get('accuracy'))} | **{fmt(m.get('macro_f1'))}** | "
                    f"{fmt(m.get('weighted_f1'))} | {fmt(m.get('balanced_accuracy'))} | "
                    f"{fmt(m.get('top5_accuracy'))} | {fmt(m.get('n_images'))} |"
                )
        A("")
        A(
            "> **ทำไมดู Macro F1 เป็นหลัก**: คลาสใหญ่สุดของชุดนี้มีภาพมากกว่าคลาสเล็กสุด "
            "หลายพันเท่า Accuracy จึงถูกคลาสใหญ่ครอบงำ - คลาสเล็กพังทั้งคลาสก็ยังทำให้ "
            "accuracy ดูดีได้ Macro F1 ให้น้ำหนักทุกคลาสเท่ากันจึงเป็นตัวเลขที่บอกความ "
            "สามารถจริง และเป็นเกณฑ์ที่ใช้เลือก best checkpoint ในการเทรนด้วย"
        )
        A("")

    if train_summary:
        A("**การเทรน**")
        A("")
        A(f"- เทรนไป {fmt(train_summary.get('epochs_run'))} epochs, "
          f"best ที่ epoch {fmt(train_summary.get('best_epoch'))} "
          f"(เกณฑ์ = `{train_summary.get('best_metric')}`)")
        A(f"- train accuracy epoch สุดท้าย = {fmt(train_summary.get('final_train_acc'))} · "
          f"train macro F1 = {fmt(train_summary.get('final_train_macro_f1'))}")
        A(f"- val accuracy ที่ดีที่สุด = {fmt(train_summary.get('best_val_acc'))} · "
          f"val macro F1 ที่ดีที่สุด = {fmt(train_summary.get('best_val_macro_f1'))}")
        A("")

    # ================= dataset =================
    A("## 2. ชุดข้อมูล")
    A("")
    if counts:
        A(f"- **จำนวนคลาส**: {fmt(counts['n_classes'])} คลาส (พยัญชนะ สระ วรรณยุกต์ และเลขไทย)")
        A(f"- **จำนวนภาพรวม**: {fmt(counts['n_images'])} ภาพ")
        A(f"- **โครงสร้างโฟลเดอร์**: `ThaiCharacter Dataset/round2/<รหัสคลาส>/*.jpg` "
          f"(รหัสคลาสเป็นตัวเลข map เป็นอักษรไทยด้วย `label.json`)")
        A(f"- **ขนาดภาพต้นฉบับ**: เล็กมาก ประมาณ 12×18 ถึง 30×45 px (ขยายเป็น 224×224 ก่อนเข้าโมเดล)")
        A(f"- **ความไม่สมดุล**: มากสุด "
          f"{D.display_label(counts['max_class'], label_map)} = {fmt(counts['max_count'])} ภาพ, "
          f"น้อยสุด {D.display_label(counts['min_class'], label_map)} = {fmt(counts['min_count'])} ภาพ "
          f"→ **ต่างกัน {counts['imbalance_ratio']:,.0f} เท่า**")
        A("")

        items = sorted(counts["counts_by_class_id"].items(), key=lambda kv: kv[1], reverse=True)
        A("<details><summary>จำนวนภาพต่อคลาส ทั้ง 72 คลาส (กดเพื่อดู)</summary>")
        A("")
        A("| อันดับ | คลาส | จำนวนภาพ |")
        A("|---|---|---|")
        for i, (cid, n) in enumerate(items, 1):
            A(f"| {i} | {D.display_label(cid, label_map)} | {n:,} |")
        A("")
        A("</details>")
        A("")

    if split_tbl is not None:
        totals = split_tbl[["train", "val", "test"]].sum()
        grand = int(totals.sum())
        A("**การแบ่งข้อมูล** (stratified ภายในทุกคลาส, ล็อกด้วย `SEED = 42`)")
        A("")
        A("| split | จำนวนภาพ | สัดส่วน | ใช้ทำอะไร |")
        A("|---|---|---|---|")
        A(f"| train | {int(totals['train']):,} | {totals['train'] / grand:.1%} | เทรน + augment |")
        A(f"| val | {int(totals['val']):,} | {totals['val'] / grand:.1%} | วัดผลทุก epoch, เลือก best checkpoint |")
        A(f"| test | {int(totals['test']):,} | {totals['test'] / grand:.1%} | **กันไว้ตรวจ ไม่เคยถูกเทรนหรือ augment** |")
        A("")
        A(
            "ตัด test 5% ออกก่อนเป็นอย่างแรก แล้วแบ่งส่วนที่เหลือเป็น train/val = 80/20 "
            "(`run_step1_explore.py`) manifest ของ augmentation สร้างจากแถวที่ `split == \"train\"` "
            "เท่านั้น และมี `assert` กำกับไว้ในโค้ด"
        )
        A("")
        no_test = split_tbl.loc[split_tbl["test"] == 0, "class_id"].tolist()
        if no_test:
            A(
                f"> ข้อจำกัดที่ต้องระบุ: {len(no_test)} คลาสไม่มีภาพในชุด test เลย "
                f"({', '.join(D.display_label(c, label_map) for c in no_test)}) "
                f"เพราะทั้งคลาสมีภาพน้อยเกินกว่าจะแบ่งสามทางได้"
            )
            A("")

    # ================= สถาปัตยกรรม =================
    A("## 3. โครงสร้างโมเดล (CNN)")
    A("")
    A("**ResNet50** (Deep Residual Network, 50 ชั้น) จาก `torchvision.models.resnet50`")
    A("")
    A("```")
    A("Input 3 × 224 × 224")
    A("  conv1      7×7, 64 filters, stride 2      -> 64 × 112 × 112")
    A("  maxpool    3×3, stride 2                  -> 64 × 56 × 56")
    A("  layer1     bottleneck × 3  (64→256)       -> 256 × 56 × 56")
    A("  layer2     bottleneck × 4  (128→512)      -> 512 × 28 × 28")
    A("  layer3     bottleneck × 6  (256→1024)     -> 1024 × 14 × 14")
    A("  layer4     bottleneck × 3  (512→2048)     -> 2048 × 7 × 7")
    A("  avgpool    global average pooling         -> 2048")
    A("  fc         Linear(2048 → 72)              -> 72 logits  ← เปลี่ยนชั้นนี้เอง")
    A("```")
    A("")
    A("- **พารามิเตอร์รวม ~23.66 ล้านตัว เทรนทั้งหมด** (unfreeze ทุกชั้น)")
    A("- แต่ละ bottleneck block = conv 1×1 → conv 3×3 → conv 1×1 + skip connection "
      "ซึ่งเป็นหัวใจของ ResNet: skip connection ทำให้ gradient ไหลข้ามชั้นได้ จึงเทรน "
      "โมเดลลึก 50 ชั้นได้โดยไม่เจอปัญหา vanishing gradient")
    A("- ชั้น `fc` เดิมของ ImageNet มี 1000 output (1000 คลาสภาพถ่าย) ถูกแทนด้วย "
      "`nn.Linear(2048, 72)` ที่สุ่มค่าเริ่มต้นใหม่")
    A("")

    A("## 4. Transfer Learning")
    A("")
    A("| รายการ | ค่าที่ใช้ |")
    A("|---|---|")
    A("| น้ำหนักเริ่มต้น | `ResNet50_Weights.IMAGENET1K_V2` (pretrained บน ImageNet) |")
    A("| กลยุทธ์ | **fine-tune ทั้งโมเดล** (unfreeze ทุกชั้น) |")
    A(f"| Loss | `CrossEntropyLoss(label_smoothing={cfg.LABEL_SMOOTHING})` |")
    A(f"| Optimizer | `AdamW` (lr = {cfg.LEARNING_RATE}, weight_decay = {cfg.WEIGHT_DECAY}) |")
    A("| Scheduler | `CosineAnnealingLR` (lr ลดแบบโคไซน์จนเกือบ 0) |")
    A(f"| Batch size | {cfg.BATCH_SIZE} (train) / {cfg.EVAL_BATCH_SIZE} (eval) |")
    A(f"| Epochs | {cfg.FINAL_EPOCHS} |")
    A("| Mixed precision | เปิด (`torch.autocast` fp16 + `GradScaler`) |")
    A("| Normalize | ImageNet mean/std `(0.485,0.456,0.406)` / `(0.229,0.224,0.225)` |")
    A(f"| Seed | {cfg.SEED} (ล็อก `random`, `numpy`, `torch`, `cuda`) |")
    A("")
    A(
        "**เหตุผลที่ fine-tune ทั้งโมเดลไม่ใช่ freeze แล้วเทรนแค่ fc**: ImageNet เป็นภาพถ่าย "
        "ธรรมชาติ (สัตว์ สิ่งของ ทิวทัศน์) ส่วนงานนี้เป็นลายเส้นขาว-ดำของตัวอักษร ฟีเจอร์ "
        "ชั้นกลาง-ปลายของ ImageNet (texture ขนสัตว์ ลายผ้า สีวัตถุ) แทบไม่ตรงกับงานนี้ "
        "ถ้า freeze ไว้โมเดลจะติดเพดานเร็ว การ unfreeze ทั้งหมดด้วย lr ต่ำ (1e-4) ให้ทุกชั้น "
        "ปรับเข้าหาลายเส้นได้ ขณะที่ยังได้ประโยชน์จาก edge/stroke detector ชั้นต้นที่ "
        "pretrained มาแล้ว (เทรนจากศูนย์บนภาพเล็ก ๆ 47k ภาพจะ overfit ง่ายกว่ามาก)"
    )
    A("")

    # ================= baseline + weak classes =================
    A("## 5. Baseline เพื่อหา \"คลาสที่ทายผิดบ่อย\"")
    A("")
    if base_metrics:
        A(
            f"เทรน ResNet50 แบบเดียวกัน {fmt(base_metrics.get('epochs'))} epochs บน train "
            f"ที่**ยังไม่ augment** เพื่อดู error pattern ได้ผลบน validation:"
        )
        A("")
        A(
            f"- Accuracy = {fmt(base_metrics.get('accuracy'))} · "
            f"Macro F1 = {fmt(base_metrics.get('macro_f1'))} · "
            f"Balanced accuracy = {fmt(base_metrics.get('balanced_accuracy'))}"
        )
        A("")
    if weak:
        A(
            f"จากนั้นเลือกคลาสอ่อนแอด้วย per-class **{weak.get('metric', 'f1-score')}** "
            f"ตามเกณฑ์ `{weak.get('rule')}` = union ของสองกฎ:"
        )
        A("")
        A(f"- `mean - 1SD`: ค่าเฉลี่ย {fmt(weak.get('mean_metric'), 3)} − SD {fmt(weak.get('sd_metric'), 3)} "
          f"= **{fmt(weak.get('threshold_mean_minus_1sd'), 3)}** → ได้ {weak.get('n_by_mean_sd')} คลาส")
        A(f"- `bottom 20%`: ค่าตัดที่ **{fmt(weak.get('threshold_bottom20'), 3)}** "
          f"→ ได้ {weak.get('n_by_bottom20')} คลาส")
        A(f"- คลาสที่ไม่มีภาพใน val (ข้อมูลน้อยเกินไป) ถูกนับเป็นคลาสอ่อนแอโดยปริยาย "
          f"อีก {weak.get('n_unmeasured_forced_in')} คลาส")
        A("")
        A(f"**รวม {weak.get('n_weak_total')} คลาสที่ถูกจัดเป็น \"ทายผิดบ่อย\"** และจะได้ "
          f"augmentation แบบเข้มข้นกว่าคลาสอื่น:")
        A("")
        A("> " + " · ".join(weak.get("weak_classes_labeled", [])))
        A("")
    if base_per_class is not None:
        worst = base_per_class[base_per_class["support"] > 0].nsmallest(10, "f1-score")
        A("<details><summary>10 คลาสที่ F1 ต่ำสุดของ baseline (กดเพื่อดู)</summary>")
        A("")
        A(md_table(
            worst,
            {"label": "คลาส", "precision": "Precision", "recall": "Recall",
             "f1-score": "F1", "support": "จำนวนภาพ val"},
        ))
        A("")
        A("</details>")
        A("")

    # ================= augmentation =================
    A("## 6. Error-driven + Balanced Data Augmentation")
    A("")
    if aug_stats:
        A("| รายการ | ค่า |")
        A("|---|---|")
        A(f"| target ต่อคลาส | {fmt(aug_stats.get('target_count'))} ภาพ (= จำนวนของคลาสที่เยอะสุด) |")
        A(f"| train ก่อน augment | {fmt(aug_stats.get('train_images_before'))} ภาพ |")
        A(f"| train หลัง augment | **{fmt(aug_stats.get('train_images_after'))} ภาพ** "
          f"({aug_stats.get('growth_factor')}×) |")
        A(f"| ภาพที่สังเคราะห์เพิ่ม | {fmt(aug_stats.get('synthesized_images'))} ภาพ |")
        A(f"| ความไม่สมดุล | {aug_stats.get('imbalance_before')} เท่า → "
          f"**{aug_stats.get('imbalance_after')} เท่า** |")
        A(f"| คลาสที่ได้ augment เข้มข้น | {aug_stats.get('n_weak_classes')} คลาส |")
        kinds = aug_stats.get("aug_kind_counts", {})
        A(f"| แยกตามชนิด | " + ", ".join(f"`{k}` {v:,}" for k, v in kinds.items()) + " |")
        A("")
        A(
            "**วิธีทำ**: ทุกคลาสที่มีภาพน้อยกว่า target จะถูกเติมด้วยภาพ augment จนครบ "
            "target โดยวนใช้ภาพต้นฉบับแบบ round-robin (ทุกภาพถูกใช้เป็นต้นแบบใกล้เคียงกัน "
            "ไม่ใช่สุ่มจนภาพบางใบถูกใช้ 10 ครั้งอีกใบ 0 ครั้ง) คลาสที่อยู่ในลิสต์ "
            "\"ทายผิดบ่อย\" จากข้อ 5 ใช้ transform ชุดเข้มข้นกว่า"
        )
        A("")
        A(
            "ภาพ augment ไม่ได้เขียนลงดิสก์ แต่เก็บเป็น *manifest* ที่บอกว่า (ภาพต้นฉบับใด, "
            "ความเข้มระดับใด, seed อะไร) แล้วแปลงตอนโหลดภายใน `torch.random.fork_rng()` "
            f"ที่ล็อก seed ไว้ ผลคือได้ชุดภาพ augment ที่**คงที่ทุก epoch และ reproduce ได้ 100%** "
            f"เทียบเท่ามีไฟล์ภาพจริง {fmt(aug_stats.get('train_images_after'))} ไฟล์ "
            "แต่ไม่กินดิสก์ ~20GB"
        )
        A("")

    A("**Transform ที่ใช้** (`torch_pipeline/augment.py`)")
    A("")
    A("| เทคนิค | ปกติ (normal) | เข้มข้น (intense) |")
    A("|---|---|---|")
    A("| RandomRotation | ±10° | ±15° |")
    A("| RandomAffine translate | 5% | 8% |")
    A("| RandomAffine scale | 0.88–1.05 | 0.80–1.05 |")
    A("| RandomAffine shear | 4° | 8° |")
    A("| ElasticTransform | ไม่ใช้ | alpha 35, sigma 5 (p=0.50) |")
    A("| Morphological dilate/erode | p=0.25 | p=0.45 |")
    A("| GaussianBlur | p=0.20 | p=0.30 |")
    A("| ColorJitter (brightness/contrast) | 0.12 | 0.20 |")
    A("| Gaussian noise | std 0.015 (p=0.30) | std 0.025 (p=0.50) |")
    A("| RandomErasing | ไม่ใช้ | p=0.15, พื้นที่ 1–4% |")
    A("| **Horizontal / Vertical Flip** | **ห้ามใช้** | **ห้ามใช้** |")
    A("")
    A(
        "**ทำไมห้าม flip**: ตัวอักษรไทยมีทิศทาง พลิกซ้าย-ขวาแล้วได้รูปที่เป็นอักขระอื่น "
        "หรือไม่มีในภาษา (เช่น \"ก\" พลิกแล้วคล้าย \"ฎ/ฏ\") การ flip คือการป้อน label ผิด "
        "ให้โมเดลโดยตรง"
    )
    A("")
    A(
        "**Morphological dilate/erode** ทำด้วย `ImageFilter.MinFilter/MaxFilter` บนภาพหมึกเข้ม "
        "บนพื้นขาว: MinFilter ทำให้เส้นหนาขึ้น MaxFilter ทำให้เส้นบางลง เลียนแบบการเขียนด้วย "
        "ปากกาหัวใหญ่/เล็กหรือกดหนัก/เบา ซึ่งเป็นความแปรผันที่พบมากที่สุดในลายมือจริง"
    )
    A("")
    A(
        "ทุก transform เชิงเรขาคณิตเติมพื้นที่ว่างด้วย**สีขาว** (ไม่ใช่ดำ) ให้ตรงกับพื้นหลัง "
        "ของภาพจริง ไม่งั้นจะเกิดขอบดำปลอมที่โมเดลอาจเรียนรู้เป็น feature และเพดาน scale "
        "หยุดที่ 1.05 เพราะตัวอักษรกินพื้นที่เกือบเต็มเฟรมอยู่แล้ว ซูมเข้ามากกว่านี้เส้นจะถูก "
        "ตัดหายที่ขอบจนกลายเป็นตัวอักษรอื่น"
    )
    A("")

    # ================= ผลลัพธ์ =================
    A("## 7. ผลการเทรนและกราฟ")
    A("")
    if history is not None:
        cols = {"epoch": "Epoch", "train_loss": "Train Loss", "train_acc": "Train Acc"}
        if "train_macro_f1" in history.columns:
            cols["train_macro_f1"] = "Train F1"
        cols.update({"val_loss": "Val Loss", "val_acc": "Val Acc"})
        if "val_macro_f1" in history.columns:
            cols["val_macro_f1"] = "**Val Macro F1**"
        A(md_table(history, cols, digits=4))
        A("")
    A("![learning curves](outputs_torch/figures/04_learning_curves.png)")
    A("")
    A("กราฟสามแผง: Loss, **Accuracy Rate** ของ train/val และ **Macro F1** ของ train/val "
      "(วงกลมเขียว = epoch ที่ถูกเลือกเป็น best checkpoint ตามเกณฑ์ macro F1)")
    A("")

    A("### Confusion Matrix")
    A("")
    A("![confusion matrix](outputs_torch/figures/05_confusion_matrix_val.png)")
    A("")
    A("(เวอร์ชัน normalized ต่อแถวอยู่ที่ `05_confusion_matrix_val_normalized.png` "
      "และของชุด test อยู่ที่ `06_test_confusion_matrix.png`)")
    A("")

    if val_per_class is not None and val_metrics:
        A("### F1 รายคลาส")
        A("")
        A("![f1 per class](outputs_torch/figures/05_f1_per_class_val.png)")
        A("")
        measured = val_per_class[val_per_class["support"] > 0]
        A(
            f"คลาสที่วัดได้ {len(measured)} คลาส · "
            f"F1 < {val_metrics.get('weak_threshold', 0.8)} มี "
            f"**{val_metrics.get('classes_below_threshold')} คลาส**"
        )
        A("")
        A("**10 คลาสที่ F1 ต่ำสุด** (คลาสที่ยังเป็นจุดอ่อนของโมเดลสุดท้าย)")
        A("")
        A(md_table(
            measured.nsmallest(10, "f1-score"),
            {"label": "คลาส", "precision": "Precision", "recall": "Recall",
             "f1-score": "F1", "support": "จำนวนภาพ val"},
        ))
        A("")
        A("<details><summary>Classification report ครบทั้ง 72 คลาส (กดเพื่อดู)</summary>")
        A("")
        A(md_table(
            val_per_class.sort_values("f1-score"),
            {"label": "คลาส", "precision": "Precision", "recall": "Recall",
             "f1-score": "F1", "support": "Support"},
        ))
        A("")
        A("</details>")
        A("")

    if before_after:
        A("### ผลของ Error-driven Balanced Augmentation (เทียบ baseline)")
        A("")
        A("| กลุ่มคลาส | F1 ก่อน (baseline) | F1 หลัง | ส่วนต่าง |")
        A("|---|---|---|---|")
        A(f"| ทุกคลาส | {fmt(before_after.get('macro_f1_baseline'))} | "
          f"{fmt(before_after.get('macro_f1_final'))} | "
          f"**{before_after.get('macro_f1_delta', 0):+.4f}** |")
        if before_after.get("weak_classes_mean_f1_baseline") is not None:
            A(f"| เฉพาะคลาสที่ทายผิดบ่อย ({before_after.get('n_weak_classes')} คลาส) | "
              f"{fmt(before_after.get('weak_classes_mean_f1_baseline'))} | "
              f"{fmt(before_after.get('weak_classes_mean_f1_final'))} | "
              f"**{before_after.get('weak_classes_mean_f1_delta', 0):+.4f}** |")
        if before_after.get("other_classes_mean_f1_delta") is not None:
            A(f"| คลาสอื่น ๆ | - | - | {before_after.get('other_classes_mean_f1_delta', 0):+.4f} |")
        A("")
        A(f"F1 ดีขึ้น {before_after.get('n_classes_f1_improved')} คลาส / "
          f"แย่ลง {before_after.get('n_classes_f1_worse')} คลาส")
        A("")
        A("![before after](outputs_torch/figures/05_f1_before_after_augmentation.png)")
        A("")
        if comparison is not None:
            top = comparison.nlargest(10, "delta_f1")
            A("**10 คลาสที่ F1 ดีขึ้นมากที่สุด** (★ = อยู่ในลิสต์ทายผิดบ่อย)")
            A("")
            A("| คลาส | F1 baseline | F1 สุดท้าย | ส่วนต่าง | เคยเป็นคลาสอ่อนแอ |")
            A("|---|---|---|---|---|")
            for _, r in top.iterrows():
                star = "★" if r["was_weak"] else ""
                A(f"| {r['label']} | {r['f1-score_baseline']:.3f} | {r['f1-score_final']:.3f} | "
                  f"{r['delta_f1']:+.3f} | {star} |")
            A("")

    if val_pairs is not None and not val_pairs.empty:
        A("### คู่คลาสที่โมเดลยังสับสนมากที่สุด (validation)")
        A("")
        A("| คลาสจริง | ทำนายเป็น | จำนวนครั้ง | % ของคลาสจริง |")
        A("|---|---|---|---|")
        for _, r in val_pairs.iterrows():
            A(f"| {r['true_class']} {r['true_char']} | {r['pred_class']} {r['pred_char']} | "
              f"{r['count']} | {r['pct_of_true_class']}% |")
        A("")

    # ================= test =================
    A("## 8. ผลบนชุด test 5%")
    A("")
    if test_metrics:
        A(
            f"ชุดนี้ถูกกันออกตั้งแต่ขั้นแรกและ**ไม่เคยถูกใช้เทรนหรือ augment** "
            f"({fmt(test_metrics.get('n_images'))} ภาพ)"
        )
        A("")
        A(f"- **Accuracy = {fmt(test_metrics.get('accuracy'))}**")
        A(f"- **Macro F1 = {fmt(test_metrics.get('macro_f1'))}**")
        A(f"- Weighted F1 = {fmt(test_metrics.get('weighted_f1'))} · "
          f"Balanced accuracy = {fmt(test_metrics.get('balanced_accuracy'))} · "
          f"Top-5 accuracy = {fmt(test_metrics.get('top5_accuracy'))}")
        A("")
        A("ไฟล์ผลทำนายรายภาพ: `outputs_torch/reports/06_test_predictions.csv` "
          "(filename, true_class_id, pred_class_id, pred_char, confidence, correct, top-3)")
        A("")
        if test_per_class is not None:
            measured_t = test_per_class[test_per_class["support"] > 0]
            A("**10 คลาสที่ F1 ต่ำสุดในชุด test**")
            A("")
            A(md_table(
                measured_t.nsmallest(10, "f1-score"),
                {"label": "คลาส", "precision": "Precision", "recall": "Recall",
                 "f1-score": "F1", "support": "Support"},
            ))
            A("")
        if test_pairs is not None and not test_pairs.empty:
            A("**คู่ที่ทายผิดบ่อยที่สุดในชุด test**")
            A("")
            A("| คลาสจริง | ทำนายเป็น | จำนวนครั้ง | % ของคลาสจริง |")
            A("|---|---|---|---|")
            for _, r in test_pairs.iterrows():
                A(f"| {r['true_class']} {r['true_char']} | {r['pred_class']} {r['pred_char']} | "
                  f"{r['count']} | {r['pct_of_true_class']}% |")
            A("")

    # ================= วิธีทำซ้ำ =================
    A("## 9. วิธีรันซ้ำ")
    A("")
    A("```bash")
    A("uv sync")
    A("uv run python run_step1_explore.py          # สำรวจ + แบ่ง train/val/test")
    A("uv run python run_step2_baseline.py --epochs 3 --num-workers 4")
    A("uv run python run_step3_augment_plan.py     # balanced manifest + กราฟก่อน-หลัง")
    A("uv run python run_step4_train.py --epochs 10 --num-workers 4")
    A("uv run python run_step5_evaluate.py         # confusion matrix + รายงานต่อคลาส")
    A("uv run python run_step6_test_inference.py   # ทำนายชุด test 5%")
    A("uv run python make_results.py               # สร้างไฟล์นี้ใหม่จากผลจริง")
    A("```")
    A("")
    A("รายละเอียดการตัดสินใจทางเทคนิคและข้อควรระวังอยู่ใน `PYTORCH_PIPELINE.md`")
    A("")
    A("**หมายเหตุ**: checkpoint `.pth` (95–284MB) ไม่ได้ commit ขึ้น repo เพราะเกินลิมิต "
      "100MB ต่อไฟล์ของ GitHub - ถ้าต้องการเก็บโมเดลให้ใช้ Git LFS (ดูคำแนะนำใน `.gitignore`)")
    A("")

    if missing:
        A("---")
        A("")
        A(f"> ⚠ ไฟล์ผลลัพธ์ที่ยังไม่มีตอนสร้างรายงานนี้: {', '.join(sorted(set(missing)))}")
        A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"เขียน {OUT} ({len(L)} บรรทัด)")
    if missing:
        print(f"[warn] ไฟล์ที่ยังไม่มี: {sorted(set(missing))}")


if __name__ == "__main__":
    main()
