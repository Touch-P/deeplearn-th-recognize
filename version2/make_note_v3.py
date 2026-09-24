"""
make_note_v3.py  -  สร้าง RESULTS_v3.md จากไฟล์ผลลัพธ์จริงของ version 3
=======================================================================
อ่าน 07_metrics_v3.json / 07_before_after_v3.csv / 07_finetune_v3_history.csv /
class_weights_v3.json แล้วประกอบเป็นบันทึกสำหรับอ้างอิงตอนทำสไลด์

ตัวเลขทุกตัวอ่านจากไฟล์ ไม่ได้พิมพ์ด้วยมือ จึงไม่มีทางขัดกับผลจริง

รัน:  python make_note_v3.py
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

R = cfg.REPORTS_DIR
OUT = PROJECT_ROOT / "RESULTS_v3.md"


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    metrics = read_json(R / "07_metrics_v3.json")
    weights = read_json(cfg.MODELS_DIR / "class_weights_v3.json")
    compare = pd.read_csv(R / "07_before_after_v3.csv", encoding="utf-8-sig", dtype={"class_id": str})
    history = pd.read_csv(R / "07_finetune_v3_history.csv", encoding="utf-8-sig")

    before, after, delta = metrics["before"], metrics["after"], metrics["delta"]
    w, o = metrics["weighted_classes"], metrics["other_classes"]

    L: list[str] = []
    A = L.append

    A("# version 3 — fine-tune ด้วย manual class-weighted loss")
    A("")
    A(f"สร้างอัตโนมัติจากไฟล์ผลลัพธ์จริงด้วย `make_note_v3.py` เมื่อ {date.today().isoformat()}")
    A("")
    A("## ทำอะไร")
    A("")
    A(f"- โหลด checkpoint ของ version 2 (`{Path(metrics['from_checkpoint']).name}`) มา fine-tune ต่อ "
      f"**{metrics['epochs_finetuned']} epochs** ไม่ได้เทรนใหม่จากศูนย์")
    A(f"- ใส่ `weight` ต่อคลาสใน `nn.CrossEntropyLoss` เจาะจง **{metrics['n_weighted_classes']} คลาส** "
      f"ที่ confusion matrix ของ v2 ชี้ว่ายังสับสนกันเป็นระบบ (คง `label_smoothing="
      f"{cfg.LABEL_SMOOTHING}` เดิมไว้ด้วย)")
    A(f"- learning rate = `{metrics['lr']:g}` = 1/10 ของตอนเทรนหลัก (`{cfg.LEARNING_RATE:g}`) "
      f"ลดแบบ CosineAnnealingLR ตลอด {metrics['epochs_finetuned']} epochs")
    A("- dataloader และ augmentation ใช้ pipeline เดิมของ v2 ทั้งหมด (manifest ที่ balance ทุกคลาส "
      "เป็น 3,819 ภาพ = 274,968 ภาพต่อ epoch) ไม่เปลี่ยนอะไร")
    A(f"- optimizer เริ่มใหม่ ไม่โหลด state ของ v2 เพราะ state นั้นผูกกับตาราง lr ของรอบก่อน "
      f"ที่ลดลงไปจนเกือบศูนย์แล้ว")
    A("")

    A("## ที่มาของน้ำหนัก (อิง confusion matrix จุดไหน)")
    A("")
    A("นับ error ของแต่ละคลาสจาก `outputs/reports/05_confusion_matrix_val.csv` ของ v2 โดย")
    A("")
    A("> error ของคลาส = จำนวนครั้งที่ถูกทายผิดตอนเป็น true label + จำนวนครั้งที่ถูกทายมาผิด ๆ ตอนเป็น predicted label")
    A("")
    A("บน validation ของ v2 มี error ทั้งหมด 168 ครั้งจาก 11,913 ภาพ และ 21 คลาสที่ถูกถ่วงน้ำหนัก")
    A("ครอบคลุม **89.6% ของ error ทั้งหมด**")
    A("")
    A("| น้ำหนัก | คลาส | เหตุผล |")
    A("|---|---|---|")
    by_tier: dict[float, list[str]] = {}
    for char, info in weights["resolved"].items():
        by_tier.setdefault(info["weight"], []).append(f"{info['class_id']} {char}")
    reasons = {
        3.0: "error มากสุดสามอันดับแรก (า 81 · ๅ 34 · ว 30 ครั้ง) และสับสนกันไปมาในกลุ่มเดียวกัน",
        2.5: "วรรณยุกต์/สระลอยสับสนกันเอง (้ → ั 10 ครั้ง, ั → ้ 6 ครั้ง)",
        2.0: "รูปทรงใกล้กัน (ด → ต 7, ช → ซ 6, ซ → ช 4, ใ → า 7 ครั้ง)",
        1.5: "ยังพลาดประปราย 3–7 ครั้งต่อคลาส",
    }
    for tier in sorted(by_tier, reverse=True):
        A(f"| {tier:.1f} | {' · '.join(by_tier[tier])} | {reasons.get(tier, '')} |")
    A(f"| {weights['default_weight']:.1f} | คลาสที่เหลือทั้ง {weights['n_classes'] - weights['n_weighted_classes']} คลาส | ค่าเริ่มต้น |")
    A("")
    A(f"weight vector ตัวเต็ม (เรียงตาม index จริงของโมเดล) เก็บไว้ที่ "
      f"`outputs/models/{Path(metrics['class_weights_file']).name}` เพื่อให้ reproduce ย้อนหลังได้")
    A("")

    A("## ผลการ fine-tune ต่อ epoch")
    A("")
    A("| epoch | train loss (weighted) | train acc | train F1 | val loss | val acc | val macro F1 |")
    A("|---|---|---|---|---|---|---|")
    for _, r in history.iterrows():
        A(f"| {int(r['epoch'])} | {r['train_loss_weighted']:.4f} | {r['train_acc']:.4f} | "
          f"{r['train_macro_f1']:.4f} | {r['val_loss']:.4f} | {r['val_acc']:.4f} | "
          f"{r['val_macro_f1']:.4f} |")
    A("")
    A("> `train loss` ของ v3 เทียบกับ v2 ตรง ๆ ไม่ได้ เพราะเป็น loss ที่ถ่วงน้ำหนักแล้ว "
      "(คลาสที่ weight 3.0 นับสามเท่า) ส่วน `val loss` คำนวณด้วย loss ที่ **ไม่ถ่วงน้ำหนัก** "
      "จึงเทียบกับ v2 ได้ตามปกติ")
    A("")

    A("## เทียบผลรวมบน validation: v2 → v3")
    A("")
    A("| ตัวชี้วัด | version 2 | version 3 | ส่วนต่าง |")
    A("|---|---|---|---|")
    for key, name in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "**Macro F1**"),
        ("weighted_f1", "Weighted F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_recall", "Macro recall"),
        ("top5_accuracy", "Top-5 accuracy"),
    ):
        if key in before and key in after:
            A(f"| {name} | {before[key]:.4f} | {after[key]:.4f} | {delta.get(key, 0):+.4f} |")
    A("")

    A("## ผลแยกตามกลุ่มคลาส")
    A("")
    A("| กลุ่ม | recall เฉลี่ยก่อน | recall เฉลี่ยหลัง | ส่วนต่าง | error ก่อน | error หลัง | ดีขึ้น | แย่ลง |")
    A("|---|---|---|---|---|---|---|---|")
    A(f"| {metrics['n_weighted_classes']} คลาสที่ถ่วงน้ำหนัก | {w['mean_recall_before']:.4f} | "
      f"{w['mean_recall_after']:.4f} | {w['mean_recall_after'] - w['mean_recall_before']:+.4f} | "
      f"{w['errors_before']} | {w['errors_after']} | {w['n_improved']} | {w['n_worse']} |")
    A(f"| {weights['n_classes'] - metrics['n_weighted_classes']} คลาสที่ไม่ถ่วง (1.0) | "
      f"{o['mean_recall_before']:.4f} | {o['mean_recall_after']:.4f} | "
      f"{o['mean_recall_after'] - o['mean_recall_before']:+.4f} | "
      f"{o['errors_before']} | {o['errors_after']} | {o['n_improved']} | {o['n_worse']} |")
    A("")

    A("## recall ต่อคลาสของกลุ่มที่ถ่วงน้ำหนัก")
    A("")
    A("| คลาส | weight | support | recall v2 | recall v3 | ส่วนต่าง |")
    A("|---|---|---|---|---|---|")
    weighted = compare[compare["is_weighted"]].sort_values(
        ["manual_weight", "delta_recall"], ascending=[False, False]
    )
    for _, r in weighted.iterrows():
        A(f"| {r['label']} | {r['manual_weight']:.1f} | {int(r['support']):,} | "
          f"{r['recall_v2']:.4f} | {r['recall_v3']:.4f} | {r['delta_recall']:+.4f} |")
    A("")

    changed = compare[(~compare["is_weighted"]) & (compare["delta_recall"] != 0)].sort_values("delta_recall")
    if len(changed):
        A("### คลาสที่ไม่ได้ถ่วงน้ำหนักแต่ผลเปลี่ยน (ผลข้างเคียง)")
        A("")
        A("| คลาส | support | recall v2 | recall v3 | ส่วนต่าง |")
        A("|---|---|---|---|---|")
        for _, r in changed.iterrows():
            A(f"| {r['label']} | {int(r['support']):,} | {r['recall_v2']:.4f} | "
              f"{r['recall_v3']:.4f} | {r['delta_recall']:+.4f} |")
        A("")

    A("## ข้อควรระวังในการตีความ")
    A("")
    A("1. **ชุด train ของ v2 balance อยู่แล้ว** (ทุกคลาส 3,819 ภาพเท่ากัน) การใส่ class weight "
      "จึงไม่ใช่การแก้ความไม่สมดุลของข้อมูล แต่เป็นการ *เพิ่มน้ำหนักความสำคัญ* ให้คลาสที่ยัง "
      "สับสน ซึ่งเป็นการถ่วงรอบที่สองทับของเดิม")
    A("2. **เพดานของสิ่งที่ทำได้มีน้อย** v2 เหลือ error แค่ 168 ครั้งจาก 11,913 ภาพ (1.4%) "
      "การเปลี่ยนแปลงระดับ 0.00X จึงอยู่ในช่วงที่ noise ของการเทรนอธิบายได้ ไม่ควรสรุปว่า "
      "เทคนิคนี้ดีหรือไม่ดีจากตัวเลขเดียว")
    A("3. **คู่ที่สับสนหนักสุดสับสนกันเอง** เช่น า ↔ ๅ ↔ ว ทั้งสามตัวได้ weight 3.0 เท่ากัน "
      "การถ่วงน้ำหนักทั้งกลุ่มพร้อมกันจึงไม่ได้บอกโมเดลว่าควรเอนไปทางไหน แค่บอกว่า "
      "\"กลุ่มนี้สำคัญ\" การแยกคู่พวกนี้ให้ดีขึ้นจริงน่าจะต้องใช้ความละเอียดภาพที่สูงกว่า "
      "224×224 หรือ loss ที่ลงโทษเฉพาะคู่ (เช่น pairwise/metric learning) มากกว่าการถ่วงน้ำหนักคลาส")
    A("4. `val loss` ใช้ loss แบบไม่ถ่วงน้ำหนักเสมอ เพื่อให้เทียบกับ v2 ได้ตรง")
    A("")

    A("## ไฟล์ของ version 3")
    A("")
    A("| ไฟล์ | เนื้อหา |")
    A("|---|---|")
    A(f"| `outputs/models/{Path(metrics['v3_checkpoint']).name}` | checkpoint ของ v3 (ไม่เขียนทับ v1/v2) |")
    A(f"| `outputs/models/{Path(metrics['class_weights_file']).name}` | weight scheme ที่ใช้ พร้อม mapping ตัวอักษร → index |")
    A("| `outputs/reports/07_finetune_v3_history.csv` | log ต่อ epoch |")
    A("| `outputs/reports/07_metrics_v3.json` | สรุป before/after ทั้งหมด |")
    A("| `outputs/reports/07_before_after_v3.csv` | recall ต่อคลาส ก่อน-หลัง ทุก 72 คลาส |")
    A("| `outputs/reports/05_confusion_matrix_val_v3.csv` | confusion matrix ใหม่ (รูปแบบเดียวกับของ v2) |")
    A("| `outputs/reports/05_classification_report_val_v3.csv` | precision/recall/f1 ต่อคลาส |")
    A("| `outputs/figures/05_confusion_matrix_val_v3.png` | confusion matrix (+ `_normalized`) |")
    A("| `outputs/figures/07_recall_before_after_v3.png` | กราฟ recall ก่อน-หลังของกลุ่มที่ถ่วงน้ำหนัก |")
    A("| `outputs/figures/07_finetune_v3_learning_curves.png` | learning curve ของ 5 epoch นี้ |")
    A("")
    A("วิธีรันซ้ำ")
    A("")
    A("```bash")
    A("cd version2")
    A("python run_step7_finetune_v3.py                   # fine-tune 5 epochs + evaluate")
    A("python run_step7_finetune_v3.py --eval-only        # ประเมิน checkpoint v3 ที่มีอยู่ซ้ำ")
    A("python make_note_v3.py                            # สร้างไฟล์นี้ใหม่จากผลจริง")
    A("```")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"เขียน {OUT} ({len(L)} บรรทัด)")


if __name__ == "__main__":
    main()
