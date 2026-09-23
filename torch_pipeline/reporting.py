"""
torch_pipeline/reporting.py
===========================
ตัวช่วยสร้างตาราง metric ที่ใช้ซ้ำทั้งขั้นที่ 2 (baseline) และขั้นที่ 5 (final)

ทุกไฟล์ CSV เซฟด้วย encoding utf-8-sig เพื่อให้เปิดใน Excel บน Windows แล้ว
ตัวอักษรไทยไม่เพี้ยน
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from torch_pipeline import config as cfg
from torch_pipeline.data import display_label


def class_tick_labels(class_to_idx: dict[str, int], label_map: dict[str, str]) -> list[str]:
    """ป้ายกำกับเรียงตาม index ของโมเดล ('161 ก', '209 ◌ั', ...)"""
    ordered = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    return [display_label(c, label_map) for c, _ in ordered]


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """คืน (cm ตัวเลขดิบ, cm normalized ต่อแถวเป็น %) - แถวที่ support = 0
    จะเป็นศูนย์ทั้งแถวแทนที่จะเป็น NaN"""
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm * 100.0, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0
    )
    return cm, cm_norm


def per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_to_idx: dict[str, int],
    label_map: dict[str, str],
) -> pd.DataFrame:
    """ตาราง precision / recall / f1 / support / accuracy ต่อคลาส

    per-class accuracy ของงาน multi-class = diagonal / row total = recall ของ
    คลาสนั้นพอดี จึงใส่เป็นคอลัมน์ accuracy ให้ตรงตามที่สเปกขอ โดยคำนวณจาก
    confusion matrix ตรง ๆ
    """
    ordered = sorted(class_to_idx.items(), key=lambda kv: kv[1])
    tick_labels = [display_label(c, label_map) for c, _ in ordered]
    num_classes = len(ordered)
    labels = np.arange(num_classes)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=tick_labels,
        output_dict=True,
        zero_division=0,
    )
    cm, _ = build_confusion(y_true, y_pred, num_classes)
    supports = cm.sum(axis=1)
    accuracy = np.divide(
        np.diag(cm).astype(float), supports, out=np.full(num_classes, np.nan), where=supports > 0
    )

    rows = []
    for (class_id, idx), tick in zip(ordered, tick_labels):
        rows.append(
            {
                "class_id": class_id,
                "thai_char": label_map.get(class_id, ""),
                "label": tick,
                "index": idx,
                "precision": report[tick]["precision"],
                "recall": report[tick]["recall"],
                "f1-score": report[tick]["f1-score"],
                "accuracy": accuracy[idx],
                "support": int(supports[idx]),
                "correct": int(cm[idx, idx]),
            }
        )
    return pd.DataFrame(rows)


def overall_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None, num_classes: int
) -> dict:
    """ตัวเลขสรุปภาพรวม

    balanced accuracy = ค่าเฉลี่ยของ recall ทุกคลาส บน dataset ที่คลาสใหญ่สุด
    มีภาพมากกว่าคลาสเล็กสุดหลายพันเท่า accuracy ปกติจะถูกคลาสใหญ่ครอบงำ
    ตัวเลขนี้จึงเป็นตัวที่บอกความสามารถจริงกับคลาสเล็ก
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    report = classification_report(
        y_true, y_pred, labels=np.arange(num_classes), output_dict=True, zero_division=0
    )
    out["macro_f1"] = float(report["macro avg"]["f1-score"])
    out["weighted_f1"] = float(report["weighted avg"]["f1-score"])
    out["macro_recall"] = float(report["macro avg"]["recall"])
    if probs is not None and probs.shape[1] == num_classes:
        out["top5_accuracy"] = float(
            top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(num_classes))
        )
    return out


def select_weak_classes(
    per_class: pd.DataFrame, rule: str = cfg.WEAK_CLASS_RULE, metric: str = cfg.WEAK_CLASS_METRIC
) -> tuple[list[str], dict]:
    """เลือก "คลาสที่ทายผิดบ่อย" จากตาราง per-class ของ baseline

    เกณฑ์ (ตามที่สเปกเสนอไว้):
      mean-sd  : metric < mean - 1SD
      bottom20 : 20% ล่างสุดเมื่อเรียงตาม metric
      both     : union ของสองเกณฑ์

    metric ค่าเริ่มต้นคือ per-class F1 ไม่ใช่ recall: recall อย่างเดียวมองไม่เห็น
    คลาสที่โมเดล "ทายเกิน" (เดาคลาสนี้พร่ำเพรื่อจน precision ต่ำ) ซึ่งก็คือ
    ความสับสนแบบหนึ่งที่ต้องแก้เหมือนกัน F1 รวมทั้งสองด้านไว้ในตัวเดียว

    คิดเฉพาะคลาสที่มี support > 0 บน validation เพราะคลาสที่ไม่มีภาพให้ทดสอบ
    ไม่ได้ "ทายผิด" - มันยังไม่ถูกทดสอบเลย (ดังนั้นจะถูกใส่กลับเข้าไปแบบ
    บังคับด้านล่าง เพราะคลาสที่ข้อมูลน้อยจนไม่มี val ก็คือคลาสที่ต้องช่วย)
    """
    if metric not in per_class.columns:
        raise ValueError(f"ไม่มีคอลัมน์ '{metric}' ในตาราง per-class")
    measured = per_class[per_class["support"] > 0]
    values = measured[metric].to_numpy()
    mean, sd = float(values.mean()), float(values.std())
    threshold_meansd = mean - sd
    threshold_bottom20 = float(np.quantile(values, 0.20))

    by_meansd = set(measured.loc[measured[metric] < threshold_meansd, "class_id"])
    by_bottom20 = set(
        measured.sort_values(metric).head(max(1, int(round(0.20 * len(measured)))))["class_id"]
    )

    if rule == "mean-sd":
        weak = by_meansd
    elif rule == "bottom20":
        weak = by_bottom20
    elif rule == "both":
        weak = by_meansd | by_bottom20
    else:
        raise ValueError(f"rule ไม่รู้จัก: {rule}")

    # คลาสที่ไม่มีภาพใน val (ข้อมูลน้อยมาก) ถูกนับเป็นคลาสอ่อนแอโดยปริยาย
    unmeasured = set(per_class.loc[per_class["support"] == 0, "class_id"])
    weak |= unmeasured

    info = {
        "rule": rule,
        "metric": metric,
        "mean_metric": mean,
        "sd_metric": sd,
        "threshold_mean_minus_1sd": threshold_meansd,
        "threshold_bottom20": threshold_bottom20,
        "n_by_mean_sd": len(by_meansd),
        "n_by_bottom20": len(by_bottom20),
        "n_unmeasured_forced_in": len(unmeasured),
        "n_weak_total": len(weak),
    }
    return sorted(weak, key=lambda c: int(c)), info


def top_confused_pairs(
    cm: np.ndarray, class_to_idx: dict[str, int], label_map: dict[str, str], top_n: int = 10
) -> pd.DataFrame:
    """คู่ (จริง -> ทำนาย) ที่สับสนกันมากที่สุด นอกเส้นทแยงมุม"""
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    off = cm.copy()
    np.fill_diagonal(off, 0)
    supports = cm.sum(axis=1)

    rows = []
    for flat in np.argsort(off.ravel())[::-1][:top_n]:
        i, j = divmod(int(flat), cm.shape[0])
        count = int(off[i, j])
        if count == 0:
            continue
        true_c, pred_c = idx_to_class[i], idx_to_class[j]
        rows.append(
            {
                "true_class": true_c,
                "true_char": label_map.get(true_c, ""),
                "pred_class": pred_c,
                "pred_char": label_map.get(pred_c, ""),
                "count": count,
                "true_support": int(supports[i]),
                "pct_of_true_class": round(count * 100.0 / max(1, int(supports[i])), 2),
            }
        )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"บันทึกตาราง -> {path}")
    return path


def save_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"บันทึกสรุป -> {path}")
    return path


def print_overall(metrics: dict, title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("-" * 66)
    print(f"Accuracy               = {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy      = {metrics['balanced_accuracy']:.4f}  (เฉลี่ย recall ต่อคลาส)")
    print(f"Macro F1               = {metrics['macro_f1']:.4f}")
    print(f"Weighted F1            = {metrics['weighted_f1']:.4f}")
    if "top5_accuracy" in metrics:
        print(f"Top-5 accuracy         = {metrics['top5_accuracy']:.4f}")
    print("=" * 66)
