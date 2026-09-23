"""
torch_pipeline/data.py
======================
สำรวจ dataset, แบ่ง split 3 ทาง (train/val/test), สร้าง balanced augmentation
manifest และ Dataset ของ PyTorch

หลักการสำคัญของไฟล์นี้
----------------------
1. ชุด test 5% ถูกตัดออก "ก่อน" ทุกอย่าง และไม่มี code path ไหนนำไป train
   หรือ augment - manifest ของ train ถูกสร้างจากแถวที่ split == "train"
   เท่านั้น
2. การ augment ไม่เขียนไฟล์ภาพลงดิสก์ แต่สร้าง *manifest* ที่บอกว่าภาพต้นฉบับ
   ไหนจะถูกแปลงด้วยความเข้มระดับไหน ด้วย seed อะไร แล้วแปลงตอน __getitem__
   เหตุผล: balance ทุกคลาสถึง 4,020 ภาพ = 289,440 ภาพ ถ้าเซฟเป็น .jpg ที่
   224x224 จะกินดิสก์ ~20GB และเสียเวลาเขียน/อ่านฟรี ๆ ผลลัพธ์ที่โมเดลเห็น
   เหมือนกันเป๊ะ เพราะ seed ต่อภาพถูกล็อกไว้ (FIXED_AUGMENTATION=True)
   ส่วนกราฟ "ก่อน-หลัง augment" นับจาก manifest ซึ่งเป็นจำนวนจริงที่โมเดลเห็น
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torch_pipeline import config as cfg


# ---------------------------------------------------------------------------
# 1. Label mapping + การสำรวจโฟลเดอร์
# ---------------------------------------------------------------------------
def load_label_map(path: Path = cfg.LABEL_JSON) -> dict[str, str]:
    """อ่าน label.json -> {"161": "ก", "162": "ข", ..., "249": "๙"}

    id 240 ใน label.json เขียนเป็น "0" (เลขอารบิก) เปลี่ยนเป็น "๐" (U+0E50)
    ให้เข้าชุดกับคลาส 241-249 = ๑-๙ เป็นการแก้เพื่อการแสดงผลเท่านั้น
    ไม่กระทบ label ที่ใช้เทรน (ซึ่งใช้ชื่อโฟลเดอร์ตัวเลข)
    """
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    mapping = {str(e["id"]): e["char"] for e in entries}
    if mapping.get("240") == "0":
        mapping["240"] = "๐"
    return mapping


def scan_dataset(data_root: Path = cfg.DATA_ROOT) -> pd.DataFrame:
    """เดินทุกโฟลเดอร์คลาส คืน DataFrame: filepath, class_id, filename"""
    if not data_root.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset: {data_root}")

    rows = []
    for class_dir in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in cfg.VALID_EXTS:
                rows.append(
                    {
                        "filepath": str(f.resolve()),
                        "class_id": class_dir.name,
                        "filename": f.name,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"ไม่พบไฟล์ภาพใด ๆ ใน {data_root}")
    return df


def count_per_class(df: pd.DataFrame) -> dict[str, int]:
    """{class_name: count} เรียงจากมากไปน้อย (ใช้พล็อต bar chart ขั้นที่ 1)"""
    counts = df.groupby("class_id").size().sort_values(ascending=False)
    return {str(k): int(v) for k, v in counts.items()}


def build_class_index(df: pd.DataFrame) -> dict[str, int]:
    """class_to_idx โดยเรียงตามรหัสคลาสเป็นตัวเลข - ลำดับคงที่และ
    reproduce ได้ ไม่ขึ้นกับลำดับที่ระบบไฟล์คืนมา"""
    classes = sorted(df["class_id"].unique(), key=lambda c: int(c))
    return {str(c): i for i, c in enumerate(classes)}


def save_class_index(class_to_idx: dict[str, int], label_map: dict[str, str]) -> Path:
    """เซฟ class_to_idx.json พร้อมตัวอักษรไทยกำกับ เพื่อใช้ตอน inference"""
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(i): c for c, i in class_to_idx.items()},
        "idx_to_thai": {str(i): label_map.get(c, "") for c, i in class_to_idx.items()},
    }
    with open(cfg.CLASS_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return cfg.CLASS_INDEX_JSON


def load_class_index(path: Path = cfg.CLASS_INDEX_JSON) -> tuple[dict[str, int], dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["class_to_idx"], payload.get("idx_to_thai", {})


def display_label(class_id: str, label_map: dict[str, str]) -> str:
    """'161 ก' / '209 ◌ั' - ป้ายกำกับสำหรับกราฟและรายงาน

    สระและวรรณยุกต์ (ั ิ ่ ) เป็น combining character ถ้าพิมพ์เดี่ยว ๆ จะไป
    เกาะกับอักขระตัวหน้าแล้วอ่านไม่ออก จึงใส่ ◌ (U+25CC) นำหน้าแบบเดียวกับ
    ที่พจนานุกรมไทยใช้
    """
    import unicodedata

    thai = label_map.get(str(class_id), "")
    if not thai:
        return str(class_id)
    if unicodedata.combining(thai[0]):
        thai = "◌" + thai
    return f"{class_id} {thai}"


# ---------------------------------------------------------------------------
# 2. Stratified split 3 ทาง
# ---------------------------------------------------------------------------
def make_splits(
    df: pd.DataFrame,
    test_frac: float = cfg.TEST_FRACTION,
    val_frac: float = cfg.VAL_FRACTION,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """แบ่ง test 5% ออกก่อน แล้วแบ่งส่วนที่เหลือเป็น train/val = 80/20
    ทั้งหมดทำแบบ stratified (แบ่งภายในแต่ละคลาส สัดส่วนคลาสจึงคงเดิมทุก split)

    ทำไมไม่ใช้ sklearn train_test_split(stratify=...):
    มันจะ error ทันทีถ้าคลาสไหนมีสมาชิกน้อยกว่าจำนวน split ที่ขอ - dataset นี้
    มีคลาสที่มีภาพเดียว (163 ฃ, 177 ฑ) และสองภาพ (204 ฬ) จริง ๆ จึงต้องมี
    กฎรองรับคลาสเล็กเอง:

      n >= 3  -> test = max(1, round(n*test_frac)), ที่เหลือแบ่ง train/val
      n == 2  -> train 1, val 1, test 0
      n == 1  -> train 1 (คลาสนั้นไม่มีข้อมูล val/test เลี่ยงไม่ได้)

    คลาสเล็กพวกนี้จะไม่ปรากฏในชุด test ซึ่งต้องระบุไว้ในรายงาน
    """
    rng = np.random.RandomState(seed)
    split_col = pd.Series(index=df.index, dtype=object)

    for _, group in df.groupby("class_id"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            split_col.loc[idx] = "train"
            continue
        if n == 2:
            split_col.loc[idx[:1]] = "val"
            split_col.loc[idx[1:]] = "train"
            continue

        n_test = max(1, int(round(n * test_frac)))
        n_test = min(n_test, n - 2)  # เหลือให้ train และ val อย่างน้อย 1 ภาพ
        test_idx, rest = idx[:n_test], idx[n_test:]

        n_val = max(1, int(round(len(rest) * val_frac)))
        n_val = min(n_val, len(rest) - 1)  # เหลือ train อย่างน้อย 1 ภาพ
        val_idx, train_idx = rest[:n_val], rest[n_val:]

        split_col.loc[test_idx] = "test"
        split_col.loc[val_idx] = "val"
        split_col.loc[train_idx] = "train"

    out = df.copy()
    out["split"] = split_col
    return out


def save_splits(df: pd.DataFrame, path: Path = cfg.SPLIT_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_splits(path: Path = cfg.SPLIT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {path} - รัน run_step1_explore.py ก่อนเพื่อสร้าง split"
        )
    return pd.read_csv(path, dtype={"class_id": str})


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ตารางสรุปจำนวนภาพต่อคลาสในแต่ละ split (ใช้ตรวจว่า stratify ถูก)"""
    pivot = (
        df.pivot_table(index="class_id", columns="split", values="filename", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    for col in ("train", "val", "test"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["train", "val", "test"]]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# 3. Balanced + error-driven augmentation manifest
# ---------------------------------------------------------------------------
def resolve_target_count(train_df: pd.DataFrame, target_count: int | None = None) -> int:
    """target = จำนวนภาพของคลาสที่เยอะสุดตามสเปก ถ้าไม่ได้ระบุมา"""
    counts = train_df.groupby("class_id").size()
    return int(target_count if target_count else counts.max())


def build_balanced_manifest(
    train_df: pd.DataFrame,
    weak_classes: set[str] | None = None,
    target_count: int | None = None,
    light_extra_frac: float = cfg.LIGHT_EXTRA_FRAC,
    seed: int = cfg.SEED,
    cap_majority: bool = False,
) -> pd.DataFrame:
    """สร้าง manifest ของ training set ที่ balance แล้ว

    ผลลัพธ์: DataFrame คอลัมน์ filepath, class_id, aug_kind, aug_seed
      aug_kind = "original" -> ใช้ภาพจริง ไม่แปลงอะไร
                 "normal"   -> augment ความเข้มปกติ
                 "intense"  -> augment เข้มข้น/หลากหลายกว่า (คลาสที่ทายผิดบ่อย)
                 "light"    -> augment เบา ๆ (ใช้กับคลาสที่เกิน target แล้ว)

    target_count = None -> ใช้จำนวนของคลาสที่เยอะสุดตามสเปก ซึ่งทำให้ทุกคลาส
    มีจำนวนแถวเท่ากับ target พอดี = balanced จริง

    cap_majority: มีผลเฉพาะเมื่อตั้ง target ต่ำกว่าคลาสที่เยอะสุดเอง
      False (ค่าเริ่มต้น) -> คลาสที่มีภาพเกิน target เก็บภาพจริงไว้ทั้งหมด
                            ไม่ทิ้งข้อมูลจริงทิ้ง แต่ผลคือยังไม่ balanced
      True                -> สุ่มลดคลาสที่เกิน target ลงมาเท่ากับ target
                            (undersampling) ได้ balanced จริงแต่เสียข้อมูลจริง
    """
    weak_classes = {str(c) for c in (weak_classes or set())}
    target = resolve_target_count(train_df, target_count)

    rng = random.Random(seed)
    records: list[dict] = []

    for class_id, group in train_df.groupby("class_id"):
        class_id = str(class_id)
        originals = group[["filepath", "class_id"]].to_dict("records")
        if cap_majority and len(originals) > target:
            originals = random.Random(f"{seed}-{class_id}").sample(originals, target)
        n_orig = len(originals)

        # 3.1 ภาพต้นฉบับใส่เข้าไปครบทุกภาพก่อน - ข้อมูลจริงมีค่ามากที่สุด
        for i, rec in enumerate(originals):
            records.append({**rec, "aug_kind": "original", "aug_seed": 0})

        # 3.2 ถ้ายังไม่ถึง target -> เติมด้วยภาพ augment
        need = target - n_orig
        if need > 0:
            kind = "intense" if class_id in weak_classes else "normal"
            order = list(range(n_orig))
            rng.shuffle(order)
            for k in range(need):
                # วนใช้ภาพต้นฉบับซ้ำแบบ round-robin ทุกภาพจึงถูกใช้เป็นต้นแบบ
                # ใกล้เคียงกัน ไม่ใช่สุ่มจนภาพบางใบถูกใช้ 10 ครั้งอีกใบ 0 ครั้ง
                src = originals[order[k % n_orig]]
                records.append(
                    {
                        **src,
                        "aug_kind": kind,
                        # seed ต่อภาพ: ล็อกให้ภาพ augment ใบที่ k ของคลาสนี้
                        # ออกมาเหมือนกันทุกครั้งที่รัน
                        "aug_seed": (hash((class_id, k)) & 0x7FFFFFFF) or 1,
                    }
                )

        # 3.3 คลาสที่เกิน target แล้ว: เพิ่มภาพ augment เบา ๆ (ไม่บังคับตามสเปก)
        elif light_extra_frac > 0:
            n_extra = int(round(n_orig * light_extra_frac))
            for k in range(n_extra):
                src = originals[k % n_orig]
                records.append(
                    {
                        **src,
                        "aug_kind": "light",
                        "aug_seed": (hash((class_id, "light", k)) & 0x7FFFFFFF) or 1,
                    }
                )

    manifest = pd.DataFrame.from_records(records)
    manifest["class_id"] = manifest["class_id"].astype(str)
    return manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """จำนวนภาพต่อคลาส แยกตามชนิดการ augment (ใช้ทำกราฟก่อน-หลัง)"""
    pivot = (
        manifest.pivot_table(
            index="class_id", columns="aug_kind", values="filepath", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
    )
    for col in ("original", "normal", "intense", "light"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["original", "normal", "intense", "light"]]
    pivot["total"] = pivot.sum(axis=1)
    return pivot


# ---------------------------------------------------------------------------
# 4. Dataset / DataLoader
# ---------------------------------------------------------------------------
def load_rgb(path: str) -> Image.Image:
    """อ่านภาพเป็น RGB (ภาพต้นฉบับเป็น grayscale แต่ ResNet รับ 3 channel)"""
    with Image.open(path) as im:
        return im.convert("RGB")


class ThaiCharDataset(Dataset):
    """Dataset ธรรมดาสำหรับ val/test และ baseline train (ไม่ balance)

    คืน (image_tensor, label, row_index) - row_index ใช้ย้อนกลับไปหา filename
    ตอนเขียนไฟล์ผลทำนาย
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform,
        has_labels: bool = True,
    ):
        self.rows = rows.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        img = self.transform(load_rgb(row["filepath"]))
        label = self.class_to_idx[str(row["class_id"])] if self.has_labels else -1
        return img, label, i


class AugmentedManifestDataset(Dataset):
    """Dataset ที่ขับด้วย manifest จาก build_balanced_manifest()

    transform ถูกเลือกตาม aug_kind ของแต่ละแถว และถูกเรียกภายใน
    torch.random.fork_rng() ที่ล็อก seed ไว้ เพื่อให้:
      - FIXED_AUGMENTATION=True  -> ภาพ augment ใบเดิมเหมือนกันทุก epoch
      - FIXED_AUGMENTATION=False -> สุ่มใหม่ทุก epoch (เรียก set_epoch())
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        class_to_idx: dict[str, int],
        transforms_by_kind: dict[str, object],
        fixed: bool = cfg.FIXED_AUGMENTATION,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transforms_by_kind = transforms_by_kind
        self.fixed = fixed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """เรียกก่อนเริ่มแต่ละ epoch - มีผลเฉพาะเมื่อ fixed=False"""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int):
        row = self.manifest.iloc[i]
        kind = row["aug_kind"]
        transform = self.transforms_by_kind[kind]
        img = load_rgb(row["filepath"])

        if kind == "original":
            out = transform(img)
        else:
            seed = int(row["aug_seed"])
            if not self.fixed:
                seed = (seed + self.epoch * 7_919) & 0x7FFFFFFF
            # fork_rng: สุ่มด้วย seed นี้โดยไม่ไปรบกวน global RNG ของ worker
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                random.seed(seed)
                out = transform(img)

        return out, self.class_to_idx[str(row["class_id"])], i


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = cfg.NUM_WORKERS,
    device=None,
) -> DataLoader:
    """DataLoader ที่ตั้งค่าเผื่อ Windows ไว้แล้ว (persistent workers +
    pin_memory เฉพาะเมื่อใช้ CUDA)"""
    pin = bool(device is not None and getattr(device, "type", "cpu") == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )
