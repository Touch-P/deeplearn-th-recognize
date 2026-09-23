"""
data_utils.py
=============
Utilities for discovering, indexing, and splitting the Thai character/digit
image dataset stored under ``ThaiCharacter Dataset/round2/<class_id>/<image>.jpg``.

Class labels
------------
The dataset provides 72 class folders named with numeric IDs (e.g. ``161``,
``185``, ``193``, ...). ``label.json`` at the project root maps each numeric
ID to the actual Thai grapheme (consonant / vowel / tone mark / digit) it
represents, e.g. ``{"id": 161, "char": "ก"}``. That mapping is loaded into
``CLASS_ID_TO_THAI`` below. Everything else in the pipeline is still keyed
by the opaque ``class_id`` string (the folder name) — the Thai character is
only used for human-readable display in plots/tables/reports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # = version1/


def resolve_data_root() -> Path:
    """หาโฟลเดอร์ภาพต้นฉบับ ตามลำดับความสำคัญนี้

    1. ตัวแปรสภาพแวดล้อม ``THAI_DATASET_ROOT`` (ถ้าตั้งไว้ ใช้อันนี้เสมอ)
    2. ``version1/ThaiCharacter Dataset/round2`` - ถ้าคัดลอกข้อมูลมาไว้ในเวอร์ชันนี้
    3. ``<repo root>/ThaiCharacter Dataset/round2`` - ที่เก็บจริงในปัจจุบัน
       ซึ่งแชร์กับ version2 เพราะเป็นข้อมูลตั้งต้นชุดเดียวกัน (62,707 ภาพ
       ~200MB) ไม่ควรเก็บสองชุด - ตัวเวอร์ชันยัง independent กันเพราะไม่มี
       การ import โค้ดข้ามโฟลเดอร์ มีแต่การชี้ไปที่ "ข้อมูล" ชุดเดียวกัน

    คืน path ตัวเลือกที่ 3 เป็นค่าตั้งต้นถ้าไม่พบที่ไหนเลย เพื่อให้ข้อความ
    error ของ list_images() บอก path ที่คาดหวังได้ถูก
    """
    env = os.environ.get("THAI_DATASET_ROOT")
    if env:
        return Path(env)
    local = PROJECT_ROOT / "ThaiCharacter Dataset" / "round2"
    if local.exists():
        return local
    return PROJECT_ROOT.parent / "ThaiCharacter Dataset" / "round2"


DATA_ROOT = resolve_data_root()
LABEL_JSON = PROJECT_ROOT / "label.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
CACHE_DIR = PROJECT_ROOT / "cache"  # precomputed arrays; gitignored (large)
SPLIT_CSV = OUTPUTS_DIR / "split.csv"

for d in (FIGURES_DIR, MODELS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def load_class_id_to_thai(path: Path = LABEL_JSON) -> dict[str, str]:
    """Load the {class_id: thai_char} mapping from label.json.

    id 240 is mapped to Thai digit zero "๐" (U+0E50) rather than the ASCII
    "0" present in the raw label.json, for consistency with the other
    9 Thai-numeral classes (241-249 = ๑-๙); this is a display-only fix and
    does not affect training, which is keyed by the numeric folder id.
    """
    if not path.exists():
        return {}
    import json

    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    mapping = {str(e["id"]): e["char"] for e in entries}
    if mapping.get("240") == "0":
        mapping["240"] = "๐"
    return mapping


# {"161": "ก", "162": "ข", ..., "249": "๙"} — see load_class_id_to_thai().
CLASS_ID_TO_THAI: dict[str, str] = load_class_id_to_thai()


def list_images(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Walk ``data_root`` and build a DataFrame with one row per image.

    Columns: filepath (str, absolute), class_id (str, folder name),
    filename (str).
    """
    rows = []
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at {data_root}. Place the provided "
            f"'round2' dataset folder at the project root."
        )
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        for f in class_dir.iterdir():
            if f.suffix.lower() in VALID_EXTS:
                rows.append(
                    {
                        "filepath": str(f.resolve()),
                        "class_id": class_id,
                        "filename": f.name,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {data_root}")
    return df


def class_label_display(class_id: str) -> str:
    """Human-readable label for plots/reports: Thai char if known, else the id."""
    thai = CLASS_ID_TO_THAI.get(str(class_id))
    return f"{class_id} ({thai})" if thai else str(class_id)


def stratified_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
    min_val_per_class: int = 1,
) -> pd.DataFrame:
    """Custom stratified 80/20 split that is robust to very small classes.

    ``sklearn.model_selection.train_test_split(..., stratify=...)`` raises an
    error whenever any class has fewer than 2 members, because it cannot put
    at least one sample in *each* split. This dataset contains two such
    classes (only 1 image each), so we implement our own rule instead:

    - class has >= 5 images  -> round(val_fraction * n) go to validation
      (at least 1), the rest to train.
    - class has 2-4 images   -> exactly 1 image goes to validation, the rest
      to train (keeps every class represented in both splits when at all
      possible).
    - class has 1 image      -> the single image goes to train only; the
      class has zero validation coverage (unavoidable, documented in the
      EDA notebook).

    Returns the input DataFrame with an added ``split`` column
    (``"train"`` or ``"val"``).
    """
    rng = np.random.RandomState(random_state)
    split_col = pd.Series(index=df.index, dtype=object)

    for class_id, group in df.groupby("class_id"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            n_val = 0
        elif n < 5:
            n_val = 1
        else:
            n_val = max(min_val_per_class, round(n * val_fraction))
            n_val = min(n_val, n - 1)  # always leave >=1 for train
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        split_col.loc[val_idx] = "val"
        split_col.loc[train_idx] = "train"

    out = df.copy()
    out["split"] = split_col
    return out


def save_split(df: pd.DataFrame, path: Path = SPLIT_CSV) -> None:
    df.to_csv(path, index=False)


def load_split(path: Path = SPLIT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/01_preprocessing_augmentation.ipynb first."
        )
    return pd.read_csv(path, dtype={"class_id": str})


def class_counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return class_id, count, sorted descending — used for the imbalance bar chart."""
    counts = (
        df.groupby("class_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return counts
