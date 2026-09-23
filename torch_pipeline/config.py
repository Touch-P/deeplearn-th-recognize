"""
torch_pipeline/config.py
========================
ทุก path และ hyperparameter อยู่ในไฟล์นี้ไฟล์เดียว - แก้ที่นี่ที่เดียวพอ

โครงสร้างข้อมูลจริงของโปรเจกต์นี้ (ตรวจสอบแล้ว):

    ThaiCharacter Dataset/round2/<class_id>/<image>.jpg

โฟลเดอร์คลาสตั้งชื่อด้วย "รหัสตัวเลข" (161, 162, ... 249) ไม่ใช่ตัวอักษรไทย
และไฟล์เป็น .jpg ไม่ใช่ .png - รหัสพวกนี้ถูก map เป็นตัวอักษรไทยด้วย
label.json ที่ root ของโปรเจกต์ (เช่น {"id": 161, "char": "ก"})
โครงสร้างนี้เข้ากันได้กับ torchvision ImageFolder โดยตรง แต่ pipeline นี้
ใช้ manifest (DataFrame/CSV) แทน เพราะต้องคุม split 3 ทางและต้องสร้าง
balanced augmentation manifest ซึ่ง ImageFolder ทำไม่ได้
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "ThaiCharacter Dataset" / "round2"
LABEL_JSON = PROJECT_ROOT / "label.json"

# แยก output ของ PyTorch ออกจาก outputs/ ของ track TensorFlow เดิม
# เพื่อไม่ให้ไฟล์ทับกัน
OUTPUT_ROOT = PROJECT_ROOT / "outputs_torch"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
MODELS_DIR = OUTPUT_ROOT / "models"
SPLITS_DIR = OUTPUT_ROOT / "splits"
LOGS_DIR = OUTPUT_ROOT / "logs"
SAMPLES_DIR = OUTPUT_ROOT / "samples"

ALL_DIRS = (FIGURES_DIR, REPORTS_DIR, MODELS_DIR, SPLITS_DIR, LOGS_DIR, SAMPLES_DIR)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
SEED = 42
TEST_FRACTION = 0.05  # ชุดที่กันไว้ให้อาจารย์ตรวจ - ห้าม train/augment
VAL_FRACTION = 0.20  # 20% ของส่วนที่เหลือ (95%) -> train/val = 80/20

SPLIT_CSV = SPLITS_DIR / "splits.csv"  # ไฟล์เดียว มีคอลัมน์ split = train/val/test
CLASS_INDEX_JSON = MODELS_DIR / "class_to_idx.json"

# ---------------------------------------------------------------------------
# Image / model
# ---------------------------------------------------------------------------
IMG_SIZE = 224  # ResNet50 pretrained ImageNet ทำงานดีสุดที่ 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ภาพต้นฉบับมีขนาดประมาณ 12x18 ถึง 30x45 px และไม่เป็นสี่เหลี่ยมจัตุรัส
# False = resize เป็น 224x224 ตรง ๆ ตามสเปก (สัดส่วนภาพเพี้ยนเล็กน้อย
#         แต่เพี้ยนเหมือนกันทั้ง train/val/test จึงไม่ทำให้โมเดลเสียเปรียบ)
# True  = เติมขอบขาวให้เป็นจัตุรัสก่อน แล้วค่อย resize (รักษารูปทรงตัวอักษร)
PAD_TO_SQUARE = False

BACKBONE = "resnet50"  # torchvision.models.resnet50, weights=IMAGENET1K_V2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BASELINE_EPOCHS = 5  # ขั้นที่ 2: เทรนเร็ว ๆ เพื่อดู error pattern เท่านั้น
FINAL_EPOCHS = 10  # ขั้นที่ 4
BATCH_SIZE = 48  # วัดจริงบน RTX 3050 4GB: 200 img/s, VRAM peak 2.3GB (ปลอดภัย)
EVAL_BATCH_SIZE = 96  # ตอน eval ไม่เก็บ gradient จึงใส่ได้มากกว่า
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 6  # ต้อง feed GPU ให้ทัน ~200 img/s ขณะที่ 80% ของภาพต้องผ่าน
#                  augmentation บน CPU (Windows: ต้องเรียกใน if __name__ == "__main__")
# ขั้น eval/inference ไม่มี augmentation จึงไม่ต้องใช้ worker มาก และบน Windows
# แต่ละ worker เป็น process ใหม่ที่ import torch ใหม่ (กิน commit memory ~700MB)
# ถ้าตั้งเยอะเกินไปแล้วรันหลายสคริปต์พร้อมกันจะเจอ
# "OSError [WinError 1455] The paging file is too small"
EVAL_NUM_WORKERS = 2
USE_AMP = True  # mixed precision: เร็วขึ้น ~2x และประหยัด VRAM

# channels_last (NHWC) ปกติเป็นการปรับแต่งที่ทำให้ conv บน tensor core เร็วขึ้น
# แต่ "วัดจริงบนเครื่องนี้แล้วช้าลง 8 เท่าตอน backward":
#   ResNet50 @224 bs16 AMP  channels_last =  24 img/s
#                           contiguous    = 189 img/s
# (forward อย่างเดียว channels_last เร็วกว่าจริง 675 vs 556 img/s)
# จึงปิดตอนเทรน และเปิดเฉพาะตอน inference ล้วน ๆ ถ้าย้ายไปรันบน GPU ตัวอื่น
# ควรวัดซ้ำด้วย: อาจกลับผลได้
CHANNELS_LAST_TRAIN = False
CHANNELS_LAST_EVAL = True

# ---------------------------------------------------------------------------
# Error-driven balanced augmentation (ขั้นที่ 3)
# ---------------------------------------------------------------------------
# None = ใช้จำนวนภาพของคลาสที่เยอะสุดเป็น target ตามสเปก
# (คลาส "า" = 3,819 ภาพในชุด train หลังกัน test 5% และ val 20% ออกแล้ว)
# ใส่ตัวเลขเพื่อลดขนาด epoch ลง (เช่น 1500) ถ้าเวลาเทรนไม่พอ
TARGET_COUNT: int | None = None

# เกณฑ์เลือก "คลาสที่ทายผิดบ่อย" จาก baseline:
#   "mean-sd"   -> recall < mean - 1SD
#   "bottom20"  -> 20% ล่างสุดของทุกคลาส
#   "both"      -> union ของสองเกณฑ์ (ครอบคลุมกว่า ใช้เป็นค่าเริ่มต้น)
WEAK_CLASS_RULE = "both"

# metric ที่ใช้วัดว่าคลาสไหน "อ่อนแอ" - per-class F1 (ไม่ใช่ recall) เพราะ F1
# รวมทั้งการทายไม่เจอ (recall ต่ำ) และการทายเกิน (precision ต่ำ) ไว้ในตัวเดียว
WEAK_CLASS_METRIC = "f1-score"

# metric ที่ใช้ตัดสินว่า checkpoint ไหน "ดีที่สุด" ในขั้นที่ 2 และ 4
#   "macro_f1" -> เฉลี่ย F1 ทุกคลาสเท่ากัน (ค่าเริ่มต้น: เหมาะกับข้อมูลไม่สมดุล)
#   "acc"      -> accuracy รวม ซึ่งลำเอียงเข้าข้างคลาสใหญ่
# accuracy ยังถูก log และพล็อตไว้ทุก epoch ตามที่โจทย์กำหนด
BEST_METRIC = "macro_f1"

# คลาสที่จำนวนภาพ >= target แล้ว จะเพิ่มภาพ augment เบา ๆ กี่ % (สเปกบอกว่า
# ไม่บังคับ) 0.0 = ไม่เพิ่ม ทำให้ทุกคลาสเท่ากันพอดี = balanced จริง
LIGHT_EXTRA_FRAC = 0.0

# True  = ภาพ augment ชุดเดียวกันทุก epoch (reproduce ได้ 100% ตรงตามแนวคิด
#         "สร้าง balanced augmented training set ขึ้นมาหนึ่งชุด")
# False = สุ่ม augment ใหม่ทุก epoch (ความหลากหลายสูงกว่า แต่ไม่ใช่ dataset
#         ชุดเดิมซ้ำ ๆ)
FIXED_AUGMENTATION = True


def ensure_dirs() -> None:
    """สร้างโฟลเดอร์ output ทั้งหมด (idempotent)"""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """ตั้ง random seed ทุกตัวเพื่อให้ reproduce ผลได้

    deterministic=True จะบังคับให้ cuDNN เลือก algorithm ที่ผลออกมาเหมือนเดิม
    ทุกครั้ง แต่ช้าลงพอสมควร - ค่าเริ่มต้นจึงเป็น False (benchmark mode)
    ซึ่ง reproduce ได้ในระดับ "ใกล้เคียงมาก" แต่ไม่ bit-exact บน GPU
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device():
    """ใช้ GPU ถ้ามี ไม่มีก็ CPU"""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device() -> str:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return (
            f"CUDA: {props.name}, VRAM {props.total_memory / 1024**3:.1f} GB, "
            f"torch {torch.__version__}"
        )
    return f"CPU only, torch {torch.__version__}"
