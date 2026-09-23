"""
torch_pipeline/augment.py
=========================
Transform สำหรับลายมือภาษาไทย 3 ระดับความเข้ม

กฎเหล็ก: ห้าม Horizontal/Vertical Flip
----------------------------------------
ตัวอักษรไทยมีทิศทาง พลิกซ้าย-ขวาแล้วกลายเป็นอักขระอื่นหรือไม่มีความหมายเลย
เช่น "ก" พลิกแล้วคล้าย "ฎ/ฏ", "เ" พลิกแล้วได้รูปที่ไม่มีในภาษาไทย การ flip
จึงเป็นการป้อน label ผิดให้โมเดลโดยตรง ทุก transform ในไฟล์นี้จึงเป็นการ
"กวน" ภาพเท่าที่ลายมือจริงของคนจะต่างกันได้ (เอียงเล็กน้อย เลื่อนตำแหน่ง
ย่อ-ขยาย เส้นหนา-บาง ความสว่าง สัญญาณรบกวน หมึกขาดหาย) โดยไม่เปลี่ยนตัวตน
ของตัวอักษร

ระดับความเข้ม
-------------
light   : สำหรับคลาสที่ข้อมูลเยอะเกิน target แล้ว - เพิ่มความหลากหลายบาง ๆ
normal  : สำหรับคลาสที่ข้อมูลน้อยแต่โมเดลยังทำได้ดี
intense : สำหรับ "คลาสที่ทายผิดบ่อย" จาก baseline (ขั้นที่ 2) - เพิ่ม
          ElasticTransform, morphological, RandomErasing และเพิ่มพิสัยของ
          ทุกพารามิเตอร์ เพื่อให้โมเดลเห็นความแปรผันของคลาสยาก ๆ มากขึ้น

หมายเหตุเรื่องสีพื้น: ภาพในชุดนี้เป็นหมึกเข้มบนพื้นขาว ทุก transform ที่ต้อง
เติมพื้นที่ว่าง (หมุน/เลื่อน/elastic) จึงเติมด้วยสีขาว (255) ถ้าเติมดำจะเกิด
ขอบดำปลอมที่โมเดลอาจเรียนรู้เป็น feature
"""

from __future__ import annotations

import random

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T

from torch_pipeline import config as cfg

WHITE = 255  # fill สำหรับ transform เชิงเรขาคณิต


# ---------------------------------------------------------------------------
# Transform ที่ torchvision ไม่มีให้ ต้องเขียนเอง
# ---------------------------------------------------------------------------
class PadToSquare:
    """เติมขอบสีขาวให้ภาพเป็นจัตุรัส (รักษาสัดส่วนตัวอักษรก่อน resize)"""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (WHITE, WHITE, WHITE))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


class RandomMorphology:
    """dilate/erode ของหมึก = ทำให้เส้นหนาขึ้น/บางลง

    ภาพเป็นหมึกเข้มบนพื้นขาว ดังนั้น
      MinFilter -> ค่าต่ำ (เข้ม) แพร่ออก = เส้นหนาขึ้น (dilate หมึก)
      MaxFilter -> ค่าสูง (ขาว) แพร่ออก = เส้นบางลง (erode หมึก)
    เลียนแบบการเขียนด้วยปากกาหัวใหญ่/เล็ก หรือกดหนัก/เบา ซึ่งเป็นความแปรผัน
    ที่พบมากที่สุดอย่างหนึ่งในลายมือจริง
    """

    def __init__(self, size: int = 3, p_thicken: float = 0.5):
        self.size = size
        self.p_thicken = p_thicken

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_thicken:
            return img.filter(ImageFilter.MinFilter(self.size))
        return img.filter(ImageFilter.MaxFilter(self.size))


class AddGaussianNoise:
    """สัญญาณรบกวนแบบเกาส์เซียนบน tensor ช่วง [0,1] (จำลอง noise ของกล้อง/สแกน)

    ต้องใส่ "หลัง" ToTensor และ "ก่อน" Normalize เพื่อให้ std ที่ตั้งไว้มี
    หน่วยเดียวกับความสว่างของภาพจริง
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ประกอบ pipeline
# ---------------------------------------------------------------------------
def _resize_head(img_size: int, pad_square: bool) -> list:
    head = [PadToSquare()] if pad_square else []
    head.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC))
    return head


def _normalize_tail() -> list:
    return [T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)]


def build_eval_transform(img_size: int = cfg.IMG_SIZE, pad_square: bool = cfg.PAD_TO_SQUARE):
    """ใช้กับ val / test / ภาพต้นฉบับใน training set - ไม่มีการสุ่มใด ๆ"""
    return T.Compose(_resize_head(img_size, pad_square) + [T.ToTensor()] + _normalize_tail())


def build_light_transform(img_size: int = cfg.IMG_SIZE, pad_square: bool = cfg.PAD_TO_SQUARE):
    """เบา ๆ: เอียงนิดเดียว เลื่อนนิดเดียว ปรับความสว่างนิดเดียว"""
    return T.Compose(
        _resize_head(img_size, pad_square)
        + [
            T.RandomRotation(6, fill=WHITE, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomAffine(0, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=WHITE),
            T.ColorJitter(brightness=0.10, contrast=0.10),
            T.ToTensor(),
            AddGaussianNoise(std=0.008),
        ]
        + _normalize_tail()
    )


def build_normal_transform(img_size: int = cfg.IMG_SIZE, pad_square: bool = cfg.PAD_TO_SQUARE):
    """ปกติ: มุมหมุน ±10 องศา เลื่อน/ย่อขยายเล็กน้อย เส้นหนา-บาง เบลอบาง ๆ"""
    return T.Compose(
        _resize_head(img_size, pad_square)
        + [
            T.RandomRotation(10, fill=WHITE, interpolation=T.InterpolationMode.BILINEAR),
            # scale ด้านบนหยุดที่ 1.05: ตัวอักษรกินพื้นที่เกือบเต็มเฟรมอยู่แล้ว
            # ถ้าซูมเข้ามากกว่านี้เส้นจะถูกตัดหายที่ขอบ (ตรวจจากภาพจริงแล้ว)
            # การซูมออก (0.88) ปลอดภัยเพราะได้ขอบขาวเพิ่มเหมือนเขียนตัวเล็กลง
            T.RandomAffine(
                0, translate=(0.05, 0.05), scale=(0.88, 1.05), shear=4, fill=WHITE
            ),
            T.RandomApply([RandomMorphology(3)], p=0.25),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.20),
            T.ColorJitter(brightness=0.12, contrast=0.12),
            T.ToTensor(),
            T.RandomApply([AddGaussianNoise(std=0.015)], p=0.30),
        ]
        + _normalize_tail()
    )


def build_intense_transform(img_size: int = cfg.IMG_SIZE, pad_square: bool = cfg.PAD_TO_SQUARE):
    """เข้มข้น (error-driven): ±15 องศา, ElasticTransform จำลองลายมือบิดเบี้ยว,
    morphological บ่อยขึ้น, RandomErasing เบา ๆ จำลองหมึกขาดหาย

    RandomErasing วางไว้ก่อน Normalize และเติมด้วยค่า 1.0 (ขาว) เพื่อให้เป็น
    "หมึกหาย" จริง ๆ ถ้าวางหลัง Normalize ค่า 0 จะกลายเป็นสีเทากลางภาพ
    ซึ่งไม่เหมือนอะไรที่เกิดขึ้นได้จริงบนกระดาษ
    """
    return T.Compose(
        _resize_head(img_size, pad_square)
        + [
            T.RandomRotation(15, fill=WHITE, interpolation=T.InterpolationMode.BILINEAR),
            # ดู build_normal_transform() เรื่องเพดาน scale = 1.05
            T.RandomAffine(
                0, translate=(0.08, 0.08), scale=(0.80, 1.05), shear=8, fill=WHITE
            ),
            T.RandomApply(
                [T.ElasticTransform(alpha=35.0, sigma=5.0, fill=WHITE)], p=0.50
            ),
            T.RandomApply([RandomMorphology(3)], p=0.45),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.30),
            # brightness/contrast แรง ๆ (0.3) ทำให้พื้นหลังกลายเป็นเทากลางภาพ
            # ซึ่งไม่เคยเกิดใน val/test ที่พื้นหลังขาวเสมอ = แจก distribution
            # ที่โมเดลจะเจอตอนใช้งานจริงผิดไป จึงคุมไว้ที่ 0.20
            T.ColorJitter(brightness=0.20, contrast=0.20),
            T.ToTensor(),
            T.RandomApply([AddGaussianNoise(std=0.025)], p=0.50),
            T.RandomErasing(p=0.15, scale=(0.01, 0.04), ratio=(0.3, 3.3), value=1.0),
        ]
        + _normalize_tail()
    )


def build_transforms(
    img_size: int = cfg.IMG_SIZE, pad_square: bool = cfg.PAD_TO_SQUARE
) -> dict[str, object]:
    """dict ที่ AugmentedManifestDataset ใช้เลือก transform ตาม aug_kind"""
    return {
        "original": build_eval_transform(img_size, pad_square),
        "light": build_light_transform(img_size, pad_square),
        "normal": build_normal_transform(img_size, pad_square),
        "intense": build_intense_transform(img_size, pad_square),
    }


# ---------------------------------------------------------------------------
# ใช้สำหรับพล็อตตัวอย่าง "ต้นฉบับ vs หลัง augment"
# ---------------------------------------------------------------------------
def denormalize(t: torch.Tensor) -> torch.Tensor:
    """แปลง tensor ที่ normalize แล้วกลับเป็นช่วง [0,1] เพื่อเอาไปแสดงภาพ"""
    mean = torch.tensor(cfg.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(t.cpu() * std + mean, 0.0, 1.0)


def augment_preview(path: str, kind: str, seed: int, img_size: int = cfg.IMG_SIZE):
    """คืนภาพ numpy HWC ของ path นี้หลังผ่าน transform ชนิด kind ด้วย seed ที่ระบุ
    (ใช้ seed เดียวกับใน manifest ผลจึงตรงกับที่โมเดลเห็นจริง)"""
    from torch_pipeline.data import load_rgb

    transform = build_transforms(img_size)[kind]
    img = load_rgb(path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random.seed(seed)
        t = transform(img)
    return denormalize(t).permute(1, 2, 0).numpy()
