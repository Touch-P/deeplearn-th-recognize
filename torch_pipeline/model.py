"""
torch_pipeline/model.py
=======================
ResNet50 transfer learning + training/validation/inference engine
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pipeline import config as cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """ResNet50 pretrained ImageNet โดยเปลี่ยน fc ชั้นสุดท้ายเป็น num_classes

    ทำไม fine-tune ทั้งโมเดล (freeze_backbone=False เป็นค่าเริ่มต้น):
    ImageNet เป็นภาพถ่ายธรรมชาติ (สัตว์ สิ่งของ ทิวทัศน์) ส่วนงานนี้เป็น
    ลายเส้นขาว-ดำของตัวอักษร ฟีเจอร์ชั้นกลาง-ปลายของ ImageNet (texture ขนสัตว์
    ลายผ้า สีของวัตถุ) แทบไม่ตรงกับงานนี้เลย ถ้า freeze ไว้แล้วเทรนแค่ fc
    โมเดลจะติดเพดานเร็ว การ unfreeze ทั้งหมดด้วย lr ต่ำ (1e-4) จึงให้ทุกชั้น
    ปรับตัวเข้าหาลายเส้นได้ ขณะที่ยังได้ประโยชน์จาก edge/stroke detector
    ชั้นต้น ๆ ที่ pretrained มาแล้ว (เทรนจากศูนย์บนข้อมูล 50k ภาพเล็ก ๆ จะ
    overfit ง่ายกว่ามาก)
    """
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features  # 2048 สำหรับ ResNet50
    model.fc = nn.Linear(in_features, num_classes)
    return model


def prepare_model(model: nn.Module, device, channels_last: bool = False) -> nn.Module:
    """ย้ายโมเดลขึ้น device และตั้ง memory format

    ดูเหตุผลที่ไม่ใช้ channels_last ตอนเทรนใน config.CHANNELS_LAST_TRAIN
    """
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def _to_device(images: torch.Tensor, device, channels_last: bool) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)
    return images


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Metric ที่ใช้ตัดสินใจ: macro F1
# ---------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> float:
    """macro F1 = เฉลี่ย F1 ของทุกคลาสโดยให้น้ำหนักเท่ากัน

    ใช้ตัวนี้เป็นเกณฑ์หลักแทน accuracy เพราะคลาสใหญ่สุดของ dataset นี้มีภาพ
    มากกว่าคลาสเล็กสุดหลายพันเท่า accuracy จะถูกคลาสใหญ่ครอบงำจนคลาสเล็ก
    พังทั้งคลาสก็ยังดูเหมือนโมเดลดี

    num_classes ที่ระบุมาทำให้คลาสที่ไม่ปรากฏใน batch/epoch นั้นถูกนับเป็น
    F1 = 0 ไม่ใช่ถูกข้ามไป ตัวเลขจึงเทียบกันได้ข้าม epoch
    """
    from sklearn.metrics import f1_score

    labels = np.arange(num_classes) if num_classes else None
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device,
    scaler=None,
    epoch: int = 0,
    total_epochs: int = 0,
    log_every: int = 50,
    channels_last: bool = cfg.CHANNELS_LAST_TRAIN,
    num_classes: int | None = None,
) -> dict:
    """เทรน 1 epoch คืน dict: loss, acc, macro_f1

    เก็บ prediction ของทั้ง epoch ไว้คำนวณ macro F1 ของ train ด้วย เพราะบน
    dataset ที่คลาสไม่สมดุล accuracy ของ train อย่างเดียวดูดีเกินจริงได้
    (คลาสใหญ่ถูกนับซ้ำ ๆ) ส่วน accuracy ยังเก็บและพล็อตไว้ตามที่โจทย์กำหนด
    """
    model.train()
    running_loss, correct, seen = 0.0, 0, 0
    epoch_true, epoch_pred = [], []
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"train {epoch}/{total_epochs}", leave=False)
    for step, (images, labels, _) in enumerate(pbar):
        images = _to_device(images, device, channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        preds = outputs.argmax(1)
        running_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        seen += bs
        epoch_true.append(labels.detach().cpu().numpy())
        epoch_pred.append(preds.detach().cpu().numpy())

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{correct / seen:.4f}")

    y_true = np.concatenate(epoch_true)
    y_pred = np.concatenate(epoch_pred)
    return {
        "loss": running_loss / seen,
        "acc": correct / seen,
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device,
    use_amp: bool = cfg.USE_AMP,
    return_probs: bool = False,
    desc: str = "val",
    channels_last: bool = cfg.CHANNELS_LAST_TRAIN,
    num_classes: int | None = None,
):
    """ประเมินบน loader คืน dict: loss, acc, macro_f1, y_true, y_pred, (probs)"""
    model.eval()
    running_loss, correct, seen = 0.0, 0, 0
    all_true, all_pred, all_probs, all_idx = [], [], [], []

    for images, labels, idx in tqdm(loader, desc=desc, leave=False):
        images = _to_device(images, device, channels_last)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels) if criterion is not None else torch.tensor(0.0)

        probs = torch.softmax(outputs.float(), dim=1)
        preds = probs.argmax(1)

        bs = labels.size(0)
        running_loss += float(loss.item()) * bs
        correct += (preds == labels).sum().item()
        seen += bs

        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_idx.append(idx.numpy())
        if return_probs:
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    out = {
        "loss": running_loss / max(1, seen),
        "acc": correct / max(1, seen),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "y_true": y_true,
        "y_pred": y_pred,
        "row_index": np.concatenate(all_idx),
    }
    if return_probs:
        out["probs"] = np.concatenate(all_probs)
    return out


# ---------------------------------------------------------------------------
# Checkpoint + training log
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_val_acc: float = 0.0,
    history: list[dict] | None = None,
    extra: dict | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "history": history or [],
        "arch": cfg.BACKBONE,
        "img_size": cfg.IMG_SIZE,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, model: nn.Module, device, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def start_log(log_path: Path) -> Path:
    """เริ่มไฟล์ log ใหม่ (ย้ายไฟล์เดิมเป็น .bak)

    append_log() เขียนต่อท้ายเสมอ ถ้าไม่เคลียร์ก่อน บรรทัดจากการรัน smoke test
    รอบก่อนจะปนกับรอบจริง แล้วกราฟ/ตารางจะอ่านผิด
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        backup = log_path.with_suffix(log_path.suffix + ".bak")
        backup.unlink(missing_ok=True)
        log_path.rename(backup)
    return log_path


def append_log(log_path: Path, record: dict) -> None:
    """เขียน training log ต่อท้ายทีละบรรทัด (JSON Lines) - ถ้าเทรนหลุดกลางทาง
    log ที่ผ่านมายังอยู่ครบ"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_eta(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class EpochTimer:
    """จับเวลาต่อ epoch และประมาณเวลาที่เหลือ"""

    def __init__(self, total_epochs: int):
        self.total = total_epochs
        self.times: list[float] = []
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.time()

    def stop(self) -> float:
        elapsed = time.time() - self._t0
        self.times.append(elapsed)
        return elapsed

    def eta(self, epochs_done: int) -> str:
        if not self.times:
            return "?"
        avg = sum(self.times) / len(self.times)
        return format_eta(avg * (self.total - epochs_done))
