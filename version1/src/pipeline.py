"""
pipeline.py
============
Turns the (filepath, class_id, split) table into cached numpy arrays and
tf.data.Dataset objects, with on-the-fly augmentation applied only to the
training split.

Why precompute/cache resized arrays instead of decoding JPEGs every epoch?
The source images are extremely small (roughly 12x18 to 30x45 px) and there
are ~30,000 of them — resized to 96x96x3 uint8 they comfortably fit in
memory (~830 MB), so decoding each JPEG from disk on every single epoch
would be a pure, avoidable bottleneck on top of everything else being
CPU-bound in this environment. We decode + resize once, cache to disk as a
single .npz per split, and reuse it for every notebook run afterwards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.augmentation import load_and_resize, augment_image
from src.data_utils import CACHE_DIR

IMG_SIZE = 96


def build_label_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    classes = sorted(df["class_id"].unique(), key=lambda c: str(c))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def _cache_path(split: str, img_size: int) -> Path:
    return CACHE_DIR / f"{split}_{img_size}.npz"


def build_or_load_cache(df: pd.DataFrame, split: str, class_to_idx: dict, img_size: int = IMG_SIZE, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [N,H,W,3], labels int32 [N]) for the given split,
    building + caching to disk on first call."""
    path = _cache_path(split, img_size)
    if path.exists() and not force:
        data = np.load(path)
        return data["images"], data["labels"]

    sub = df[df["split"] == split].reset_index(drop=True)
    images = np.zeros((len(sub), img_size, img_size, 3), dtype=np.uint8)
    labels = np.zeros((len(sub),), dtype=np.int32)
    filepaths = sub["filepath"].tolist()
    class_ids = sub["class_id"].tolist()
    for i in tqdm(range(len(sub)), total=len(sub), desc=f"Caching {split}"):
        images[i] = load_and_resize(filepaths[i], size=img_size)
        labels[i] = class_to_idx[class_ids[i]]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, labels=labels)
    return images, labels


def _augment_map_fn(image, label):
    def _apply(img):
        img_np = img.numpy()
        return augment_image(img_np, apply_prob=0.5)

    aug = tf.py_function(func=_apply, inp=[image], Tout=tf.uint8)
    aug.set_shape(image.shape)
    return aug, label


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    training: bool = False,
    shuffle_buffer: int = 4096,
    seed: int = 42,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(images)), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(_augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
