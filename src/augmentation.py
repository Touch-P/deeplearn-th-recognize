"""
augmentation.py
================
Data-augmentation techniques tailored for handwritten Thai characters and
digits.

Design rule (explained in notebooks/01_preprocessing_augmentation.ipynb):
we deliberately **never use horizontal or vertical flips**. Flipping a
character can turn it into a different, unrelated glyph (or something with
no meaning at all) — this would inject wrong training signal for a
character-recognition task. Every technique below only perturbs the image
in ways a real handwritten sample could plausibly vary (small rotation,
position shift, scale, shear, stroke-width/contrast, sensor noise, and
locally missing ink), while preserving the character's identity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image, ImageEnhance


def to_rgb_uint8(img: Image.Image) -> Image.Image:
    """Ensure a PIL image is 3-channel RGB uint8 (dataset images are grayscale)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_and_resize(filepath: str, size: int = 96) -> np.ndarray:
    """Load an image from disk, convert to RGB, resize to (size, size)."""
    with Image.open(filepath) as im:
        im = to_rgb_uint8(im)
        im = im.resize((size, size), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Individual augmentation techniques (each takes/returns a uint8 HWC array)
# ---------------------------------------------------------------------------

def random_rotation(img: np.ndarray, max_degrees: float = 12.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Small-angle rotation (+/- max_degrees). Simulates natural handwriting tilt."""
    rng = rng or np.random
    angle = rng.uniform(-max_degrees, max_degrees)
    pil = Image.fromarray(img)
    pil = pil.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return np.asarray(pil, dtype=np.uint8)


def random_shift(img: np.ndarray, max_frac: float = 0.08, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Random width/height shift up to max_frac of the image size."""
    rng = rng or np.random
    h, w = img.shape[:2]
    dx = int(rng.uniform(-max_frac, max_frac) * w)
    dy = int(rng.uniform(-max_frac, max_frac) * h)
    pil = Image.fromarray(img)
    pil = pil.transform(
        (w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
        resample=Image.BICUBIC, fillcolor=(255, 255, 255),
    )
    return np.asarray(pil, dtype=np.uint8)


def random_zoom(img: np.ndarray, zoom_range: tuple[float, float] = (0.9, 1.1), rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Small random zoom in/out, re-centered and cropped/padded back to original size."""
    rng = rng or np.random
    h, w = img.shape[:2]
    scale = rng.uniform(*zoom_range)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    pil = Image.fromarray(img).resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    off_x = (w - new_w) // 2
    off_y = (h - new_h) // 2
    if scale <= 1.0:
        canvas.paste(pil, (off_x, off_y))
    else:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        pil = pil.crop((left, top, left + w, top + h))
        canvas.paste(pil, (0, 0))
    return np.asarray(canvas, dtype=np.uint8)


def random_shear(img: np.ndarray, max_shear: float = 0.15, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Small random shear transform — simulates slanted handwriting."""
    rng = rng or np.random
    h, w = img.shape[:2]
    shear = rng.uniform(-max_shear, max_shear)
    pil = Image.fromarray(img)
    pil = pil.transform(
        (w, h), Image.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0),
        resample=Image.BICUBIC, fillcolor=(255, 255, 255),
    )
    return np.asarray(pil, dtype=np.uint8)


def random_brightness_contrast(
    img: np.ndarray,
    brightness_range: tuple[float, float] = (0.85, 1.15),
    contrast_range: tuple[float, float] = (0.85, 1.15),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Randomly perturb brightness and contrast (simulates ink darkness / scan lighting)."""
    rng = rng or np.random
    pil = Image.fromarray(img)
    pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(*brightness_range))
    pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(*contrast_range))
    return np.asarray(pil, dtype=np.uint8)


def gaussian_noise(img: np.ndarray, sigma: float = 8.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Additive Gaussian sensor/scan noise."""
    rng = rng or np.random
    noise = rng.normal(0, sigma, size=img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def elastic_transform(
    img: np.ndarray,
    alpha: float = 8.0,
    sigma: float = 3.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Elastic deformation (Simard et al.) — simulates the natural wobble/
    unevenness of handwritten strokes without changing the character's
    topology (unlike a flip, which changes identity)."""
    rng = rng or np.random
    h, w = img.shape[:2]
    dx = gaussian_filter((rng.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((rng.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices_x = np.reshape(x + dx, (-1, 1))
    indices_y = np.reshape(y + dy, (-1, 1))
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = map_coordinates(
            img[..., c], [indices_y.reshape(h, w), indices_x.reshape(h, w)],
            order=1, mode="constant", cval=255,
        ).reshape(h, w)
    return out.astype(np.uint8)


def random_erasing(
    img: np.ndarray,
    max_area_frac: float = 0.12,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Randomly erase (whiten) a small rectangular patch — cutout. Simulates
    partially faded ink / scan artifacts; forces the model not to rely on
    any single stroke fragment."""
    rng = rng or np.random
    h, w = img.shape[:2]
    area = rng.uniform(0.02, max_area_frac) * h * w
    aspect = rng.uniform(0.5, 2.0)
    eh = int(round(np.sqrt(area * aspect)))
    ew = int(round(np.sqrt(area / aspect)))
    eh, ew = min(eh, h - 1), min(ew, w - 1)
    if eh <= 0 or ew <= 0:
        return img
    y0 = rng.randint(0, h - eh)
    x0 = rng.randint(0, w - ew)
    out = img.copy()
    out[y0:y0 + eh, x0:x0 + ew, :] = 255
    return out


# Techniques applied to every training image, in order, each with its own
# random per-call parameters. Kept mild/composable since several are
# stacked on top of each other for every sample.
DEFAULT_PIPELINE = [
    ("rotation", lambda im, rng: random_rotation(im, 12.0, rng)),
    ("shift", lambda im, rng: random_shift(im, 0.08, rng)),
    ("zoom", lambda im, rng: random_zoom(im, (0.9, 1.1), rng)),
    ("shear", lambda im, rng: random_shear(im, 0.15, rng)),
    ("brightness_contrast", lambda im, rng: random_brightness_contrast(im, rng=rng)),
    ("gaussian_noise", lambda im, rng: gaussian_noise(im, 6.0, rng)),
    ("elastic", lambda im, rng: elastic_transform(im, 6.0, 3.0, rng)),
    ("random_erasing", lambda im, rng: random_erasing(im, 0.10, rng)),
]


def augment_image(img: np.ndarray, rng: Optional[np.random.RandomState] = None, apply_prob: float = 0.5) -> np.ndarray:
    """Apply the full augmentation pipeline; each technique is applied
    independently with probability ``apply_prob`` so not every sample gets
    every distortion stacked at full strength."""
    rng = rng or np.random
    out = img
    for _name, fn in DEFAULT_PIPELINE:
        if rng.rand() < apply_prob:
            out = fn(out, rng)
    return out
