"""
model_utils.py
===============
Transfer-learning CNN builder + Focal Loss (our chosen "extra technique"
for handling class imbalance, see notebooks/02_model_transfer_learning.ipynb)
+ small plotting helpers reused across notebooks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

BackboneName = Literal["mobilenetv2", "efficientnetb0"]


def get_preprocess_fn(name: BackboneName):
    """Return just the backbone-specific preprocess_input function, without
    constructing (and downloading weights for) a model instance."""
    if name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return preprocess_input


def get_backbone(name: BackboneName, input_shape=(96, 96, 3)):
    """Return (backbone_model, preprocess_fn) for a supported ImageNet backbone."""
    if name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        backbone = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    elif name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        backbone = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return backbone, preprocess_input


def build_transfer_model(
    backbone_name: BackboneName,
    num_classes: int,
    input_shape=(96, 96, 3),
    dropout: float = 0.3,
    dense_units: int = 256,
    freeze_backbone: bool = True,
):
    """Build: Input -> preprocess -> backbone -> GAP -> Dense -> Dropout -> Dense(softmax).

    Returns (model, backbone) so the caller can later flip
    ``backbone.trainable`` for the fine-tuning phase.
    """
    backbone, preprocess_input = get_backbone(backbone_name, input_shape)
    backbone.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(preprocess_input, name="preprocess")(inputs)
    x = backbone(x, training=False if freeze_backbone else None)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"{backbone_name}_transfer")
    return model, backbone


def unfreeze_top_layers(backbone, n_layers: int = 30):
    """Unfreeze only the last ``n_layers`` of the backbone for fine-tuning,
    keeping earlier (more generic edge/texture) layers frozen. BatchNorm
    layers are kept frozen regardless, which is standard practice when
    fine-tuning with small batch sizes."""
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - n_layers)
    for i, layer in enumerate(backbone.layers):
        if i < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    return backbone


# ---------------------------------------------------------------------------
# Focal Loss — chosen technique for class imbalance (notebook 02 & 03)
# ---------------------------------------------------------------------------

def sparse_categorical_focal_loss(gamma: float = 2.0, alpha: float | None = None):
    """Focal Loss (Lin et al., 2017) for sparse integer labels.

    Standard cross-entropy weighs every sample equally regardless of how
    confidently the model already classifies it. With 31 classes ranging
    from ~30 to ~4,800 images, plain cross-entropy is dominated by the
    majority classes' easy, already-correct predictions. Focal loss adds a
    ``(1 - p_t) ** gamma`` modulating factor that down-weights easy/
    well-classified examples and keeps gradient signal focused on hard/
    minority-class examples, which directly targets this dataset's
    imbalance without discarding any data (unlike undersampling) and
    without inflating the effective dataset size (unlike oversampling).
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, num_classes)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        ce = -tf.math.log(p_t)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = modulating_factor * ce
        if alpha is not None:
            loss = alpha * loss
        return tf.reduce_mean(loss)

    return loss_fn
