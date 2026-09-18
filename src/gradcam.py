"""
gradcam.py
===========
Grad-CAM (Selvaraju et al., 2017) for our nested transfer-learning model —
our second "extra technique" (alongside Focal Loss) for the presentation.

Why Grad-CAM for this project: it visualizes *which pixels of the character*
the model actually looked at to make its decision. For handwritten Thai
character recognition this is a genuinely useful diagnostic — it can show,
for example, whether the model is focusing on a distinguishing loop/tail
(good) or on background noise/artifacts (bad), and is a strong visual for
a presentation slide.

Implementation note
--------------------
Our full model nests the ImageNet backbone as a single sub-model (so its
Lambda-preprocessed input can flow straight through it). Trying to build a
single static ``keras.Model(inputs=model.inputs, outputs=[conv_layer.output,
model.output])`` fails with "Graph disconnected", because
``backbone.get_layer(x).output`` resolves to the tensor from the backbone's
*own* standalone construction call (node 0), not the second call made when
it was embedded inside our model. Instead of fighting the static graph, we
rebuild the forward pass explicitly (eager, inside a GradientTape) by
reusing the exact same layer *objects* (so weights are shared, not copied)
for the head: feature_extractor(backbone) -> GAP -> Dense -> Dropout ->
Dense(softmax).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer_name(backbone: keras.Model) -> str:
    """Auto-detect the last Conv2D-ish layer inside a backbone (for Grad-CAM)."""
    for layer in reversed(backbone.layers):
        if len(layer.output_shape) == 4:  # (batch, H, W, C) feature map
            return layer.name
    raise ValueError("No 4D (conv-like) layer found in backbone.")


class GradCAMHelper:
    """Bundles a feature-extractor (backbone up to its last conv layer) with
    the trained model's own head layers (reused by reference, so no weights
    are duplicated), so we can run one eager forward pass under a
    GradientTape and get both the conv feature map and the final prediction.
    """

    def __init__(self, model: keras.Model, backbone: keras.Model, preprocess_input, conv_layer_name: str | None = None):
        if conv_layer_name is None:
            conv_layer_name = find_last_conv_layer_name(backbone)
        self.conv_layer_name = conv_layer_name
        self.preprocess_input = preprocess_input
        self.feature_extractor = keras.Model(
            inputs=backbone.input,
            outputs=backbone.get_layer(conv_layer_name).output,
            name="gradcam_feature_extractor",
        )
        self.gap = model.get_layer("gap")
        self.dense = model.get_layer("head_dense")
        self.dropout = model.get_layer("head_dropout")
        self.pred = model.get_layer("predictions")

    def preprocess(self, image_batch):
        x = tf.cast(image_batch, tf.float32)
        return self.preprocess_input(x)

    def head_forward(self, conv_out):
        h = self.gap(conv_out)
        h = self.dense(h)
        h = self.dropout(h, training=False)
        return self.pred(h)


def make_gradcam_helper(model: keras.Model, backbone: keras.Model, preprocess_input, conv_layer_name: str | None = None) -> GradCAMHelper:
    return GradCAMHelper(model, backbone, preprocess_input, conv_layer_name)


def compute_gradcam(helper: GradCAMHelper, image_batch: np.ndarray, class_index: int | None = None):
    """Compute a Grad-CAM heatmap for a single image (image_batch shape (1,H,W,3)).

    Returns (heatmap [Hc,Wc] normalized to [0,1], predicted_class_index, predicted_prob).
    """
    x = helper.preprocess(image_batch)
    with tf.GradientTape() as tape:
        conv_out = helper.feature_extractor(x, training=False)
        tape.watch(conv_out)
        predictions = helper.head_forward(conv_out)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # importance weight per channel
    conv_out0 = conv_out[0]
    heatmap = tf.reduce_sum(conv_out0 * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy(), class_index, float(predictions[0, class_index])


def overlay_heatmap(image_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """Resize heatmap to image size and overlay it (jet colormap) on the original image."""
    import matplotlib as mpl

    h, w = image_rgb_uint8.shape[:2]
    heatmap_img = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy()[..., 0]
    colored = mpl.colormaps["jet"](heatmap_img)[..., :3]  # RGBA -> RGB
    colored = (colored * 255).astype(np.uint8)
    overlay = (image_rgb_uint8.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
