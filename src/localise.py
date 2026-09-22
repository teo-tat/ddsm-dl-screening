"""Lesion localisation: the letterbox, box utilities, the U-Nets, their losses and
augmentation, and tiled inference. The canvas keeps the mammogram's aspect ratio;
padding to a square would waste about 40% of the input."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy import ndimage
from . import config
from .geometry import Letterbox, area_downsample, tile_positions  # noqa: F401

Box = tuple[int, int, int, int]  # (y0, y1, x0, x1), inclusive


def letterbox(
    arr: np.ndarray,
    out_h: int = config.LOC_INPUT_H,
    out_w: int = config.LOC_INPUT_W,
    *,
    is_mask: bool = False,
    fill: float = 0.0,
) -> tuple[np.ndarray, Letterbox]:
    """Aspect-preserving resize into an (out_h, out_w) canvas, centre-padded. Masks
    resize as float and re-threshold at 0.5, since nearest neighbour aliases."""
    h, w = arr.shape[:2]
    lb = Letterbox.for_shape(h, w, out_h, out_w)
    new_h, new_w = lb.content_h, lb.content_w

    with tf.device("/CPU:0"):
        resized = tf.image.resize(
            arr[..., np.newaxis].astype(np.float32), [new_h, new_w], antialias=True
        ).numpy()[..., 0]
    if is_mask:
        resized = (resized > 0.5).astype(np.float32)

    canvas = np.full((out_h, out_w), fill, dtype=np.float32)
    canvas[lb.pad_y : lb.pad_y + new_h, lb.pad_x : lb.pad_x + new_w] = resized
    return canvas, lb


def mask_to_box(
    prob: np.ndarray,
    threshold: float = config.LOC_MASK_THRESHOLD,
    min_px: int = config.LOC_MIN_COMPONENT_PX,
    keep_largest: bool = config.LOC_KEEP_LARGEST_COMPONENT,
) -> Box | None:
    """Box of the largest thresholded component (all foreground if not keep_largest), or
    None under min_px; one component keeps a stray blob from stretching the box."""
    binary = np.asarray(prob) > threshold
    if not binary.any():
        return None

    if keep_largest:
        lab, n = ndimage.label(binary)
        if n == 0:
            return None
        sizes = np.bincount(lab.ravel())[1:]
        if sizes.max() < min_px:
            return None
        binary = lab == (int(sizes.argmax()) + 1)
    elif int(binary.sum()) < min_px:
        return None

    rows, cols = np.any(binary, axis=1), np.any(binary, axis=0)
    y = np.flatnonzero(rows)
    x = np.flatnonzero(cols)
    return int(y[0]), int(y[-1]), int(x[0]), int(x[-1])


def topk_components(
    prob: np.ndarray,
    k: int = config.LOC_TOPK,
    threshold: float = config.LOC_MASK_THRESHOLD,
    min_px: int = config.LOC_MIN_COMPONENT_PX,
) -> list[tuple[Box, float, int]]:
    """Up to k candidate boxes in canvas coordinates, best first, ranked by summed
    probability: candidate 1 is usually but not always the largest component."""
    binary = np.asarray(prob) > threshold
    if not binary.any():
        return []
    lab, n = ndimage.label(binary)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())[1:]
    sums = ndimage.sum(np.asarray(prob, np.float64), lab, index=np.arange(1, n + 1))
    keep = [i for i in range(n) if sizes[i] >= min_px]
    keep.sort(key=lambda i: -float(sums[i]))
    out = []
    for i in keep[:k]:
        comp = lab == (i + 1)
        y = np.flatnonzero(comp.any(axis=1))
        x = np.flatnonzero(comp.any(axis=0))
        out.append(((int(y[0]), int(y[-1]), int(x[0]), int(x[-1])), float(sums[i]), int(sizes[i])))
    return out


def box_iou(a: Box, b: Box) -> float:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    ih = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    iw = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    inter = ih * iw
    area_a = (ay1 - ay0 + 1) * (ax1 - ax0 + 1)
    area_b = (by1 - by0 + 1) * (bx1 - bx0 + 1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def box_centroid_hit(pred: Box, truth: Box) -> bool:
    """True if the predicted box's centre falls inside the true box: laxer than
    IoU >= 0.5, it asks only whether the lesion is in frame."""
    cy = (pred[0] + pred[1]) / 2
    cx = (pred[2] + pred[3]) / 2
    return truth[0] <= cy <= truth[1] and truth[2] <= cx <= truth[3]


def expand_box(box: Box, margin: float, shape: tuple[int, int]) -> Box:
    """Expand a box by `margin` of its side each way, clamped to shape. Matches
    data_loader._crop_to_mask_bbox, so oracle and predicted boxes share geometry."""
    y0, y1, x0, x1 = box
    h, w = (y1 - y0 + 1), (x1 - x0 + 1)
    my, mx = int(h * margin), int(w * margin)
    return (
        max(0, y0 - my),
        min(shape[0] - 1, y1 + my),
        max(0, x0 - mx),
        min(shape[1] - 1, x1 + mx),
    )


# Model


def _conv_block(x, filters: int, name: str, l2: float = 0.0):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    for i in (1, 2):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
            name=f"{name}_conv{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def build_unet(
    input_h: int = config.LOC_INPUT_H,
    input_w: int = config.LOC_INPUT_W,
    depth: int = config.LOC_DEPTH,
    base_filters: int = config.LOC_BASE_FILTERS,
    bottleneck_dropout: float = 0.3,
    output_prior: float | None = 0.0034,
) -> tf.keras.Model:
    """Encoder-decoder U-Net (Ronneberger et al., 2015) for lesion segmentation;
    the output stays float32, since fp16 sigmoids underflow and Dice collapses."""
    if input_h % (2**depth) or input_w % (2**depth):
        raise ValueError(f"({input_h}, {input_w}) not divisible by {2 ** depth}")

    inputs = tf.keras.Input((input_h, input_w, 1), name="mammogram")
    x, skips = inputs, []
    for d in range(depth):
        x = _conv_block(x, base_filters * 2**d, f"enc{d + 1}")
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2, name=f"enc{d + 1}_pool")(x)

    x = _conv_block(x, base_filters * 2**depth, "bottleneck")
    if bottleneck_dropout:
        x = tf.keras.layers.Dropout(bottleneck_dropout, name="bottleneck_drop")(x)

    for d in reversed(range(depth)):
        x = tf.keras.layers.Conv2DTranspose(
            base_filters * 2**d, 2, strides=2, padding="same", name=f"dec{d + 1}_up"
        )(x)
        x = tf.keras.layers.Concatenate(name=f"dec{d + 1}_skip")([x, skips[d]])
        x = _conv_block(x, base_filters * 2**d, f"dec{d + 1}")

    # Foreground is a small fraction of the canvas, so a zero bias would start the
    # sigmoid at 0.5 everywhere: the bias is initialised to the prior's logit.
    bias_init = (
        "zeros"
        if output_prior is None
        else tf.keras.initializers.Constant(float(np.log(output_prior / (1.0 - output_prior))))
    )
    outputs = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", dtype="float32", bias_initializer=bias_init, name="lesion_prob"
    )(x)
    return tf.keras.Model(inputs, outputs, name="unet_localiser")


# ImageNet-pretrained encoder U-Net

# Skip taps per encoder, shallowest first: strides 1, 2, 4, 8, 16. VGG16 has a
# full-resolution tap, which lesions under 32 px need; ResNet, DenseNet and
# EfficientNet start at stride 2.
_ENCODER_TAPS: dict[str, tuple[str, ...]] = {
    "vgg16": ("block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3")
}
_ENCODER_LAYER_PREFIX: dict[str, str] = {"vgg16": "block"}
_CAFFE_BGR_MEAN = (103.939, 116.779, 123.68)


def build_unet_pretrained(
    encoder: str = config.LOC_ENCODER,
    input_h: int | None = None,
    input_w: int | None = None,
    bottleneck_dropout: float = 0.3,
    output_prior: float | None = 0.0034,
    decoder_dropout: float = 0.0,
    decoder_l2: float = 0.0,
) -> tf.keras.Model:
    """U-Net with an ImageNet-pretrained VGG16 encoder (TernausNet-style), fully
    convolutional: the same weights train on patches and infer on the canvas."""
    if encoder not in _ENCODER_TAPS:
        raise ValueError(f"No pretrained encoder '{encoder}'; choose from {list(_ENCODER_TAPS)}")
    for d in (input_h, input_w):
        if d is not None and d % 32:
            raise ValueError(f"input dimension {d} not divisible by 32")

    inputs = tf.keras.Input((input_h, input_w, 1), name="mammogram")
    mean = tf.constant(_CAFFE_BGR_MEAN, tf.float32)
    x = tf.keras.layers.Lambda(
        lambda t: tf.repeat(t * 255.0, 3, axis=-1)[..., ::-1] - mean, name="caffe_preprocess"
    )(inputs)
    if encoder == "vgg16":
        backbone = tf.keras.applications.VGG16(
            include_top=False, weights="imagenet", input_tensor=x
        )
    # VGG16 has no normalisation, so its Caffe-scale taps would swamp the decoder
    # features; a BatchNorm per tap puts both branches on one scale.
    taps = [
        tf.keras.layers.BatchNormalization(name=f"tap{i + 1}_bn")(backbone.get_layer(n).output)
        for i, n in enumerate(_ENCODER_TAPS[encoder])
    ]
    # block5_pool, stride 32
    x = tf.keras.layers.BatchNormalization(name="tap6_bn")(backbone.output)

    x = _conv_block(x, 512, "bottleneck", l2=decoder_l2)
    if bottleneck_dropout:
        x = tf.keras.layers.Dropout(bottleneck_dropout, name="bottleneck_drop")(x)

    filters = (32, 64, 128, 256, 512)  # per tap, shallow -> deep
    # Decoder dropout and L2 add no weights, so a model built with or without
    # them loads the same checkpoint.
    reg = tf.keras.regularizers.l2(decoder_l2) if decoder_l2 else None
    for d in reversed(range(len(taps))):
        x = tf.keras.layers.Conv2DTranspose(
            filters[d], 2, strides=2, padding="same", kernel_regularizer=reg, name=f"dec{d + 1}_up"
        )(x)
        x = tf.keras.layers.Concatenate(name=f"dec{d + 1}_skip")([x, taps[d]])
        x = _conv_block(x, filters[d], f"dec{d + 1}", l2=decoder_l2)
        if decoder_dropout:
            x = tf.keras.layers.Dropout(decoder_dropout, name=f"dec{d + 1}_drop")(x)

    bias_init = (
        "zeros"
        if output_prior is None
        else tf.keras.initializers.Constant(float(np.log(output_prior / (1.0 - output_prior))))
    )
    outputs = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", dtype="float32", bias_initializer=bias_init, name="lesion_prob"
    )(x)
    return tf.keras.Model(inputs, outputs, name=f"unet_{encoder}_localiser")


# Elastic + photometric augmentation of a (patch, mask) pair


def elastic_photometric_pair(
    img: np.ndarray,
    msk: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: tuple[float, float],
    sigma: tuple[float, float],
    gamma: tuple[float, float],
    brightness: float,
    contrast: float,
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Elastic deformation applied jointly to image and mask, then photometric
    jitter on the image only. Returns the pair and the realised parameters."""
    h, w = img.shape[:2]
    a = float(rng.uniform(*alpha))
    s = float(rng.uniform(*sigma))
    # The displacement field (Simard et al., 2003) is drawn at 1/8 resolution and
    # up-sampled: for sigma >= 8 px it is nearly the same field, on 1/64 of the pixels.
    coarse = 8
    field = rng.standard_normal((2, h // coarse + 1, w // coarse + 1)).astype(np.float32)
    field = ndimage.gaussian_filter(field, sigma=(0, s / coarse, s / coarse), mode="reflect")
    field = ndimage.zoom(field, (1, coarse, coarse), order=1)[:, :h, :w]
    field *= a / (np.abs(field).mean() + 1e-6)  # mean |displacement| == alpha px
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    coords = np.stack([yy + field[0], xx + field[1]])
    img2 = ndimage.map_coordinates(img.astype(np.float32), coords, order=1, mode="reflect")
    msk2 = ndimage.map_coordinates(
        msk.astype(np.float32), coords, order=0, mode="constant", cval=0.0
    )

    g = float(rng.uniform(*gamma))
    b = float(rng.uniform(-brightness, brightness))
    c = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
    img2 = np.clip(img2, 0.0, 1.0) ** g
    m = float(img2.mean())
    img2 = (img2 - m) * c + m + b
    if noise_sd:
        img2 = img2 + rng.normal(0.0, noise_sd, img2.shape).astype(np.float32)
    img2 = np.clip(img2, 0.0, 1.0).astype(np.float32)
    return (
        img2,
        (msk2 > 0.5).astype(np.float32),
        dict(
            elastic_alpha=a,
            elastic_sigma=s,
            gamma=g,
            brightness=b,
            contrast=c,
            mean_disp_px=float(np.abs(field).mean()),
        ),
    )


def encoder_decoder_variables(
    model: tf.keras.Model, encoder: str = config.LOC_ENCODER
) -> tuple[list, list]:
    """Split trainable variables into (encoder, decoder) by layer-name prefix."""
    prefix = _ENCODER_LAYER_PREFIX[encoder]
    enc, dec = [], []
    for layer in model.layers:
        if not layer.trainable_weights:
            continue
        (enc if layer.name.startswith(prefix) else dec).extend(layer.trainable_weights)
    if not enc or not dec:
        raise RuntimeError(f"variable split failed: {len(enc)} encoder, {len(dec)} decoder")
    return enc, dec


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to `peak` over `warmup_steps`, then cosine to `floor` at `total_steps`."""

    def __init__(self, peak: float, floor: float, warmup_steps: int, total_steps: int):
        self.peak, self.floor = float(peak), float(floor)
        self.warmup_steps, self.total_steps = int(warmup_steps), int(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak * (step + 1.0) / float(max(1, self.warmup_steps))
        prog = tf.clip_by_value(
            (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps)),
            0.0,
            1.0,
        )
        cos = self.floor + 0.5 * (self.peak - self.floor) * (1.0 + tf.cos(np.pi * prog))
        return tf.where(step < self.warmup_steps, warm, cos)

    def get_config(self) -> dict:
        return dict(
            peak=self.peak,
            floor=self.floor,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
        )


class TwoRateModel(tf.keras.Model):
    """Keras Model applying two optimisers to two variable groups: the decoder's
    is compile()'s, the encoder's is attached by set_encoder_optimizer()."""

    def set_encoder_optimizer(self, optimizer, encoder_prefix: str | None) -> None:
        """encoder_prefix=None means one group (everything on compile's optimiser)."""
        self.encoder_optimizer = optimizer
        self._encoder_prefix = encoder_prefix

    # Derived inside the step, not stored: variable lists held as attributes are
    # tracked by Keras and change the saved weight layout.
    def _groups(self) -> tuple[list, list]:
        enc, dec = [], []
        for layer in self.layers:
            if not layer.trainable_weights:
                continue
            is_enc = self._encoder_prefix is not None and layer.name.startswith(
                self._encoder_prefix
            )
            (enc if is_enc else dec).extend(layer.trainable_weights)
        return enc, dec

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        enc_vars, dec_vars = self._groups()
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        grads = tape.gradient(loss, enc_vars + dec_vars)
        n = len(enc_vars)
        if n:
            self.encoder_optimizer.apply_gradients(zip(grads[:n], enc_vars))
        self.optimizer.apply_gradients(zip(grads[n:], dec_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# Losses and metrics


def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    """Per-sample Dice, averaged over the batch: pooling over the flattened batch
    lets one large lesion dominate the gradient for a batch of small ones."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [1, 2, 3]
    num = 2.0 * tf.reduce_sum(y_true * y_pred, axis=axes) + smooth
    den = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) + smooth
    return tf.reduce_mean(num / den)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """BCE plus soft Dice, weighted by config.LOC_DICE_BCE_WEIGHTS: foreground is a small
    fraction of the pixels, so BCE alone converges to all-background."""
    w_dice, w_bce = config.LOC_DICE_BCE_WEIGHTS
    bce = tf.keras.losses.binary_crossentropy(
        tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    )
    return w_bce * tf.reduce_mean(bce) + w_dice * dice_loss(y_true, y_pred)


def hard_iou(y_true, y_pred, threshold: float = config.LOC_MASK_THRESHOLD):
    """Thresholded IoU - the quantity the box derivation depends on."""
    y_true = tf.cast(tf.cast(y_true, tf.float32) > 0.5, tf.float32)
    y_pred = tf.cast(tf.cast(y_pred, tf.float32) > threshold, tf.float32)
    axes = [1, 2, 3]
    inter = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) - inter
    return tf.reduce_mean(inter / (union + 1e-7))


# Paired augmentation


_LOC_AUG_ROTATION = tf.keras.layers.RandomRotation(
    factor=config.AUG_MAX_ROTATION_DEG / 360.0,
    fill_mode="constant",
    fill_value=0.0,
    name="loc_aug_rotation",
)


def _augment_pair(image, mask):
    """Flip and rotate image and mask identically by concatenating them on the
    channel axis; rotation interpolates, so the mask is re-thresholded."""
    # Horizontal flip only: mammograms have a fixed superior-inferior orientation.
    pair = tf.concat([image, mask], axis=-1)
    pair = tf.image.random_flip_left_right(pair)
    pair = _LOC_AUG_ROTATION(pair, training=True)
    return pair[..., :1], tf.cast(pair[..., 1:] > 0.5, tf.float32)


# Tiled inference


def predict_tiled(
    model, img: np.ndarray, *, patch: int = config.LOC_PATCH_SIZE, overlap: int = 64, batch: int = 4
) -> np.ndarray:
    """Probability map for one image, predicted in overlapping tiles at the scale
    the network was trained at; the overlap is averaged so no tile edge shows."""
    h, w = img.shape[:2]
    ys, xs = tile_positions(h, patch, overlap), tile_positions(w, patch, overlap)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    windows = [(y, x) for y in ys for x in xs]
    for i in range(0, len(windows), batch):
        chunk = windows[i : i + batch]
        block = np.stack([img[y : y + patch, x : x + patch] for y, x in chunk]).astype(np.float32)
        # Eager call, never model.predict: tensorflow-metal drops ReLU from the
        # compiled graph.
        pred = np.asarray(model(block[..., None], training=False))[..., 0]
        for (y, x), p in zip(chunk, pred):
            acc[y : y + patch, x : x + patch] += p
            cnt[y : y + patch, x : x + patch] += 1.0
    return acc / np.maximum(cnt, 1.0)
