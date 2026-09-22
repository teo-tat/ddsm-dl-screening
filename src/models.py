"""Model architectures for CBIS-DDSM mass classification: three from-scratch CNNs of
increasing capacity and four ImageNet-pretrained backbones (VGG16, ResNet50,
DenseNet121, EfficientNetB0), all sharing one classification head."""

from __future__ import annotations

import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from . import config

# Registry

MODEL_REGISTRY: dict[str, dict] = {
    "baseline": {"normalisation": "scratch", "channels": 1},
    "scaled": {"normalisation": "scratch", "channels": 1},
    "regularised": {"normalisation": "scratch", "channels": 1},
    "vgg16": {"normalisation": "imagenet", "channels": 3},
    "resnet50": {"normalisation": "imagenet", "channels": 3},
    "densenet121": {"normalisation": "imagenet", "channels": 3},
    "efficientnet": {"normalisation": "imagenet", "channels": 3},
}

# From-scratch models. train_baseline augments baseline and regularised (unless --ablate aug)
# on crops only, never on whole images, and never scaled; train_indomain augments every stage.
_SCRATCH_MODELS: frozenset[str] = frozenset({"baseline", "scaled", "regularised"})

# Layer at which fine-tuning starts unfreezing. Earlier layers stay frozen, as do
# BatchNormalization layers, whose ImageNet statistics this dataset cannot re-estimate.
_UNFREEZE_FROM: dict[str, str] = {
    "vgg16": "block5_conv1",  # last conv block
    "resnet50": "conv5_block1_1_conv",  # last residual stage
    "densenet121": "conv5_block8_0_bn",  # from block 8 of the last dense block's 16
    "efficientnet": "block6a_expand_conv",  # last two MBConv blocks
}

_L2_REG: float = 3e-5
_HEAD_DROPOUT: float = 0.5
_HEAD_UNITS: int = 256


# Public helpers


def get_normalisation_mode(name: str) -> str:
    return MODEL_REGISTRY[name]["normalisation"]


def get_input_shape(name: str) -> tuple[int, int, int]:
    h, w = config.TARGET_SIZE
    return (h, w, MODEL_REGISTRY[name]["channels"])


# Factory


def build_model(
    name: str, input_shape: tuple[int, int, int] | None = None, **builder_kwargs
) -> keras.Model:
    """Build and return an **uncompiled** model by name; ``train.py`` compiles it, so one
    architecture can be compiled differently for the frozen and fine-tuning stages."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")

    if input_shape is None:
        input_shape = get_input_shape(name)

    if name == "baseline":
        return _build_baseline(input_shape)
    if name == "scaled":
        return _build_scaled_cnn(input_shape)
    if name == "regularised":
        # builder_kwargs (l2_reg, head_dropout) reach this builder only, so the crop
        # path can regularise more lightly than the whole-image model.
        return _build_regularised_cnn(input_shape, **builder_kwargs)
    return _build_transfer(name, input_shape)


# From-scratch CNNs


def _se_block(x: tf.Tensor, filters: int, reduction: int = 16) -> tf.Tensor:
    """Squeeze-and-Excitation channel attention (Hu et al., 2018): pools each channel to
    a scalar, learns weights through a bottleneck MLP and rescales the map by them."""
    r = max(1, filters // reduction)
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(r, activation="relu", use_bias=False)(se)
    se = layers.Dense(filters, activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([x, se])


def _conv_block(
    x: tf.Tensor,
    filters: int,
    n_convs: int = 2,
    dropout_rate: float = 0.0,
    l2_reg: float = _L2_REG,
    use_se: bool = False,
) -> tf.Tensor:
    """``n_convs`` × [Conv → BN → ReLU] → [SE] → MaxPool → [Dropout]."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    for _ in range(n_convs):
        x = layers.Conv2D(
            filters, 3, padding="same", kernel_regularizer=reg, kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    if use_se:
        x = _se_block(x, filters)
    x = layers.MaxPooling2D(pool_size=2)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def _build_baseline(input_shape: tuple[int, int, int]) -> keras.Model:
    # Three conv blocks, sized for ~1,100 training images: 224→112→56→28 after
    # the max-pools, then GAP to 128 features. Dropout is in the head only; every layer
    # carries the default light L2.
    inputs = keras.Input(shape=input_shape, name="image")

    x = _conv_block(inputs, 32, dropout_rate=0.0)
    x = _conv_block(x, 64, dropout_rate=0.0)
    x = _conv_block(x, 128, dropout_rate=0.0)

    x = _classification_head(x, units=128, dropout_rate=0.3)

    return keras.Model(inputs=inputs, outputs=x, name="baseline_cnn")


def _build_scaled_cnn(input_shape: tuple[int, int, int]) -> keras.Model:
    # Four conv blocks with no regularisation at all, so the model can overfit and show
    # the capacity is there. 224→112→56→28→14 after the max-pools, then GAP to 256.
    inputs = keras.Input(shape=input_shape, name="image")

    x = _conv_block(inputs, 32, dropout_rate=0.0, l2_reg=0.0)
    x = _conv_block(x, 64, dropout_rate=0.0, l2_reg=0.0)
    x = _conv_block(x, 128, dropout_rate=0.0, l2_reg=0.0)
    x = _conv_block(x, 256, dropout_rate=0.0, l2_reg=0.0)

    x = _classification_head(x, units=256, dropout_rate=0.0, l2_reg=0.0)

    return keras.Model(inputs=inputs, outputs=x, name="scaled_cnn")


def _build_regularised_cnn(
    input_shape: tuple[int, int, int],
    l2_reg: float = 1e-4,
    head_dropout: float = 0.5,
    use_se: bool = True,
) -> keras.Model:
    # The capacity of the scaled CNN, regularised: L2 everywhere and dropout in the head
    # only. The defaults suit the whole-mammogram task; the crop path passes lighter ones.
    inputs = keras.Input(shape=input_shape, name="image")

    # use_se=False is the single-variable SE ablation (train.py --ablate se).
    x = _conv_block(inputs, 32, dropout_rate=0.0, l2_reg=l2_reg, use_se=use_se)
    x = _conv_block(x, 64, dropout_rate=0.0, l2_reg=l2_reg, use_se=use_se)
    x = _conv_block(x, 128, dropout_rate=0.0, l2_reg=l2_reg, use_se=use_se)
    x = _conv_block(x, 256, dropout_rate=0.0, l2_reg=l2_reg, use_se=use_se)

    # Mixed pooling concatenates GAP and GMP, so the dense head sees 512
    # features rather than 256.
    x = _classification_head(
        x, units=256, dropout_rate=head_dropout, l2_reg=l2_reg, mixed_pool=True
    )

    return keras.Model(inputs=inputs, outputs=x, name="regularised_cnn")


# Transfer-learning models


def _get_backbone(name: str, input_shape: tuple[int, int, int]) -> keras.Model:
    """Return the Keras Application backbone with ImageNet weights."""
    kw = dict(include_top=False, weights="imagenet", input_shape=input_shape)
    if name == "vgg16":
        return keras.applications.VGG16(**kw)
    if name == "resnet50":
        return keras.applications.ResNet50(**kw)
    if name == "densenet121":
        return keras.applications.DenseNet121(**kw)
    if name == "efficientnet":
        # EfficientNetB0 carries its own rescaling and expects raw [0, 255], so the
        # standardised tensors the pipeline delivers reach it as a near-constant image.
        mean = tf.constant(config.NORM_STATS["imagenet_mean"], tf.float32)
        std = tf.constant(config.NORM_STATS["imagenet_std"], tf.float32)
        inp = keras.Input(shape=input_shape, name="efficientnet_input")
        # Undone inside the backbone graph, so the data pipeline and every
        # backbone-splitting consumer stay uniform across architectures.
        raw = layers.Lambda(lambda t: (t * std + mean) * 255.0, name="destandardise_to_0_255")(inp)
        return keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=raw
        )
    raise ValueError(f"No backbone defined for '{name}'")


def _build_transfer(name: str, input_shape: tuple[int, int, int]) -> keras.Model:
    """Frozen backbone + fresh classification head (stage 1). ``training=False`` keeps
    the backbone's BatchNormalization on ImageNet statistics after unfreezing."""
    backbone = _get_backbone(name, input_shape)
    backbone.trainable = False  # freeze stage 1

    inputs = keras.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)
    x = _classification_head(x)

    return keras.Model(inputs=inputs, outputs=x, name=f"{name}_transfer")


def build_fusion(
    arch: str = "vgg16", input_shape: tuple[int, int, int] | None = None
) -> keras.Model:
    """Two-crop fusion model, returned **uncompiled** with the trunk frozen: one backbone
    instance is called on both crops, so the streams share every convolutional weight.
    """
    if arch not in _UNFREEZE_FROM:
        raise ValueError(f"No fusion trunk for '{arch}'; choose from {list(_UNFREEZE_FROM)}")
    if input_shape is None:
        input_shape = get_input_shape(arch)

    backbone = _get_backbone(arch, input_shape)
    backbone.trainable = False

    in_a = keras.Input(shape=input_shape, name="crop_a")
    in_b = keras.Input(shape=input_shape, name="crop_b")
    # training=False pins BatchNormalization to ImageNet statistics on both
    # calls, as in _build_transfer.
    feat_a = layers.GlobalAveragePooling2D(name="gap_a")(backbone(in_a, training=False))
    feat_b = layers.GlobalAveragePooling2D(name="gap_b")(backbone(in_b, training=False))
    fused = layers.Concatenate(name="fusion_concat")([feat_a, feat_b])
    # A 1x1 map makes the head's own pooling the identity, so the head is layer for
    # layer the one every single-crop transfer model uses.
    fused = layers.Reshape((1, 1, -1), name="fusion_as_map")(fused)
    out = _classification_head(fused)
    return keras.Model(inputs=[in_a, in_b], outputs=out, name=f"{arch}_fusion")


# Classification head, shared by all architectures


def _classification_head(
    x: tf.Tensor,
    units: int = _HEAD_UNITS,
    dropout_rate: float = _HEAD_DROPOUT,
    l2_reg: float = _L2_REG,
    mixed_pool: bool = False,
) -> tf.Tensor:
    """[GAP (+GMP)] → Dense(units, ReLU) → BN → Dropout → Dense(1, σ). With ``mixed_pool``
    GAP carries distributed texture and GMP the peak activation over a localised mass.
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    if mixed_pool:
        gap = layers.GlobalAveragePooling2D()(x)
        gmp = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([gap, gmp])
    else:
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal",
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return x


# The head layers carrying weights, the only ones trained in the frozen phase of
# in-domain transfer; everything else, SE-block Dense layers included, stays frozen.
_SCRATCH_HEAD_LAYERS: frozenset[str] = frozenset({"head_dense", "head_bn", "prediction"})


def freeze_scratch_backbone(model: keras.Model) -> keras.Model:
    """Stage 1 of in-domain transfer: only the head layers stay trainable, so the conv
    weights pretrained on ROI crops are held fixed. The head is not re-initialised.
    Mutates in place; re-compile for it to take effect."""
    for layer in model.layers:
        layer.trainable = layer.name in _SCRATCH_HEAD_LAYERS
    n_train = sum(1 for l in model.layers if l.trainable)
    print(
        f"[models] In-domain stage 1: {n_train}/{len(model.layers)} layers "
        f"trainable (head only, conv body frozen)"
    )
    return model


def unfreeze_scratch_backbone(model: keras.Model) -> keras.Model:
    """Unfreeze the whole network for stage 2 of in-domain transfer, so crop-pretrained
    features adapt to whole-mammogram statistics at a low learning rate. Mutates."""
    for layer in model.layers:
        layer.trainable = True
    print(
        f"[models] In-domain stage 2: all {len(model.layers)} layers trainable "
        f"(full fine-tune at low LR)"
    )
    return model


# Fine-tuning


def _find_backbone(model: keras.Model) -> keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def unfreeze_for_finetuning(model: keras.Model, backbone_name: str) -> keras.Model:
    """Unfreeze layers from ``_UNFREEZE_FROM[backbone_name]`` onward, BatchNormalization
    excepted, and return the same model mutated for ``train.py`` to re-compile."""
    if backbone_name not in _UNFREEZE_FROM:
        raise ValueError(f"No fine-tuning config for '{backbone_name}'")

    backbone = _find_backbone(model)
    if backbone is None:
        raise RuntimeError("Could not locate the backbone sub-model")

    target_name = _UNFREEZE_FROM[backbone_name]
    backbone.trainable = True  # unfreeze all first

    # Locate the target layer and freeze everything before it
    found = False
    for layer in backbone.layers:
        if layer.name == target_name:
            found = True
        if not found:
            layer.trainable = False
        else:
            # Keep BN frozen → ImageNet running stats preserved
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    # A Keras version change can rename the target layer, so fall back to a fraction.
    if not found:
        warnings.warn(
            f"Layer '{target_name}' not found in {backbone_name} backbone. "
            f"Falling back to unfreezing the last 20 % of layers."
        )
        n = len(backbone.layers)
        cutoff = int(n * 0.8)
        for i, layer in enumerate(backbone.layers):
            if i < cutoff or isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    total_count = len(backbone.layers)
    print(
        f"[models] Fine-tuning: {trainable_count}/{total_count} backbone "
        f"layers unfrozen (from '{target_name}', BN frozen)"
    )

    return model
