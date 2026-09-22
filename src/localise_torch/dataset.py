"""Patch sampling and encoder input preparation for the torch localiser members,
mirroring the Keras sampler: lesion-centred, background and anywhere draws, with
the epoch plan a pure function of (seed + epoch) so an epoch replays from its index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KIND_ANYWHERE, KIND_CENTRED, KIND_BACKGROUND = 0, 1, 2

# Encoder input

# The stores hold single-channel float16 in [0, 1]; the torchvision/SMP encoders expect
# the ImageNet convention — 3 channels, RGB, [0, 1], then mean/std normalised.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Horizontal flip only — mammograms have a fixed superior-inferior orientation —
# plus rotation within +-AUG_MAX_ROTATION_DEG.
AUG_MAX_ROTATION_DEG = 10.0


def to_encoder_input(x, normalise: bool = True):
    """(N,1,H,W) in [0,1] -> (N,3,H,W), ImageNet-normalised unless told otherwise:
    torchvision detection models apply mean/std themselves, so pass normalise=False.
    """
    import torch

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if not normalise:
        return x
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment_pair(
    img: np.ndarray,
    msk: np.ndarray,
    rng: np.random.Generator,
    max_rot_deg: float = AUG_MAX_ROTATION_DEG,
):
    """Flip and rotate an image and its mask as one stacked array, so one interpolation
    transforms both; returns the realised flip and angle, not the requested one.
    """
    from scipy import ndimage

    pair = np.stack([np.asarray(img, np.float32), np.asarray(msk, np.float32)], axis=-1)
    flipped = rng.random() < 0.5  # horizontal flip, p=0.5
    if flipped:
        pair = pair[:, ::-1]
    deg = float(rng.uniform(-max_rot_deg, max_rot_deg))
    if abs(deg) > 1e-6:
        # axes=(0,1) rotates in the spatial plane only, so image and mask stay
        # registered.
        pair = ndimage.rotate(
            pair, deg, axes=(0, 1), reshape=False, order=1, mode="constant", cval=0.0
        )
    image = np.ascontiguousarray(pair[..., 0])
    mask = np.ascontiguousarray((pair[..., 1] > 0.5).astype(np.float32))
    return image, mask, {"flipped": bool(flipped), "deg": deg}


class PatchPlan:
    """Per-epoch (row, y, x, kind) plans over an mmapped cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        *,
        patch: int = 512,
        per_image: int = 2,
        centred_frac: float = 0.5,
        background_frac: float = 0.0,
        jitter_frac: float = 0.25,
        seed: int = 28,
        breast_boxes: np.ndarray | None = None,
        limit: int | None = None,
    ):
        d = Path(cache_dir)
        self.imgs = np.load(d / f"{split}_images.npy", mmap_mode="r")
        self.msks = np.load(d / f"{split}_masks.npy", mmap_mode="r")
        self.meta = pd.read_csv(d / f"{split}_meta.csv")
        if limit is not None:  # smoke: first N images
            self.imgs, self.msks = self.imgs[:limit], self.msks[:limit]
            self.meta = self.meta.iloc[:limit]
        self.patch, self.per_image, self.seed = patch, per_image, seed
        self.centred_frac, self.background_frac = float(centred_frac), float(background_frac)
        self.jitter = int(round(jitter_frac * patch))
        self.h, self.w = self.imgs.shape[1], self.imgs.shape[2]
        if self.centred_frac + self.background_frac > 1.0 + 1e-9:
            raise ValueError("centred_frac + background_frac exceeds 1")
        # Foreground pixel coordinates per row, the pool the centred draws use.
        self.fg = [np.argwhere(np.asarray(self.msks[i]) > 0) for i in range(len(self.msks))]
        self.breast = breast_boxes
        self.stats: dict[int, dict] = {}

    def _box(self, i: int):
        if self.breast is None:
            return 0, self.h - self.patch, 0, self.w - self.patch
        y0, y1, x0, x1 = self.breast[i]
        return (
            max(0, y0),
            max(0, min(y1 - self.patch, self.h - self.patch)),
            max(0, x0),
            max(0, min(x1 - self.patch, self.w - self.patch)),
        )

    def _background(self, rng, i, tries: int = 12):
        ya, yb, xa, xb = self._box(i)
        m = np.asarray(self.msks[i])
        for _ in range(tries):
            y = int(rng.integers(ya, max(ya, yb) + 1))
            x = int(rng.integers(xa, max(xa, xb) + 1))
            if not m[y : y + self.patch, x : x + self.patch].any():
                return y, x, True
        return y, x, False

    def plan(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + epoch)
        rows, want, got = [], 0, 0
        for i in range(len(self.imgs)):
            for _ in range(self.per_image):
                u = rng.random()
                if u < self.centred_frac and len(self.fg[i]):
                    kind = KIND_CENTRED
                elif u < self.centred_frac + self.background_frac:
                    kind = KIND_BACKGROUND
                else:
                    kind = KIND_ANYWHERE
                if kind == KIND_CENTRED:
                    cy, cx = self.fg[i][rng.integers(len(self.fg[i]))]
                    y = int(
                        np.clip(
                            cy - self.patch // 2 + rng.integers(-self.jitter, self.jitter + 1),
                            0,
                            self.h - self.patch,
                        )
                    )
                    x = int(
                        np.clip(
                            cx - self.patch // 2 + rng.integers(-self.jitter, self.jitter + 1),
                            0,
                            self.w - self.patch,
                        )
                    )
                elif kind == KIND_BACKGROUND:
                    want += 1
                    y, x, ok = self._background(rng, i)
                    got += int(ok)
                    if not ok:
                        kind = KIND_ANYWHERE  # a failed draw is labelled as what it is
                else:
                    ya, yb, xa, xb = self._box(i)
                    y = int(rng.integers(ya, max(ya, yb) + 1))
                    x = int(rng.integers(xa, max(xa, xb) + 1))
                rows.append((i, y, x, kind))
        plan = np.asarray(rows, np.int64)[rng.permutation(len(rows))]
        self.stats[epoch] = {
            "lesion_centred_frac": float((plan[:, 3] == KIND_CENTRED).mean()),
            "background_frac_realised": float((plan[:, 3] == KIND_BACKGROUND).mean()),
            "background_frac_requested": self.background_frac,
            "background_draws_satisfied": (got / want) if want else 1.0,
        }
        return plan
