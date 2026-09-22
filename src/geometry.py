"""Tiling, map downsampling, and box conversion between the mammogram and the letterboxed
canvas. numpy only, so the TensorFlow and the PyTorch localisers share one definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import config


def tile_positions(size: int, patch: int, overlap: int) -> list[int]:
    """Start offsets covering ``size`` with ``patch``-wide tiles overlapping by ``overlap``.
    The last tile is clamped to the edge, not padded, so no zeros are read as tissue."""
    if patch >= size:
        return [0]
    stride = patch - overlap
    pos = list(range(0, size - patch + 1, stride))
    if pos[-1] != size - patch:
        pos.append(size - patch)
    return pos


def area_downsample(m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Mean-pool a map to (out_h, out_w) so members trained at different canvas sizes
    average in one frame; pooling keeps a narrow ridge as a lower-probability band."""
    h, w = m.shape[:2]
    if (h, w) == (out_h, out_w):
        return np.asarray(m, np.float32)
    if h % out_h or w % out_w:
        raise ValueError(f"area_downsample needs integer factors: {(h, w)} -> {(out_h, out_w)}")
    fy, fx = h // out_h, w // out_w
    return np.asarray(m, np.float32).reshape(out_h, fy, out_w, fx).mean(axis=(1, 3))


@dataclass(frozen=True)
class Letterbox:
    """The forward transform applied to one mammogram, and its inverse."""

    scale: float
    pad_y: int
    pad_x: int
    orig_h: int
    orig_w: int
    out_h: int
    out_w: int

    @classmethod
    def for_shape(
        cls, h: int, w: int, out_h: int = config.LOC_INPUT_H, out_w: int = config.LOC_INPUT_W
    ) -> "Letterbox":
        scale = min(out_h / h, out_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return cls(
            scale=scale,
            pad_y=(out_h - new_h) // 2,
            pad_x=(out_w - new_w) // 2,
            orig_h=int(h),
            orig_w=int(w),
            out_h=out_h,
            out_w=out_w,
        )

    @property
    def content_h(self) -> int:
        return max(1, int(round(self.orig_h * self.scale)))

    @property
    def content_w(self) -> int:
        return max(1, int(round(self.orig_w * self.scale)))

    def forward_box(self, box: Box) -> Box:
        """Map a mammogram box onto the canvas, clamped to the content region so the
        far edge stays inside the canvas and the box stays out of the padding."""
        cy0, cy1 = self.pad_y, self.pad_y + self.content_h - 1
        cx0, cx1 = self.pad_x, self.pad_x + self.content_w - 1

        def _c(v: float, lo: int, hi: int) -> int:
            return max(lo, min(int(round(v)), hi))

        y0, y1, x0, x1 = box
        return (
            _c(y0 * self.scale + self.pad_y, cy0, cy1),
            _c(y1 * self.scale + self.pad_y, cy0, cy1),
            _c(x0 * self.scale + self.pad_x, cx0, cx1),
            _c(x1 * self.scale + self.pad_x, cx0, cx1),
        )

    def inverse_box(self, box: Box) -> Box:
        """Map a canvas box back to mammogram coordinates, clamped to the image, so a
        box overlapping the padding cannot give an out-of-bounds crop."""
        y0, y1, x0, x1 = box

        def _c(v: float, hi: int) -> int:
            return max(0, min(int(round(v)), hi - 1))

        return (
            _c((y0 - self.pad_y) / self.scale, self.orig_h),
            _c((y1 - self.pad_y) / self.scale, self.orig_h),
            _c((x0 - self.pad_x) / self.scale, self.orig_w),
            _c((x1 - self.pad_x) / self.scale, self.orig_w),
        )

    def as_dict(self) -> dict:
        return asdict(self)
