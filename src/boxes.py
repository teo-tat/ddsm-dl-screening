"""Interchangeable box sources for the localisation-degradation ladder: rungs differ
only in the crop box, so the crop pipeline, augmentation and metrics stay identical.
Ground-truth boxes are read from localiser_boxes_{split}.csv."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from . import config

Box = tuple[int, int, int, int]
# oracle = the ground-truth box; unet = the localiser's box; cam = the Grad-CAM box; jitter =
# oracle displaced by a measured localiser error; shuffled = another lesion's; whole = none.
BoxSource = Literal["oracle", "unet", "cam", "jitter", "shuffled", "whole"]

ART = Path(__file__).resolve().parent.parent / "artifacts"


def _stable_rng(lesion_id: str, seed: int) -> np.random.Generator:
    """Per-lesion RNG keyed on the lesion id, so a lesion draws the same values every
    run and every epoch and the perturbation stays a fixed condition."""
    h = hashlib.sha256(f"{lesion_id}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _derange_by_view(lesion_ids: list[str], rng: np.random.Generator) -> dict[str, str]:
    """Donor map for the shuffled rung: each lesion borrows another lesion's box from
    its own laterality and view, which keeps the borrowed box over breast tissue."""

    def _view(lid: str) -> str:
        m = re.search(r"_(LEFT|RIGHT)_(CC|MLO)", lid)
        return m.group(0) if m else "OTHER"

    buckets: dict[str, list[str]] = {}
    for lid in lesion_ids:
        buckets.setdefault(_view(lid), []).append(lid)
    donor: dict[str, str] = {}
    for _, members in sorted(buckets.items()):
        if len(members) < 2:
            continue  # cannot derange a single element
        order = rng.permutation(len(members))
        for i in range(len(members)):
            donor[members[order[i]]] = members[order[(i + 1) % len(members)]]
    return donor


class BoxProvider:
    """Resolves lesion_id -> box in mammogram coordinates, or None."""

    def __init__(
        self,
        source: BoxSource,
        splits=("train", "val", "test"),
        seed: int = config.SEED,
        error_model: dict | None = None,
        cam_csv: str | Path | None = None,
        k: int = 1,
        mode: str = "single",
        boxes_dir: str | Path | None = None,
    ):
        """k / mode / boxes_dir apply to source="unet" only: mode serves the box
        table's single box or the matched top-k candidate."""
        self.source = source
        self.seed = seed
        self.k, self.mode = int(k), mode
        self.boxes_dir = str(boxes_dir) if boxes_dir is not None else None
        if mode not in ("single", "matched"):
            raise ValueError(f"mode must be single|matched, not {mode!r}")
        if mode != "single" and source != "unet":
            raise ValueError("mode matched is defined for source='unet' only")
        if self.k > config.LOC_TOPK or self.k < 1:
            raise ValueError(f"k must be in 1..{config.LOC_TOPK}")
        if source == "whole":
            self.gt = self.pred = {}
            return
        self.matched: dict[str, Box | None] = {}
        if mode != "single":
            bdir = Path(boxes_dir) if boxes_dir is not None else ART
            for sp in splits:
                path = bdir / f"localiser_boxes_k{config.LOC_TOPK}_{sp}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"{path} missing - run localise_eval --out-dir first")
                kdf = pd.read_csv(path)
                for r in kdf.itertuples():
                    self.matched[r.lesion_id] = (
                        (int(r.k3_pred_y0), int(r.k3_pred_y1), int(r.k3_pred_x0), int(r.k3_pred_x1))
                        if int(r.k3_rank) > 0
                        else None
                    )

        frames = []
        test_ids: set[str] = set()
        for sp in splits:
            path = ART / f"localiser_boxes_{sp}.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run `python -m src.localise_eval` first")
            frames.append(pd.read_csv(path))
            if sp == "test":
                test_ids = set(frames[-1]["lesion_id"])
        df = pd.concat(frames, ignore_index=True)

        self.gt = {
            r.lesion_id: (int(r.gt_y0), int(r.gt_y1), int(r.gt_x0), int(r.gt_x1))
            for r in df.itertuples()
            if not pd.isna(getattr(r, "gt_y0", np.nan))
        }

        self.breast = {
            r.lesion_id: (int(r.breast_y0), int(r.breast_y1), int(r.breast_x0), int(r.breast_x1))
            for r in df.itertuples()
            if not pd.isna(getattr(r, "breast_y0", np.nan))
        }
        if source == "unet":
            src_df = df
            if mode == "single" and boxes_dir is not None:
                cand = []
                for sp in splits:
                    path = Path(boxes_dir) / f"localiser_boxes_{sp}.csv"
                    if not path.exists():
                        raise FileNotFoundError(
                            f"{path} missing - the candidate has no {sp} boxes yet "
                            f"(localise_eval --out-dir, or localiser_bundle.py apply)"
                        )
                    cand.append(pd.read_csv(path))
                src_df = pd.concat(cand, ignore_index=True)
                missing = set(df["lesion_id"]) - set(src_df["lesion_id"])
                if missing:
                    raise RuntimeError(
                        f"candidate boxes under {boxes_dir} lack "
                        f"{len(missing)} lesions of the frozen tables"
                    )
            self.pred = {
                r.lesion_id: (int(r.pred_y0), int(r.pred_y1), int(r.pred_x0), int(r.pred_x1))
                for r in src_df.itertuples()
                if not pd.isna(getattr(r, "pred_y0", np.nan))
            }
        elif source == "cam":
            cam = pd.read_csv(cam_csv or ART / "cam_boxes.csv")
            self.pred = {
                r.lesion_id: (int(r.pred_y0), int(r.pred_y1), int(r.pred_x0), int(r.pred_x1))
                for r in cam.itertuples()
                if not pd.isna(getattr(r, "pred_y0", np.nan))
            }
        else:
            self.pred = {}

        if source == "jitter":
            if error_model is None:
                import json

                error_model = json.loads((ART / "localiser_error_model.json").read_text())
            self.err = error_model

        if source == "shuffled":
            # Train/val and test are deranged on separate random streams, so a donor
            # never crosses the split boundary and loading test changes no train/val donor.
            self.donor = {}
            self.donor.update(
                _derange_by_view(
                    sorted(lid for lid in self.gt if lid not in test_ids),
                    np.random.default_rng(seed),
                )
            )
            self.donor.update(
                _derange_by_view(
                    sorted(lid for lid in self.gt if lid in test_ids),
                    np.random.default_rng([seed, 1]),
                )
            )

    def get(self, lesion_id: str, image_shape: tuple[int, int]) -> Box | None:
        h, w = image_shape
        if self.source == "whole":
            return None
        if self.source == "oracle":
            return self.gt.get(lesion_id)
        if self.source == "unet" and self.mode == "matched":
            return self.matched.get(lesion_id)  # None == no matched candidate
        if self.source in ("unet", "cam"):
            return self.pred.get(lesion_id)  # None == localiser miss
        if self.source == "jitter":
            return self._jitter(lesion_id, h, w)
        if self.source == "shuffled":
            return self._shuffled(lesion_id, h, w)
        raise ValueError(f"Unknown box source: {self.source!r}")

    def _clamp(self, cy, cx, bh, bw, h, w) -> Box:
        """A (bh, bw) box centred on (cy, cx), shifted to sit inside the image."""
        bh, bw = max(1, int(round(bh))), max(1, int(round(bw)))
        y0 = int(round(cy - bh / 2))
        x0 = int(round(cx - bw / 2))
        y0 = max(0, min(y0, h - bh))
        x0 = max(0, min(x0, w - bw))
        return (y0, min(h - 1, y0 + bh - 1), x0, min(w - 1, x0 + bw - 1))

    def _jitter(self, lesion_id, h, w) -> Box | None:
        """Oracle box displaced by a draw from the localiser's measured error, so a gap to
        the localiser rung is failure on particular lesions rather than geometry alone."""
        gt = self.gt.get(lesion_id)
        if gt is None:
            return None
        rng = _stable_rng(lesion_id, self.seed)

        # Match the localiser's fallback rate, drawn at random rather than by its
        # own failure pattern.
        if rng.random() < float(self.err.get("fallback_rate", 0.0)):
            return None

        # Whole observed error tuples are replayed: the measured distribution is bimodal,
        # so a fitted Gaussian would put its mass where the data has none.
        samples = self.err["samples"]
        dy, dx, sh, sw = samples[int(rng.integers(len(samples)))]

        gh, gw = gt[1] - gt[0] + 1, gt[3] - gt[2] + 1
        cy = (gt[0] + gt[1]) / 2 + dy * gh
        cx = (gt[2] + gt[3]) / 2 + dx * gw
        return self._clamp(cy, cx, gh * max(0.1, sh), gw * max(0.1, sw), h, w)

    def _shuffled(self, lesion_id, h, w) -> Box | None:
        """Another lesion's box, taken as centre and size fractions of the donor's own
        breast box, since raw pixel coordinates do not transfer between mammograms."""
        donor_id = self.donor.get(lesion_id)
        donor = self.gt.get(donor_id) if donor_id else None
        db, rb = self.breast.get(donor_id), self.breast.get(lesion_id)
        if donor is None or db is None or rb is None:
            return None
        dbh, dbw = db[1] - db[0] + 1, db[3] - db[2] + 1
        rbh, rbw = rb[1] - rb[0] + 1, rb[3] - rb[2] + 1
        fy = ((donor[0] + donor[1]) / 2 - db[0]) / dbh
        fx = ((donor[2] + donor[3]) / 2 - db[2]) / dbw
        bh = (donor[1] - donor[0] + 1) / dbh * rbh
        bw = (donor[3] - donor[2] + 1) / dbw * rbw
        return self._clamp(rb[0] + fy * rbh, rb[2] + fx * rbw, bh, bw, h, w)


def provider_for(
    source: BoxSource, boxes_dir: str | Path | None = None, splits=("train", "val", "test"), **kw
) -> "BoxProvider":
    """Builds a provider; boxes_dir points "unet" and "jitter" at a candidate
    localiser's boxes and error model instead of artifacts/."""
    if boxes_dir is not None and source == "unet":
        return BoxProvider("unet", boxes_dir=boxes_dir, splits=splits, **kw)
    if boxes_dir is not None and source == "jitter":
        import json

        em = Path(boxes_dir) / "localiser_error_model.json"
        if not em.exists():
            raise FileNotFoundError(
                f"{em} missing - the jitter rung needs the candidate's validation "
                f"error model (localiser_bundle.py apply --split val)"
            )
        return BoxProvider("jitter", error_model=json.loads(em.read_text()), splits=splits, **kw)
    return BoxProvider(source, splits=splits, **kw)


LADDER = {
    1: ("oracle", 0.0, "Perfect localisation, tight"),
    2: ("oracle", None, "Perfect localisation + margin (the v4 condition)"),
    3: ("unet", None, "Supervised U-Net - the realisable system"),
    4: ("cam", None, "Weakly-supervised CAM - no mask annotation"),
    5: ("jitter", None, "Oracle + the U-Net's measured error"),
    6: ("shuffled", None, "Size- and position-plausible but wrong lesion"),
    7: ("whole", None, "No localisation (Pipeline A)"),
}
