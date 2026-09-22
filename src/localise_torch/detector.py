"""Trains a Faster R-CNN localiser member, which proposes and scores boxes directly
instead of segmenting and reducing a mask to a box. Targets come from the merged
per-mammogram masks: one box per connected component, the smallest dropped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from .. import config
from .dataset import augment_pair, to_encoder_input
from ..geometry import Letterbox
from .evaluate import bundle


def component_boxes(mask: np.ndarray, min_px: int = 16) -> list[tuple[int, int, int, int]]:
    """One box per connected component of a merged mask whose box covers at least min_px,
    as (x0, y0, x1, y1): torchvision wants xyxy, the rest of the project (y0, y1, x0, x1).
    """
    lab, n = ndimage.label(np.asarray(mask) > 0.5)
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop - 1
        x0, x1 = sl[1].start, sl[1].stop - 1
        if (y1 - y0 + 1) * (x1 - x0 + 1) >= min_px:
            out.append((x0, y0, x1, y1))
    return out


class DetDataset(torch.utils.data.Dataset):
    """Whole cached images with one target box per mask component."""

    def __init__(
        self,
        cache_dir,
        split="train",
        limit=None,
        augment=True,
        seed: int = config.SEED,
        epoch: int = 0,
    ):
        d = Path(cache_dir)
        self.imgs = np.load(d / f"{split}_images.npy", mmap_mode="r")
        self.msks = np.load(d / f"{split}_masks.npy", mmap_mode="r")
        self.meta = pd.read_csv(d / f"{split}_meta.csv")
        self.n = len(self.imgs) if limit is None else min(limit, len(self.imgs))
        self.augment = augment
        # Per-epoch RNG, as in the U-Net members: one generator reused across
        # epochs would make an epoch irreproducible from its index.
        self.set_epoch(epoch)
        self.aug_stats = {"flips": 0, "degs": []}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.rng = np.random.default_rng(config.SEED + 100_000 + int(epoch))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = np.asarray(self.imgs[i], np.float32)
        msk = np.asarray(self.msks[i], np.float32)
        if self.augment:
            # Flip and rotation through the shared implementation, then boxes re-derived
            # from the transformed mask: a box is not closed under rotation.
            img, msk, info = augment_pair(img, msk, self.rng)
            self.aug_stats["flips"] += int(info["flipped"])
            self.aug_stats["degs"].append(info["deg"])
        boxes = component_boxes(msk)
        # 1ch -> 3ch via the shared function. normalise=False because the model's
        # own GeneralizedRCNNTransform applies ImageNet mean/std.
        t = to_encoder_input(
            torch.from_numpy(np.ascontiguousarray(img))[None, None], normalise=False
        )[0]
        if boxes:
            tb = torch.tensor(boxes, dtype=torch.float32)
            lbl = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            tb = torch.zeros((0, 4), dtype=torch.float32)
            lbl = torch.zeros((0,), dtype=torch.int64)
        return t, {"boxes": tb, "labels": lbl}


def collate(b):
    return [x[0] for x in b], [x[1] for x in b]


@torch.no_grad()
def predict(model, imgs, i, device="cpu"):
    """All boxes and scores for one image, in canvas (y0, y1, x0, x1) order."""
    model.eval()
    x = torch.from_numpy(np.asarray(imgs[i], np.float32))[None].repeat(3, 1, 1).to(device)
    out = model([x])[0]
    bx = out["boxes"].cpu().numpy()
    sc = out["scores"].cpu().numpy()
    return [(int(b[1]), int(b[3]), int(b[0]), int(b[2])) for b in bx], sc


def evaluate_split(model, cache_dir, split, *, device="cpu", limit=None):
    """Top-scoring box per image against each lesion's ground truth, scored with
    `localiser_bundle`'s `_iou` and `_gt_box`, the functions the U-Net members use.
    """
    lb = bundle()
    d = Path(cache_dir)
    imgs = np.load(d / f"{split}_images.npy", mmap_mode="r")
    msks = np.load(d / f"{split}_masks.npy", mmap_mode="r")
    meta = pd.read_csv(d / f"{split}_meta.csv")
    n = len(meta) if limit is None else min(limit, len(meta))
    rows, all_rows = [], []
    for i in range(n):
        r = meta.iloc[i]
        boxes, scores = predict(model, imgs, i, device)
        gt = lb._gt_box(msks[i])
        lbx = Letterbox(
            scale=float(r["scale"]),
            pad_y=int(r["pad_y"]),
            pad_x=int(r["pad_x"]),
            orig_h=int(r["orig_h"]),
            orig_w=int(r["orig_w"]),
            out_h=int(r["out_h"]),
            out_w=int(r["out_w"]),
        )
        for rank, (b, sc) in enumerate(zip(boxes, scores), 1):
            mb = lbx.inverse_box(b)
            all_rows.append(
                {
                    "split": split,
                    "lesion_id": r["lesion_id"],
                    "image_id": r.get("image_id"),
                    "rank": rank,
                    "score": float(sc),
                    "y0": mb[0],
                    "y1": mb[1],
                    "x0": mb[2],
                    "x1": mb[3],
                }
            )
        rec = {
            "split": split,
            "lesion_id": r["lesion_id"],
            "image_id": r.get("image_id"),
            "patient_id": r["patient_id"],
            "label": int(r["label"]),
            "orig_h": int(r["orig_h"]),
            "orig_w": int(r["orig_w"]),
            "detected": len(boxes) > 0,
            "n_boxes": len(boxes),
            "top_score": float(scores[0]) if len(scores) else np.nan,
        }
        if gt is None:
            rec.update(status="no_gt", box_iou=np.nan)
        elif not boxes:
            rec.update(status="miss", box_iou=0.0)
        else:
            top = boxes[int(np.argmax(scores))]
            g, p = lbx.inverse_box(gt), lbx.inverse_box(top)
            rec.update(
                status="ok",
                box_iou=lb._iou(gt, top),
                box_iou_mammogram=lb._iou(g, p),
                gt_y0=g[0],
                gt_y1=g[1],
                gt_x0=g[2],
                gt_x1=g[3],
                pred_y0=p[0],
                pred_y1=p[1],
                pred_x0=p[2],
                pred_x1=p[3],
            )
        rows.append(rec)
    t = pd.DataFrame(rows)
    # An empty DataFrame writes a zero-byte CSV that pandas cannot read back, so
    # give the box table its columns even when there are no rows.
    all_cols = ["split", "lesion_id", "image_id", "rank", "score", "y0", "y1", "x0", "x1"]
    allb = pd.DataFrame(all_rows, columns=all_cols) if not all_rows else pd.DataFrame(all_rows)
    scored = t[t["status"] != "no_gt"]["box_iou"].fillna(0.0)
    summary = {
        "val_box_iou": float(scored.mean()) if len(scored) else float("nan"),
        "detect_at_iou50": (float((scored >= 0.5).mean()) if len(scored) else float("nan")),
        "detection_rate": float(t["detected"].mean()),
        "n": int(n),
    }
    return t, allb, summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    # The default schedule is a constant rate; --warmup-epochs and --lr-min apply to
    # warmup_cosine only.
    ap.add_argument(
        "--schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
        help="warmup_cosine gives the detector the U-Net members' schedule shape: "
        "linear warm-up, then cosine down to --lr-min. Default constant holds "
        "the rate fixed.",
    )
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument(
        "--select-on",
        choices=["val_box_iou", "val_loss"],
        default="val_box_iou",
        help="val_box_iou is the gate's own metric, so selecting on it selects on the "
        "metric of record. val_loss is the Faster R-CNN multi-task loss on the "
        "validation split.",
    )
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument(
        "--score-only",
        default=None,
        help="score an existing checkpoint and write its box tables, without "
        "training. Use it to read the gate against a checkpoint the selection "
        "did not pick, e.g. last.pt.",
    )
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    args = ap.parse_args(argv)
    if args.parse_only:
        print(f"[parse-only] {__name__}: arguments valid; no data loaded")
        return

    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    inf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inf, 2)  # background + lesion
    model.to(dev)
    # Adam with no weight decay and the Keras eps, as the U-Net members use. A
    # detector has one parameter group: there is no encoder/decoder split here.
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0.0,
    )
    # The pretrained weights' own preprocessing lives in the model's transform.
    im, ist = model.transform.image_mean, model.transform.image_std
    assert [round(v, 3) for v in im] == [0.485, 0.456, 0.406] and [round(v, 3) for v in ist] == [
        0.229,
        0.224,
        0.225,
    ], f"torchvision transform is not ImageNet-normalising: mean {im} std {ist}"
    print(
        f"[detector] input: 3-ch replication (shared), ImageNet mean/std applied by "
        f"GeneralizedRCNNTransform mean={im} std={ist}",
        flush=True,
    )

    if args.score_only:
        ck = Path(args.score_only)
        if not ck.exists():
            raise SystemExit(f"--score-only: no such checkpoint {ck}")
        model.load_state_dict(torch.load(ck, map_location=dev))
        stem = ck.stem  # e.g. best or last
        print(f"[detector] score-only: {ck} -> localiser_boxes_{{split}}_{stem}.csv", flush=True)
        summary = {}
        for split in ("val", "train"):
            if not (Path(args.val_cache) / f"{split}_images.npy").exists():
                continue
            tb, allb, s = evaluate_split(
                model, args.val_cache, split, device=dev, limit=args.val_limit
            )
            tb.to_csv(out / f"localiser_boxes_{split}_{stem}.csv", index=False)
            allb.to_csv(out / f"detector_boxes_{split}_{stem}.csv", index=False)
            summary[split] = s
            print(
                f"[detector] {split} ({stem}): box IoU {s['val_box_iou']:.4f} "
                f"detect@0.5 {s['detect_at_iou50']:.4f}",
                flush=True,
            )
        (out / f"score_only_{stem}.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(ck),
                    "splits": summary,
                    "why": (
                        "boxes from a checkpoint the selection did not choose, so the gate "
                        "can be read against an epoch it did not pick"
                    ),
                },
                indent=2,
            )
        )
        print(f"-> {out}")
        return

    ds = DetDataset(args.cache, "train", args.limit, augment=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    # The U-Net members' schedule, imported rather than re-derived so the two
    # families cannot drift apart in shape.
    from .train import warmup_cosine_factor

    steps_per_epoch = max(len(dl), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    gstep = 0

    val_loss_ds = None
    if args.select_on == "val_loss":
        val_loss_ds = torch.utils.data.DataLoader(
            DetDataset(args.val_cache, "val", args.val_limit, augment=False),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )

    def _val_loss() -> float:
        """Faster R-CNN's multi-task loss over the validation split. The model returns
        losses only in train mode, in which its BatchNorm layers update running statistics."""
        model.train()
        tot_, n_ = 0.0, 0
        with torch.no_grad():
            for imgs_, tgts_ in val_loss_ds:
                imgs_ = [i.to(dev) for i in imgs_]
                tgts_ = [{k: v.to(dev) for k, v in x.items()} for x in tgts_]
                tot_ += float(sum(model(imgs_, tgts_).values()))
                n_ += 1
        return tot_ / max(n_, 1)

    hist, best, since = [], -1.0, 0
    for epoch in range(args.epochs):
        t0 = time.time()
        ds.set_epoch(epoch)
        ds.aug_stats = {"flips": 0, "degs": []}
        model.train()
        tot = 0.0
        for imgs, tgts in dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            opt.zero_grad(set_to_none=True)
            # fp32 throughout, as the U-Net members are: half precision diverges
            # to NaN here.
            loss = sum(model(imgs, tgts).values())
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            if args.schedule == "warmup_cosine":
                f = warmup_cosine_factor(gstep, args.lr, args.lr_min, warmup_steps, total_steps)
                for g in opt.param_groups:
                    g["lr"] = args.lr * f
            gstep += 1
        _, _, s = evaluate_split(model, args.val_cache, "val", device=dev, limit=args.val_limit)
        rec = {
            "epoch": epoch,
            "loss": tot / max(len(dl), 1),
            "wall_s": round(time.time() - t0, 1),
            "lr": opt.param_groups[0]["lr"],
            **s,
        }
        if val_loss_ds is not None:
            rec["val_loss"] = _val_loss()
        # Higher is better for box IoU, lower for loss; one comparison either way.
        score = -rec["val_loss"] if args.select_on == "val_loss" else s["val_box_iou"]
        if score > best:
            best, since = score, 0
            torch.save(model.state_dict(), out / "best.pt")
            rec["is_best"] = True
        else:
            since += 1
        hist.append(rec)
        print(
            f"[detector] epoch {epoch}: loss {rec['loss']:.4f} "
            f"lr {rec['lr']:.2e} val_box_iou {s['val_box_iou']:.4f} "
            f"detect@0.5 {s['detect_at_iou50']:.4f}"
            + (f" val_loss {rec['val_loss']:.4f}" if "val_loss" in rec else "")
            + f"{' *best*' if rec.get('is_best') else ''} ({rec['wall_s']:.0f} s)",
            flush=True,
        )
        torch.save(model.state_dict(), out / "last.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        if since >= args.patience:
            print(
                f"[detector] early stop: {args.patience} epochs without improving "
                f"the selection metric (best score {best:.4f})",
                flush=True,
            )
            break

    if (out / "best.pt").exists():
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    for split in ("val", "train"):
        if not (Path(args.val_cache) / f"{split}_images.npy").exists():
            continue
        t, allb, s = evaluate_split(model, args.val_cache, split, device=dev, limit=args.val_limit)
        t.to_csv(out / f"localiser_boxes_{split}.csv", index=False)
        allb.to_csv(out / f"detector_boxes_{split}.csv", index=False)
        print(
            f"[detector] {split}: box IoU {s['val_box_iou']:.4f} "
            f"detect@0.5 {s['detect_at_iou50']:.4f} {len(allb)} boxes -> {out}"
        )
    import platform, subprocess

    try:
        sha = (
            subprocess.run(["date"], capture_output=True, text=True)
            and subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        sha = "unknown"
    degs = ds.aug_stats["degs"]
    # The fields the recipe audit reads, less those a detector does not have, so it
    # is audited by the same tool as a U-Net member.
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run": out.name,
                "framework": f"pytorch-{torch.__version__.split('+')[0]}",
                "encoder": "resnet50-fpn (COCO)",
                "decoder": "Faster R-CNN heads (RPN + RoI)",
                "pretrained_weights": "COCO DEFAULT (fasterrcnn_resnet50_fpn_v2)",
                "input_preprocessing": "replicate 1->3 channels (shared to_encoder_input, "
                "normalise=False); ImageNet mean/std applied by "
                "torchvision GeneralizedRCNNTransform",
                "input_channels": 3,
                "cache": args.cache,
                "val_cache": args.val_cache,
                "canvas": list(ds.imgs.shape[1:3]),
                "input": list(ds.imgs.shape[1:3]),
                "steps_per_epoch": len(dl),
                "n_train_images": int(ds.n),
                "n_val_lesions": int(
                    len(np.load(Path(args.val_cache) / "val_images.npy", mmap_mode="r"))
                ),
                "train_store": "merged (one row per mammogram, union of lesion masks)",
                "mask_merging": "merged per mammogram (union of lesion masks)",
                "augmentation": "h-flip + rotation (run 4 parameters), boxes re-derived "
                "from the transformed mask",
                "aug_hflip_p": 0.5,
                "aug_rotation_deg": {"min": -10.0, "max": 10.0},
                "aug_rotation_fill": {"mode": "constant", "value": 0.0},
                "aug_interpolation": "bilinear (scipy order=1)",
                "aug_mask_handling": "image and mask stacked on the channel axis, one "
                "transform, mask re-thresholded at > 0.5; boxes from "
                "component_boxes on the result",
                "aug_flip_rate_last_epoch": ((ds.aug_stats["flips"] / len(degs)) if degs else None),
                "aug_deg_absmax_last_epoch": (float(max(abs(d) for d in degs)) if degs else None),
                "loss": "Faster R-CNN multi-task (RPN cls+reg, RoI cls+reg)",
                "optimizer": "Adam, no weight decay",
                "optimizer_params": {
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-7,
                    "weight_decay": 0.0,
                },
                "lr": args.lr,
                "lr_encoder": args.lr,
                "lr_decoder": args.lr,
                "schedule": (
                    "constant" if args.schedule == "constant" else "linear warm-up then cosine"
                ),
                "lr_min": (args.lr if args.schedule == "constant" else args.lr_min),
                "warmup_epochs": (0 if args.schedule == "constant" else args.warmup_epochs),
                "precision": "float32",
                "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "epochs_run": len(hist),
                "patience": args.patience,
                "early_stopping": (
                    "none binding — patience == epochs"
                    if args.patience >= args.epochs
                    else f"patience {args.patience}"
                ),
                "classes": 2,
                "seed": args.seed,
                "seed_handling": "torch.manual_seed + np.random.seed; dataset RNG is "
                "default_rng(SEED + 100000 + epoch)",
                "selection_metric": args.select_on,
                "hard_iou_threshold": config.LOC_MASK_THRESHOLD,
                "selection_min_px": config.LOC_MIN_COMPONENT_PX,
                "secondary_checkpoint": None,
                "best_val_box_iou": (
                    best
                    if args.select_on == "val_box_iou"
                    else max((h["val_box_iou"] for h in hist), default=float("nan"))
                ),
                "best_selection_score": best,
                "inference_mode": "whole-image (detector proposes boxes directly)",
                "flip_tta_at_gate": False,
                "box_rule": "highest-scoring detection per image",
                "box_threshold": config.LOC_MASK_THRESHOLD,
                "box_min_px": config.LOC_MIN_COMPONENT_PX,
                "encoder_bn": "frozen BatchNorm (torchvision FrozenBatchNorm2d in the "
                "detection backbone)",
                "git_sha": sha,
                "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                "env": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "python": platform.python_version(),
                },
            },
            indent=2,
        )
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
