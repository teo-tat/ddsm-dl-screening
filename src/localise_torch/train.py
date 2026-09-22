"""Trains an SMP U-Net localiser member. The training settings are ported from the
Keras recipe (the SMP decoder is not the Keras one), and maps are written in the
1024x576 canvas frame the bundle averages members in.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from .dataset import (
    AUG_MAX_ROTATION_DEG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PatchPlan,
    augment_pair,
    to_encoder_input,
)
from .evaluate import boxes_table, validate, write_maps

_ROOT = Path(__file__).resolve().parents[2]


def dice_bce(logits, target, eps: float = 1e-6):
    """Half binary cross-entropy plus half soft Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 0.5 * bce + 0.5 * (1 - (num / den).mean())


def warmup_cosine_factor(
    step: int, peak: float, floor: float, warmup_steps: int, total_steps: int
) -> float:
    """Learning-rate multiplier for LambdaLR: linear warm-up to `peak`, then cosine down
    to `floor`. LambdaLR multiplies base_lr, so base_lr is `peak` and this returns lr/peak.
    """
    if step < warmup_steps:
        return (step + 1.0) / float(max(1, warmup_steps))
    prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    prog = min(max(prog, 0.0), 1.0)
    return (floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))) / peak


def foreground_prior(msks, limit: int | None = None) -> float:
    """Mean foreground fraction of the training masks."""
    n = len(msks) if limit is None else min(limit, len(msks))
    tot = cnt = 0.0
    for i in range(n):
        m = np.asarray(msks[i])
        tot += float((m > 0.5).sum())
        cnt += float(m.size)
    return tot / max(cnt, 1.0)


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--cache", required=True, help="merged cache for training")
    ap.add_argument("--val-cache", default=None, help="per-lesion cache for validation")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--background-frac", type=float, default=0.0)
    ap.add_argument("--centred-frac", type=float, default=0.5)
    ap.add_argument("--jitter-frac", type=float, default=config.LOC_PATCH_JITTER_FRAC)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--lr-decoder", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-6, help="cosine floor")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N images")
    ap.add_argument("--val-limit", type=int, default=None, help="smoke: first N val images")
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="early stopping on val_hard_iou. Default is epochs, so it never binds: "
        "the LR schedule is part of the recipe and a truncated run is measured "
        "mid-schedule.",
    )
    ap.add_argument(
        "--no-augment",
        action="store_true",
        help="disable the h-flip and rotation. For ablations only; recipe.py check "
        "reports it as a deviation from the recipe.",
    )
    ap.add_argument(
        "--tiled",
        action="store_true",
        help="tiled inference at the training patch size. Off by default, because "
        "the Keras reference is scored whole-image at 1024x576 and the gate compares "
        "against that. Use it for members cached at 2048x1152, where whole-image "
        "inference needs four times the activation memory.",
    )
    ap.add_argument("--tile-overlap", type=int, default=64)
    ap.add_argument(
        "--output-prior",
        type=float,
        default=None,
        help="foreground prior for the output bias init. Default: measured from "
        "the training masks, as the Keras trainer does.",
    )
    ap.add_argument(
        "--write-boxes",
        action="store_true",
        help="with --write-maps, also write localiser_boxes_{split}.csv for each split",
    )
    ap.add_argument(
        "--write-maps",
        action="store_true",
        help="after training, write 1024x576 maps for localiser_bundle (needs --val-cache)",
    )
    ap.add_argument(
        "--run-name", default=None, help="member name in the bundle; default: the --out-dir name"
    )
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="parse the command line and exit 0 without loading data, so a flag that "
        "no longer exists fails in a second.",
    )
    args = ap.parse_args(argv)
    patience = args.epochs if args.patience is None else args.patience
    if args.parse_only:
        print(f"[parse-only] {__name__}: arguments valid; no data loaded")
        return

    import segmentation_models_pytorch as smp

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 3 input channels: the encoder carries ImageNet weights and `to_encoder_input`
    # hands it the distribution they were trained on.
    model = smp.Unet(
        encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=1
    ).to(dev)

    plan = PatchPlan(
        args.cache,
        "train",
        patch=args.patch,
        centred_frac=args.centred_frac,
        background_frac=args.background_frac,
        jitter_frac=args.jitter_frac,
        seed=args.seed,
        limit=args.limit,
    )

    # Output bias at the log-odds of the foreground prior, so the model starts at
    # the prior instead of at p=0.5.
    prior = args.output_prior if args.output_prior is not None else foreground_prior(plan.msks)
    prior_bias = float(np.log(prior / (1.0 - prior)))
    head = model.segmentation_head[0]
    torch.nn.init.constant_(head.bias, prior_bias)
    print(f"[{args.encoder}] output prior {prior:.7f} -> bias {prior_bias:.4f}", flush=True)

    # Two Adam instances, one per learning rate, rather than param groups: same update
    # semantics, separate state. eps 1e-7 is the Keras default; torch defaults to 1e-8.
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    adam_kw = dict(betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0)
    opt_enc = torch.optim.Adam(enc, lr=args.lr_encoder, **adam_kw)
    opt_dec = torch.optim.Adam(dec, lr=args.lr_decoder, **adam_kw)

    steps = int(np.ceil(len(plan.imgs) * plan.per_image / args.batch_size))
    total_steps = steps * args.epochs
    warmup_steps = steps * args.warmup_epochs
    sch_enc = torch.optim.lr_scheduler.LambdaLR(
        opt_enc,
        lambda s: warmup_cosine_factor(s, args.lr_encoder, args.lr_min, warmup_steps, total_steps),
    )
    sch_dec = torch.optim.lr_scheduler.LambdaLR(
        opt_dec,
        lambda s: warmup_cosine_factor(s, args.lr_decoder, args.lr_min, warmup_steps, total_steps),
    )

    val_imgs = val_msks = None
    if args.val_cache:
        vc = Path(args.val_cache)
        val_imgs = np.load(vc / "val_images.npy", mmap_mode="r")
        val_msks = np.load(vc / "val_masks.npy", mmap_mode="r")
        print(
            f"[{args.encoder}] validation: {len(val_imgs)} whole images from {vc}"
            f" ({'tiled' if args.tiled else 'whole-image'})",
            flush=True,
        )

    hist, best, best_loss, since_best = [], -1.0, float("inf"), 0
    for epoch in range(args.epochs):
        t0 = time.time()
        p = plan.plan(epoch)
        # One augmentation RNG per epoch, seeded as the plan is, so an epoch's
        # augmentation replays from its index.
        aug_rng = np.random.default_rng(args.seed + 100_000 + epoch)
        model.train()
        tot = 0.0
        flips, degs, chan_mean, chan_std = 0, [], None, None
        for s in range(steps):
            sel = p[s * args.batch_size : (s + 1) * args.batch_size]
            if not len(sel):
                break
            xs, ms = [], []
            for i, y, x0, _ in sel:
                im = np.asarray(plan.imgs[i, y : y + args.patch, x0 : x0 + args.patch], np.float32)
                mk = np.asarray(plan.msks[i, y : y + args.patch, x0 : x0 + args.patch], np.float32)
                if not args.no_augment:
                    im, mk, info = augment_pair(im, mk, aug_rng)
                    flips += int(info["flipped"])
                    degs.append(info["deg"])
                xs.append(im)
                ms.append(mk)
            xb = torch.from_numpy(np.stack(xs))[:, None].to(dev)
            mb = torch.from_numpy(np.stack(ms))[:, None].to(dev)
            xb = to_encoder_input(xb)
            if chan_mean is None:  # first batch of the epoch only
                chan_mean = xb.mean(dim=(0, 2, 3)).detach().cpu().tolist()
                chan_std = xb.std(dim=(0, 2, 3)).detach().cpu().tolist()
            opt_enc.zero_grad(set_to_none=True)
            opt_dec.zero_grad(set_to_none=True)
            # fp32 throughout: no autocast, no GradScaler, since half precision
            # diverges to NaN on this loss.
            loss = dice_bce(model(xb), mb)
            loss.backward()
            opt_enc.step()
            opt_dec.step()
            sch_enc.step()
            sch_dec.step()
            tot += float(loss.detach())
        n_aug = max(len(degs), 1)
        rec = {
            "epoch": epoch,
            "loss": tot / max(steps, 1),
            "lr_encoder": opt_enc.param_groups[0]["lr"],
            "lr_decoder": opt_dec.param_groups[0]["lr"],
            "aug_flip_rate": flips / n_aug if degs else None,
            "aug_deg_mean": float(np.mean(degs)) if degs else None,
            "aug_deg_absmax": float(np.max(np.abs(degs))) if degs else None,
            "input_mean_per_channel": chan_mean,
            "input_std_per_channel": chan_std,
            "wall_s": round(time.time() - t0, 1),
            **plan.stats[epoch],
        }
        if val_imgs is not None:
            rec.update(
                validate(
                    model,
                    val_imgs,
                    val_msks,
                    device=dev,
                    batch=args.batch_size,
                    limit=args.val_limit,
                    tiled=args.tiled,
                    overlap=args.tile_overlap,
                )
            )
            if rec["val_hard_iou"] > best:
                best, since_best = rec["val_hard_iou"], 0
                torch.save(model.state_dict(), out / "best.pt")
                rec["is_best"] = True
            else:
                since_best += 1
            # Secondary checkpoint on val_loss, kept for the record; selection is
            # on val_hard_iou alone.
            if rec.get("val_loss", float("inf")) < best_loss:
                best_loss = rec["val_loss"]
                torch.save(model.state_dict(), out / "best_valloss.pt")
                rec["is_best_valloss"] = True
        hist.append(rec)
        print(
            f"[{args.encoder}] epoch {epoch}: loss {rec['loss']:.4f} "
            f"lr_dec {rec['lr_decoder']:.2e} bg {rec['background_frac_realised']:.3f}"
            + (f" flip {rec['aug_flip_rate']:.3f}" if rec["aug_flip_rate"] is not None else "")
            + (
                f" val_hard_iou {rec['val_hard_iou']:.4f} box_iou "
                f"{rec['val_box_iou_inclusive']:.4f} val_loss {rec['val_loss']:.4f}"
                f"{' *best*' if rec.get('is_best') else ''}"
                if val_imgs is not None
                else ""
            )
            + f" ({rec['wall_s']:.0f} s)",
            flush=True,
        )
        torch.save(model.state_dict(), out / "last.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        if val_imgs is not None and since_best >= patience:
            print(
                f"[{args.encoder}] early stop: {patience} epochs without improving "
                f"val_hard_iou (best {best:.4f})",
                flush=True,
            )
            break

    best_epoch = (
        max(range(len(hist)), key=lambda i: hist[i].get("val_hard_iou", -1))
        if val_imgs is not None
        else None
    )
    # Every field the recipe audit reads, so the run describes itself.
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run": args.run_name or out.name,
                "framework": f"pytorch-{torch.__version__.split('+')[0]}",
                "encoder": args.encoder,
                "decoder": "segmentation_models_pytorch Unet decoder",
                "pretrained_weights": "imagenet (SMP encoder weights)",
                "input_preprocessing": "replicate 1->3 channels, ImageNet mean/std (RGB)",
                "input_channels": 3,
                "imagenet_mean": list(IMAGENET_MEAN),
                "imagenet_std": list(IMAGENET_STD),
                "cache": args.cache,
                "val_cache": args.val_cache,
                "canvas": list(plan.imgs.shape[1:3]),
                "train_store": "merged (one row per mammogram, union of lesion masks)",
                "mask_merging": "merged per mammogram (union of lesion masks)",
                "n_train_images": int(len(plan.imgs)),
                "n_val_lesions": int(len(val_imgs)) if val_imgs is not None else None,
                "patch_size": args.patch,
                "patches_per_image": plan.per_image,
                "lesion_centred_frac": args.centred_frac,
                "patch_jitter_frac": args.jitter_frac,
                "background_frac": args.background_frac,
                "background_frac_requested": args.background_frac,
                "augmentation": (
                    "none" if args.no_augment else "h-flip + rotation (run 4 parameters)"
                ),
                "aug_hflip_p": None if args.no_augment else 0.5,
                "aug_rotation_deg": (
                    None
                    if args.no_augment
                    else {"min": -AUG_MAX_ROTATION_DEG, "max": AUG_MAX_ROTATION_DEG}
                ),
                "aug_rotation_fill": (
                    None if args.no_augment else {"mode": "constant", "value": 0.0}
                ),
                "aug_interpolation": (None if args.no_augment else "bilinear (scipy order=1)"),
                "aug_mask_handling": (
                    None
                    if args.no_augment
                    else "image and mask stacked on the channel axis, one transform, "
                    "mask re-thresholded at > 0.5 after rotation"
                ),
                "loss": "0.5 * BCE + 0.5 * soft-Dice",
                "dice_bce_weights": [0.5, 0.5],
                "optimizer": "Adam (two instances: encoder, decoder), no weight decay",
                "optimizer_params": {
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-7,
                    "weight_decay": 0.0,
                },
                "lr_encoder": args.lr_encoder,
                "lr_decoder": args.lr_decoder,
                "lr_min": args.lr_min,
                "warmup_epochs": args.warmup_epochs,
                "schedule": "linear warm-up then cosine",
                "schedule_formula": "warm = peak*(step+1)/max(1,warmup_steps); "
                "prog = clip((step-warmup_steps)/max(1,total_steps-warmup_steps),0,1); "
                "cos = floor + 0.5*(peak-floor)*(1+cos(pi*prog)); "
                "lr = warm if step < warmup_steps else cos",
                "two_rate": "encoder and decoder on separate Adam instances of the same "
                "schedule shape",
                "precision": "float32",
                "batch_size": args.batch_size,
                "steps_per_epoch": steps,
                "epochs_requested": args.epochs,
                "epochs_run": len(hist),
                "early_stopping": (
                    "none binding — patience == epochs"
                    if patience >= args.epochs
                    else f"patience {patience}"
                ),
                "patience": patience,
                "output_prior": prior,
                "output_prior_bias": prior_bias,
                "output_bias_formula": "log(output_prior / (1 - output_prior))",
                "selection_metric": "val_hard_iou",
                "hard_iou_threshold": config.LOC_MASK_THRESHOLD,
                "selection_min_px": config.LOC_MIN_COMPONENT_PX,
                "secondary_checkpoint": "val_loss",
                "val_loss_note": "computed from WHOLE-IMAGE probabilities on the validation "
                "split; run 4's Keras val_loss came from the training graph on "
                "PATCHES. The two are not comparable across families. It drives "
                "only the secondary checkpoint and never selection.",
                "best_epoch": best_epoch,
                "best_val_hard_iou": best,
                "inference_mode": (
                    f"tiled {args.patch}-px, overlap {args.tile_overlap}"
                    if args.tiled
                    else f"whole-image at {plan.imgs.shape[1]}x{plan.imgs.shape[2]} (no tiling)"
                ),
                "flip_tta_at_gate": False,
                "box_rule": "largest surviving connected component",
                "box_threshold": config.LOC_MASK_THRESHOLD,
                "box_min_px": config.LOC_MIN_COMPONENT_PX,
                "encoder_bn": "trainable, running stats updating (inherent to the encoder)",
                "seed": args.seed,
                "seed_handling": "torch.manual_seed + np.random.seed; the patch plan is a pure "
                "function of (seed + epoch); augmentation RNG is "
                "default_rng(seed + 100000 + epoch)",
                "git_sha": _git_sha(),
                "device": (
                    torch.cuda.get_device_name(0)
                    if dev == "cuda"
                    else platform.processor() or "cpu"
                ),
                "env": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "smp": getattr(smp, "__version__", "unknown"),
                    "python": platform.python_version(),
                },
            },
            indent=2,
        )
    )

    if args.write_maps and args.val_cache:
        if (out / "best.pt").exists():  # the selected epoch, not the last
            model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        run = args.run_name or out.name
        for split in ("val", "train"):
            if (Path(args.val_cache) / f"{split}_images.npy").exists():
                r = write_maps(
                    model,
                    args.val_cache,
                    split,
                    run,
                    out / "maps",
                    device=dev,
                    batch=args.batch_size,
                    limit=args.val_limit,
                    tiled=args.tiled,
                    overlap=args.tile_overlap,
                )
                print(f"[{args.encoder}] maps {split}: {r['n']} rows -> {r['plain']}")
                if args.write_boxes:
                    t = boxes_table(
                        model,
                        args.val_cache,
                        split,
                        device=dev,
                        batch=args.batch_size,
                        limit=args.val_limit,
                        tiled=args.tiled,
                        overlap=args.tile_overlap,
                    )
                    bp = out / f"localiser_boxes_{split}.csv"
                    t.to_csv(bp, index=False)
                    print(f"[{args.encoder}] boxes {split}: {len(t)} rows -> {bp}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
