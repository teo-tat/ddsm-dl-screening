#!/usr/bin/env python
"""recipe.py — run 4's recipe, and what a candidate changes against it.

Run 4 is the reference in gate (a), so any setting that differs and is not the
declared variable is a confound the gate cannot separate.

  build write the recipe file from run 4's manifest and the settings recorded here from the code
  check the config diff for one candidate: declared, inherent, or a confound
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# build — the recipe file
MANIFEST = ROOT / "outputs" / "weights" / "localiser_run4_vgg16enc_patch" / "manifest.json"

OUT = ROOT / "docs" / "run4_recipe.json"


def build_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.parse_args(argv)
    if not MANIFEST.exists():
        raise SystemExit(f"run 4 manifest not found: {MANIFEST}")
    m = json.loads(MANIFEST.read_text())

    def rec(value, source, recorded):
        return {"value": value, "source": source, "recorded_in_manifest": recorded}

    def mk(key, source_note=""):
        return rec(m[key], f"manifest:{key}" + (f" ({source_note})" if source_note else ""), True)

    R = {
        # identity
        "run": mk("run"),
        "device": mk("device"),
        "framework": rec(
            "keras-tf2.17", "src/localise.py (Keras Model); tf pin in requirements.txt", False
        ),
        "encoder": mk("encoder"),
        "decoder": rec(
            "TernausNet-style U-Net decoder, transposed-conv upsampling, "
            "base_filters 32, depth 4",
            "src/localise.py:210-213 build_unet_pretrained; config.py:306-307",
            False,
        ),
        "params": mk("params"),
        # pretrained weights and their preprocessing
        "pretrained_weights": rec(
            "ImageNet VGG16 (Keras applications)",
            "src/localise.py:252 build_unet_pretrained docstring",
            False,
        ),
        "input_preprocessing": rec(
            "in-graph: x * 255.0 -> tf.repeat to 3 channels -> RGB->BGR ([..., ::-1]) "
            "-> subtract Caffe channel means (103.939, 116.779, 123.68)",
            "src/localise.py:242 _CAFFE_BGR_MEAN; :281-284 caffe_preprocess Lambda",
            False,
        ),
        "input_channels": rec(3, "src/localise.py:283 tf.repeat(t * 255.0, 3, axis=-1)", False),
        "store_intensity_range": rec(
            "[0,1] float16 in the tensor store; scaling happens in-graph",
            "src/localise_eval.py:36-41 _load (scale=1.0 for float16)",
            False,
        ),
        # data
        "cache_version": rec(
            "v1 (outputs/localiser_tensors)",
            "run 4 predates cache v2; localiser_bundle.MEMBER_TENSORS['run4']",
            False,
        ),
        "cache_percentiles": rec(
            {"low": 1.0, "high": 99.0},
            "src/config.py:85-86 PERCENTILE_LOW / PERCENTILE_HIGH",
            False,
        ),
        "canvas": mk("input", "letterbox canvas h,w"),
        "train_store": mk("train_store"),
        "mask_merging": rec(
            "merged per mammogram (union of lesion masks), run 3 onward",
            "manifest:train_store",
            True,
        ),
        "n_train_images": mk("n_train_images"),
        "n_val_lesions": mk("n_val_lesions"),
        # patch sampler
        "patch_size": mk("patch_size"),
        "patches_per_image": mk("patches_per_image"),
        "lesion_centred_frac": mk("lesion_centred_frac"),
        "patch_jitter_frac": mk("patch_jitter_frac"),
        "background_frac": rec(
            0.0, "run 4 had no background draw; the option arrived with L5 (work plan §5)", False
        ),
        # augmentation
        "augmentation": mk("augmentation"),
        "aug_hflip_p": rec(0.5, "src/localise.py:516 tf.image.random_flip_left_right", False),
        "aug_rotation_deg": rec(
            {"min": -10.0, "max": 10.0},
            "src/localise.py:503-504 RandomRotation(factor=AUG_MAX_ROTATION_DEG/360) "
            "= +-10 deg; config.py:172",
            False,
        ),
        "aug_rotation_fill": rec({"mode": "constant", "value": 0.0}, "src/localise.py:505", False),
        "aug_interpolation": rec(
            "bilinear (Keras RandomRotation default)", "src/localise.py:503", False
        ),
        "aug_mask_handling": rec(
            "image and mask concatenated on the channel axis, one transform, "
            "mask re-thresholded at > 0.5 after rotation",
            "src/localise.py:516-519 _augment_pair",
            False,
        ),
        # objective
        "loss": rec(
            "0.5 * BCE + 0.5 * soft-Dice",
            "src/localise.py:486 bce_dice_loss; train_localiser_l4.py:482",
            False,
        ),
        "dice_bce_weights": mk("dice_bce_weights"),
        # optimisation
        "optimizer": rec(
            "Adam (Keras), no weight decay",
            "src/train_localiser_l4.py:482,484 tf.keras.optimizers.Adam",
            False,
        ),
        "optimizer_params": rec(
            {"beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7, "weight_decay": 0.0},
            "Keras Adam defaults (NB torch Adam eps default is 1e-8)",
            False,
        ),
        "lr_encoder": mk("lr_encoder"),
        "lr_decoder": mk("lr_decoder"),
        "lr_min": mk("lr_min", "cosine floor"),
        "warmup_epochs": mk("warmup_epochs"),
        "schedule": mk("schedule"),
        "schedule_formula": rec(
            "warm = peak*(step+1)/max(1,warmup_steps); "
            "prog = clip((step-warmup_steps)/max(1,total_steps-warmup_steps),0,1); "
            "cos = floor + 0.5*(peak-floor)*(1+cos(pi*prog)); "
            "lr = warm if step < warmup_steps else cos",
            "src/localise.py:384-399 WarmupCosine.__call__",
            False,
        ),
        "two_rate": rec(
            "encoder and decoder on separate optimisers of the same schedule shape",
            "src/localise.py:404-443 TwoRateModel; train_localiser_l4.py:484",
            False,
        ),
        "precision": mk("precision"),
        "batch_size": mk("batch_size"),
        "steps_per_epoch": mk("steps_per_epoch"),
        "epochs_requested": mk("epochs_requested"),
        "epochs_run": mk("epochs_run"),
        "early_stopping": rec(
            "none binding — ran all 100 epochs",
            "manifest:epochs_run == epochs_requested == 100",
            False,
        ),
        "output_prior": mk("output_prior"),
        "output_prior_bias": mk("output_prior_bias", "log(p/(1-p))"),
        "output_bias_formula": rec(
            "log(output_prior / (1 - output_prior))",
            "src/train_localiser_l4.py:513; localise.py:219-221",
            False,
        ),
        # selection
        "selection_metric": mk("selection_metric"),
        "hard_iou_threshold": mk("hard_iou_threshold"),
        "selection_min_px": rec(64, "src/config.py:319 LOC_MIN_COMPONENT_PX", False),
        "secondary_checkpoint": mk("secondary_checkpoint"),
        "best_epoch": mk("best_epoch"),
        # inference and boxes
        "inference_mode": rec(
            "whole-image at 1024x576 (no tiling)",
            "notebooks/scripts/localise_eval_run4.sh:14 passes no --tiled; "
            "tiling arrived with cache v3",
            False,
        ),
        "flip_tta_at_gate": rec(False, "localiser_bundle.REFERENCE_RULE tta=False", False),
        "box_rule": rec(
            "largest surviving connected component",
            "src/localise.py:58-90 mask_to_box, keep_largest="
            "config.LOC_KEEP_LARGEST_COMPONENT (True, config.py:318)",
            False,
        ),
        "box_threshold": rec(0.10, "src/config.py:317 LOC_MASK_THRESHOLD", False),
        "box_min_px": rec(64, "src/config.py:319 LOC_MIN_COMPONENT_PX", False),
        # reproducibility
        "seed": mk("seed"),
        "seed_handling": rec(
            "config.set_seeds(seed): tf.random.set_seed + np.random.seed; "
            "the patch plan is a pure function of (seed + epoch)",
            "src/localise.py:541 config.set_seeds; config.py:7 SEED=28",
            False,
        ),
        # the reference number every candidate is gated against
        "val_box_iou_inclusive_mean": rec(
            0.4190, "outputs/results/localiser_run4_eval/localiser_metrics.json (val)", False
        ),
        "val_detect_at_iou50": rec(
            0.5391, "outputs/results/localiser_run4_eval/localiser_metrics.json (val)", False
        ),
    }

    sha = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    doc = {
        "_what": "Run 4, the localiser baseline every candidate's gate (a) is measured against. "
        "Generated by notebooks/scripts/build_run4_recipe.py from run 4's manifest and "
        "the Keras code paths that trained it — never from prose summaries, one of which "
        "was wrong about batch size.",
        "_generated_at_repo_sha": sha,
        "_manifest": str(MANIFEST.relative_to(ROOT)),
        "recipe": R,
    }
    inferred = sorted(k for k, v in R.items() if not v["recorded_in_manifest"])
    recorded = sorted(k for k, v in R.items() if v["recorded_in_manifest"])
    doc["_recorded_in_run4_manifest"] = recorded
    doc["_inferred_from_code"] = inferred
    OUT.write_text(json.dumps(doc, indent=2))

    print(
        f"fields total {len(R)} | from manifest {len(recorded)} "
        f"| inferred from code {len(inferred)}"
    )
    print("\n--- RECORDED IN RUN 4's MANIFEST ---")
    for k in recorded:
        print(f"  {k:28} {str(R[k]['value'])[:48]:48} <- {R[k]['source']}")
    print("\n--- INFERRED FROM CODE (run 4 did not record these) ---")
    for k in inferred:
        print(f"  {k:28} {str(R[k]['value'])[:48]:48} <- {R[k]['source']}")
    print(f"\n-> {OUT}")
    return 0


# check — the config diff for one candidate
RUN4 = ROOT / "refs" / "run4_manifest.json"

RUN4_CODE = {
    "framework": ("keras-tf2.17", "src/localise.py"),
    "input_scaling": (
        "x255 -> RGB->BGR -> Caffe channel means subtracted",
        "localise.py:283 caffe_preprocess",
    ),
    "input_channels": (3, "localise.py:283 tf.repeat(..., 3)"),
    "optimizer": ("Adam", "train_localiser_l4.py:482"),
    "weight_decay": (0.0, "Adam has none"),
    "encoder_bn": ("none (VGG16 has no BatchNorm)", "architecture"),
    "inference": ("whole-image at 1024x576", "localise_eval, run 4 was not tiled"),
    "flip_tta_at_gate": (False, "REFERENCE_RULE tta=False"),
    "box_rule": ("largest component", "localise.mask_to_box keep_largest"),
    "box_threshold": (0.10, "config.LOC_MASK_THRESHOLD"),
    "box_min_px": (64, "config.LOC_MIN_COMPONENT_PX"),
    "decoder": (
        "TernausNet-style U-Net decoder, transposed-conv upsampling, base_filters 32, depth 4",
        "src/localise.py:210-213",
    ),
    "pretrained_weights": ("ImageNet VGG16 (Keras applications)", "src/localise.py:252"),
}

FIELDS = [
    ("framework", None),
    ("encoder", "encoder"),
    # Checked, not assumed: a port's decoder differs from run 4's, and the check has to
    # be what says so.
    ("decoder", None),
    ("pretrained_weights", None),
    ("cache", None),  # run 4 is v1 by construction
    ("augmentation", "augmentation"),
    ("loss", None),
    ("dice_bce_weights", "dice_bce_weights"),
    ("optimizer", None),
    ("weight_decay", None),
    ("schedule", "schedule"),
    ("lr_encoder", "lr_encoder"),
    ("lr_decoder", "lr_decoder"),
    ("lr_min", "lr_min"),
    ("warmup_epochs", "warmup_epochs"),
    ("patience", None),
    ("batch_size", "batch_size"),
    ("steps_per_epoch", "steps_per_epoch"),
    ("epochs", "epochs_requested"),
    ("patch_size", "patch_size"),
    ("input", "input"),
    ("patches_per_image", "patches_per_image"),
    ("lesion_centred_frac", "lesion_centred_frac"),
    ("patch_jitter_frac", "patch_jitter_frac"),
    ("background_frac", None),
    ("train_store", "train_store"),
    ("n_train_images", "n_train_images"),
    ("n_val_lesions", "n_val_lesions"),
    ("input_scaling", None),
    ("input_channels", None),
    ("precision", "precision"),
    ("output_prior_bias", "output_prior_bias"),
    ("encoder_bn", None),
    ("seed", "seed"),
    ("selection_metric", "selection_metric"),
    ("hard_iou_threshold", "hard_iou_threshold"),
    ("inference", None),
    ("flip_tta_at_gate", None),
    ("box_rule", None),
    ("box_threshold", None),
    ("box_min_px", None),
]

CAND_KEYS = {
    "input": ("canvas", "input"),
    "input_scaling": ("input_preprocessing", "input_scaling"),
    "epochs": ("epochs_requested", "epochs"),
    "inference": ("inference_mode", "inference"),
    "weight_decay": (("optimizer_params", "weight_decay"), "weight_decay"),
    # NOT init_weights: that is a warm-start path, None for a from-ImageNet run, and
    # reading it here would call a correct run a deviation.
    "pretrained_weights": ("pretrained_weights",),
}


def cand_get(cand: dict, label: str, *, none_is_absent: bool = False):
    """The candidate's value for a label, trying every spelling in CAND_KEYS."""
    for key in CAND_KEYS.get(label, (label,)):
        if isinstance(key, tuple):
            cur, ok = cand, True
            for k in key:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if ok and not (none_is_absent and cur is None):
                return cur, True
        elif key in cand and not (none_is_absent and cand[key] is None):
            return cand[key], True
    return None, False


def _ws(s) -> str:
    """Remove whitespace and fold case: prose differences are not confounds."""
    return re.sub(r"\s+", "", str(s).strip().lower())


def _canon_aug(v) -> str:
    s = _ws(v)
    if s in ("none", "nonetype", "null"):
        return "none"
    return "hflip+rotation" if "h-flip" in str(v).lower() or "hflip" in s else s


def _canon_opt(v) -> str:
    s = _ws(v)
    if s.startswith("adamw"):
        return "adamw"
    return "adam" if s.startswith("adam") else s


def _canon_box(v) -> str:
    return "largest" if "largest" in _ws(v) else _ws(v)


def _canon_num_or_none(v):
    """`none (run 4 had no background draw)` and 0.0 are the same setting."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0 if "none" in _ws(v) else _ws(v)


def _canon_inference(v) -> str:
    s = _ws(v)
    return "tiled" if "tiled" in s else ("whole" if "whole-image" in s else s)


CANON = {
    "augmentation": _canon_aug,
    "optimizer": _canon_opt,
    "box_rule": _canon_box,
    "background_frac": _canon_num_or_none,
    "inference": _canon_inference,
    "loss": _ws,
    "schedule": _ws,
    "input_scaling": lambda v: (
        "imagenet" if "imagenet" in _ws(v) else ("caffe" if "caffe" in _ws(v) else _ws(v))
    ),
    "train_store": lambda v: "merged" if "merged" in _ws(v) else _ws(v),
    "mask_merging": lambda v: "merged" if "merged" in _ws(v) else _ws(v),
    "selection_metric": lambda v: ("val_hard_iou" if _ws(v).startswith("val_hard_iou") else _ws(v)),
    "encoder_bn": lambda v: "none" if "none" in _ws(v) else "trainable",
    "seed_handling": lambda v: "seeded",
    "early_stopping": lambda v: "notbinding" if "none" in _ws(v) else _ws(v),
}


def cache_version(v) -> str:
    """v1/v2/v3 from a store path, so a path string is not mistaken for a frame."""
    s = str(v)
    # Laptop stores name the version with a suffix, pod mounts inline; both must read
    # as the same frame, or a correct launch is refused for its mount path.
    for n in ("3", "2", "1"):
        if f"_v{n}" in s or f"cache_v{n}" in s:
            return f"v{n}"
    return "v1" if "localiser_tensors" in s else s


DETECTOR_INHERENT = {
    "decoder",
    "pretrained_weights",
    "input_scaling",
    "encoder_bn",
    "loss",  # multi-task RPN+RoI, not Dice+BCE on pixels
    "schedule",
    "lr_min",
    "warmup_epochs",
    "lr_encoder",
    "lr_decoder",
    "two_rate",  # one parameter group; there is no encoder/decoder split
    "box_rule",  # highest-scoring detection, not largest component
    "inference",  # proposes boxes directly; no probability map to threshold
    "patch_size",
    "patches_per_image",
    "lesion_centred_frac",
    "patch_jitter_frac",
    "background_frac",  # trains on whole images, not sampled patches
    "output_prior_bias",
    "selection_metric",
    "secondary_checkpoint",
    "dice_bce_weights",
    # Derived, not chosen: steps_per_epoch is n_train_images / batch_size, and pure
    # arithmetic on an already inherent field cannot be an independent deviation.
    "steps_per_epoch",
}

SCALE_FIELDS = {
    "epochs",
    "epochs_requested",
    "epochs_run",
    "steps_per_epoch",
    "n_train_images",
    "n_val_lesions",
    "best_epoch",
    "best_val_hard_iou",
    "patch_size",
    "batch_size",
    "patience",
    "output_prior",
    "output_prior_bias",
    "input",
    "canvas",
}


def norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(v)
    if isinstance(v, float):
        return round(v, 8)
    if isinstance(v, str):
        return v.strip()
    return v


def equivalent(label, want, got, cand=None) -> bool:
    if label == "cache":
        return cache_version(want) == cache_version(got)
    if label == "patience":
        # What matters is not the number but whether early stopping CAN bind:
        # patience >= the candidate's own epoch count cannot, which is the recipe.
        ep = (cand or {}).get("epochs_requested") or (cand or {}).get("epochs")
        try:
            not_binding = ep is not None and float(got) >= float(ep)
        except (TypeError, ValueError):
            not_binding = "none" in _ws(got)
        return not_binding and "none" in _ws(want)
    if norm(want) == norm(got):
        return True
    fn = CANON.get(label)
    if fn is not None:
        return fn(want) == fn(got)
    return _ws(want) == _ws(got) if isinstance(want, str) and isinstance(got, str) else False


def check_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="a completed run's manifest.json")
    src.add_argument("--spec", help="a JSON spec for a run not yet launched")
    ap.add_argument(
        "--intended",
        nargs="*",
        default=[],
        help="settings that are the declared experimental variable, "
        "e.g. --intended encoder cache",
    )
    ap.add_argument(
        "--family",
        choices=["unet", "detector"],
        default=None,
        help="declare the candidate's model family. `detector` marks the rows "
        "a detector cannot match a segmentation baseline on (DETECTOR_INHERENT: "
        "loss, schedule and rates, decoder, box rule, patch sampling and others). "
        "Everything else must still match.",
    )
    ap.add_argument(
        "--inherent",
        nargs="*",
        default=[],
        help="settings the intended variable forces to differ and that cannot "
        "be matched, e.g. --inherent framework encoder_bn. Reported "
        "separately and not counted as deviations.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="the manifest came from a --limit smoke run, so scale fields "
        "(epochs, steps, split sizes) cannot match run 4 and are reported "
        "separately. Never pass this when authorising a real launch.",
    )
    ap.add_argument(
        "--spec-fallback",
        default=None,
        metavar="SPEC.json",
        help="the run's own launch spec. Where the manifest is silent, the spec "
        "supplies the setting and the row is marked 'from launch spec'; "
        "where both record it they must agree, and a disagreement counts "
        "as a deviation. The spec never overrides the manifest.",
    )
    ap.add_argument("--reference", default=str(RUN4))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not Path(args.reference).exists():
        print(f"run 4 manifest not found: {args.reference}", file=sys.stderr)
        return 2
    r4 = json.loads(Path(args.reference).read_text())
    cand = json.loads(Path(args.manifest or args.spec).read_text())
    # A torch manifest names the cache by path; run 4's is v1 by construction.
    r4v = {
        "cache": "outputs/localiser_tensors (v1, p1/p99.0)",
        "framework": RUN4_CODE["framework"][0],
        "loss": "0.5*BCE + 0.5*soft-Dice",
        "background_frac": "none (run 4 had no background draw)",
        "patience": "none (ran all 100 epochs)",
    }
    for k, (v, _src) in RUN4_CODE.items():
        r4v.setdefault(k, v)

    # The launch spec, if one was kept: it fills only the gaps a run manifest leaves,
    # and where both record a setting they must agree.
    spec = json.loads(Path(args.spec_fallback).read_text()) if args.spec_fallback else {}

    rows, deviations = [], 0
    for label, r4key in FIELDS:
        want = r4v[label] if r4key is None else r4.get(r4key, r4v.get(label))
        got, present = cand_get(cand, label)
        src = "run"
        disagrees = False
        if spec:
            sval, spresent = cand_get(spec, label, none_is_absent=True)
            if present and spresent and not equivalent(label, sval, got, cand):
                disagrees = True
            elif not present and spresent:
                got, present, src = sval, True, "spec"
        if disagrees:
            verdict, mark = "RUN DISAGREES WITH ITS LAUNCH SPEC", "!"
            deviations += 1
        elif not present and args.family == "detector" and label in DETECTOR_INHERENT:
            # A detector has no patch sampler and no pixel loss; those fields are
            # not missing, they do not exist. Absence is the correct record.
            verdict, mark = "not applicable to the detector family", "~"
        elif not present:
            verdict, mark = "not recorded", "?"
        elif equivalent(label, want, got, cand):
            verdict, mark = "same", "="
        elif label in args.intended:
            verdict, mark = "INTENDED VARIABLE", "*"
        elif args.family == "detector" and label in DETECTOR_INHERENT:
            verdict, mark = "inherent to the detector family", "~"
        elif label in args.inherent:
            verdict, mark = "inherent to the intended variable", "~"
        elif args.smoke and label in SCALE_FIELDS:
            verdict, mark = "scale (smoke run, not counted)", "s"
        else:
            verdict, mark = "DEVIATION", "!"
            deviations += 1
        rows.append(
            (mark, label, want, got, verdict + (" (from launch spec)" if src == "spec" else ""))
        )

    w = max(len(r[1]) for r in rows)
    print(f"\n{'':2} {'setting':{w}} {'run 4':52} {'candidate':52} verdict")
    print("-" * (w + 118))
    for mark, label, want, got, verdict in rows:
        print(f"{mark:2} {label:{w}} {str(want)[:52]:52} {str(got)[:52]:52} {verdict}")
    print()
    intended = [r[1] for r in rows if r[4] == "INTENDED VARIABLE"]
    inherent = [r[1] for r in rows if r[4].startswith("inherent")]
    fam = [r[1] for r in rows if r[4].endswith("the detector family")]
    scale = [r[1] for r in rows if r[4].startswith("scale")]
    unrec = [r[1] for r in rows if r[4] == "not recorded"]
    from_spec = [r[1] for r in rows if r[4].endswith("(from launch spec)")]
    if from_spec:
        print(f"from launch spec   : {from_spec}")
        print(f"  (the run manifest is silent on these; {args.spec_fallback} supplies them)")
    print(f"intended variables : {intended or 'none declared'}")
    if inherent:
        print(f"inherent           : {inherent}")
    if fam:
        print(f"  of which family  : {fam}")
    if scale:
        print(f"scale (smoke)      : {scale}")
    if unrec:
        print(
            f"NOT RECORDED       : {unrec} <- fill these in; an unrecorded setting is "
            f"a deviation you cannot see"
        )
    print(f"deviations         : {deviations}")
    ok = deviations == 0 and not unrec
    print(
        "\n"
        + (
            "LAUNCH AUTHORISED: no deviation and no unrecorded setting."
            if ok
            else "LAUNCH BLOCKED: fix the deviations, or declare them with --intended "
            "and record why."
        )
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "deviations": deviations,
                    "intended": intended,
                    "inherent": inherent,
                    "scale_exempt": scale,
                    "not_recorded": unrec,
                    "rows": [
                        {
                            "setting": r[1],
                            "run4": str(r[2]),
                            "candidate": str(r[3]),
                            "verdict": r[4],
                        }
                        for r in rows
                    ],
                },
                indent=2,
            )
        )
        print(f"-> {args.out}")
    return 0 if ok else 1


# CLI
SUBCOMMANDS = {"build": build_main, "check": check_main}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("subcommands: " + ", ".join(SUBCOMMANDS))
        return 0
    sub, rest = argv[0], argv[1:]
    if sub not in SUBCOMMANDS:
        print(f"unknown subcommand {sub!r}; one of: {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        return 2
    rc = SUBCOMMANDS[sub](rest)
    return 0 if rc is None else int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
