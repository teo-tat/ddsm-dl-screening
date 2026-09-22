#!/usr/bin/env python
"""test_localise_torch_port.py — the PyTorch localiser trainer follows the reference recipe.

Every check is CPU-only and runs in .torch-env. Preprocessing, augmentation, precision and
the learning-rate schedule run the port's own code; the optimiser settings and the output bias
are rebuilt here with the recipe's values (docs/run4_recipe.json), so those checks pin the
values, not train.py.
"""

from __future__ import annotations

import ast
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.localise_torch import dataset as ds  # noqa: E402
from src.localise_torch import evaluate as ev  # noqa: E402
from src.localise_torch import train as tr  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def stop_if_failed(phase):
    if FAIL:
        print(f"\n*** STOPPED at {phase}: {FAIL[-1]} ***")
        print(f"{len(PASS)} passed, {len(FAIL)} failed")
        sys.exit(1)


# 3.1
print("\n--- 3.1 preprocessing ---")
x = torch.rand(2, 1, 8, 8)
y = ds.to_encoder_input(x)
mean = torch.tensor(ds.IMAGENET_MEAN).view(1, 3, 1, 1)
std = torch.tensor(ds.IMAGENET_STD).view(1, 3, 1, 1)
expect = (x.repeat(1, 3, 1, 1) - mean) / std
check("shape (N,1,H,W) -> (N,3,H,W)", tuple(y.shape) == (2, 3, 8, 8), str(tuple(y.shape)))
check(
    "matches ImageNet mean/std to 1e-6",
    torch.allclose(y, expect, atol=1e-6),
    f"max abs diff {float((y - expect).abs().max()):.2e}",
)
check(
    "channels are replicas of the single input channel",
    bool(torch.allclose(x[:, 0], (y[:, 0] * std[0, 0] + mean[0, 0]), atol=1e-6)),
)
# The same function object at all three call sites.
check(
    "train.py uses dataset.to_encoder_input (same object)",
    tr.to_encoder_input is ds.to_encoder_input,
)
check(
    "evaluate.py uses dataset.to_encoder_input (same object)",
    ev.to_encoder_input is ds.to_encoder_input,
)
src_ev = pathlib.Path(ROOT / "src/localise_torch/evaluate.py").read_text()
check(
    "evaluate.py calls it in both tiled and whole paths",
    src_ev.count("to_encoder_input(") == 2,
    f"{src_ev.count('to_encoder_input(')} call sites",
)
stop_if_failed("3.1")

# 3.2
print("\n--- 3.2 augmentation ---")
rng = np.random.default_rng(28)
img0 = np.zeros((32, 32), np.float32)
msk0 = np.zeros((32, 32), np.float32)
flips, degs = 0, []
for _ in range(1000):
    _, _, info = ds.augment_pair(img0, msk0, rng)
    flips += int(info["flipped"])
    degs.append(info["deg"])
rate = flips / 1000
se = math.sqrt(0.5 * 0.5 / 1000)
check(
    f"flip rate {rate:.3f} within 4 SE of 0.5",
    abs(rate - 0.5) <= 4 * se,
    f"|{rate:.3f}-0.5| = {abs(rate-0.5):.4f} vs 4SE = {4*se:.4f}",
)
check(
    "rotation angles inside +-AUG_MAX_ROTATION_DEG",
    max(abs(d) for d in degs) <= ds.AUG_MAX_ROTATION_DEG + 1e-9,
    f"max |deg| {max(abs(d) for d in degs):.4f}",
)
check(
    "AUG_MAX_ROTATION_DEG matches config",
    ds.AUG_MAX_ROTATION_DEG == config.AUG_MAX_ROTATION_DEG,
    f"{ds.AUG_MAX_ROTATION_DEG} vs {config.AUG_MAX_ROTATION_DEG}",
)
# Alignment through the real path: bilinear resample, then re-threshold.
sq_m = np.zeros((64, 64), np.float32)
sq_m[20:44, 20:44] = 1.0
sq_i = sq_m.copy()  # image == mask, so alignment is checkable
rng2 = np.random.default_rng(7)
worst = 1.0
for _ in range(50):
    ai, am, _ = ds.augment_pair(sq_i, sq_m, rng2)
    a = ai > 0.5
    b = am > 0.5
    inter = float((a & b).sum())
    union = float((a | b).sum())
    worst = min(worst, inter / union if union else 1.0)
check(
    "image and mask stay aligned through bilinear+re-threshold (IoU 1.0)",
    worst == 1.0,
    f"worst IoU over 50 draws {worst:.6f}",
)
# The flip is horizontal, i.e. on the width axis of the store's (H, W) layout.
asym = np.zeros((8, 16), np.float32)
asym[:, :4] = 1.0  # mass on the LEFT
rngf = np.random.default_rng(0)
got_flip = None
for _ in range(50):
    a, _, info = ds.augment_pair(asym, np.zeros_like(asym), rngf, max_rot_deg=0.0)
    if info["flipped"]:
        got_flip = a
        break
check("a flip actually occurred in 50 draws", got_flip is not None)
if got_flip is not None:
    check(
        "flip is on the WIDTH axis (mass moves left->right, rows unchanged)",
        bool(got_flip[:, -4:].sum() > got_flip[:, :4].sum())
        and bool(np.allclose(got_flip.sum(axis=1), asym.sum(axis=1))),
        f"left {got_flip[:, :4].sum():.0f} right {got_flip[:, -4:].sum():.0f}",
    )
    check("flip equals img[:, ::-1] exactly", bool(np.array_equal(got_flip, asym[:, ::-1])))
stop_if_failed("3.2")

# 3.3
print("\n--- 3.3 precision ---")
import segmentation_models_pytorch as smp  # noqa: E402

m = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
xb = ds.to_encoder_input(torch.rand(2, 1, 64, 64))
logits = m(xb)
loss = tr.dice_bce(logits, torch.zeros_like(logits))
loss.backward()
check("logits dtype float32 in a real train step", logits.dtype == torch.float32, str(logits.dtype))
check("loss dtype float32", loss.dtype == torch.float32, str(loss.dtype))
hits = {}
for f in ("dataset.py", "evaluate.py", "train.py"):
    s = pathlib.Path(ROOT / "src/localise_torch" / f).read_text()
    tree = ast.parse(s)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    hits[f] = sorted(names & {"autocast", "GradScaler", "amp"})
check(
    "no autocast/GradScaler/amp symbols in dataset|evaluate|train",
    all(not v for v in hits.values()),
    str(hits),
)
check(
    "no --amp flag in the CLI",
    '"--amp"' not in pathlib.Path(ROOT / "src/localise_torch/train.py").read_text(),
)
stop_if_failed("3.3")

# 3.4
print("\n--- 3.4 optimiser and schedule ---")
recipe = json.loads((ROOT / "docs/run4_recipe.json").read_text())["recipe"]
enc = [p for n, p in m.named_parameters() if n.startswith("encoder")]
dec = [p for n, p in m.named_parameters() if not n.startswith("encoder")]
kw = dict(betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0)
oe = torch.optim.Adam(enc, lr=1e-4, **kw)
od = torch.optim.Adam(dec, lr=1e-3, **kw)
check(
    "two separate Adam instances (not one with param groups)",
    isinstance(oe, torch.optim.Adam) and isinstance(od, torch.optim.Adam) and oe is not od,
)
check("optimiser is Adam, NOT AdamW", type(oe).__name__ == "Adam" and type(od).__name__ == "Adam")
check("encoder lr 1e-4", oe.param_groups[0]["lr"] == 1e-4)
check("decoder lr 1e-3", od.param_groups[0]["lr"] == 1e-3)
check(
    "weight_decay 0 on both",
    oe.param_groups[0]["weight_decay"] == 0.0 and od.param_groups[0]["weight_decay"] == 0.0,
)
check(
    "eps 1e-7 (Keras default), not torch's 1e-8",
    oe.param_groups[0]["eps"] == 1e-7 and od.param_groups[0]["eps"] == 1e-7,
    f"{oe.param_groups[0]['eps']}",
)
check("betas 0.9/0.999", oe.param_groups[0]["betas"] == (0.9, 0.999))


def run4_lr(step, peak, floor, warm, total):
    """The recipe's WarmupCosine schedule, re-implemented here from its formula."""
    if step < warm:
        return peak * (step + 1.0) / max(1, warm)
    prog = min(max((step - warm) / max(1, total - warm), 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))


steps, epochs, warm_ep = 287, 100, 5
warm, total = steps * warm_ep, steps * epochs
floor = recipe["lr_min"]["value"]
ok = True
for peak in (recipe["lr_encoder"]["value"], recipe["lr_decoder"]["value"]):
    for s in (0, warm - 1, warm, (warm + total) // 2, total - 1):
        want = run4_lr(s, peak, floor, warm, total)
        got = tr.warmup_cosine_factor(s, peak, floor, warm, total) * peak
        if abs(want - got) > 1e-9:
            ok = False
            print(f"      step {s} peak {peak}: want {want:.12e} got {got:.12e}")
check("schedule matches the recipe formula to 1e-9 at steps 0 / warm-1 / warm / mid / final", ok)
check(
    "final LR == lr_min 1e-6",
    abs(tr.warmup_cosine_factor(total - 1, 1e-3, floor, warm, total) * 1e-3 - floor) < 2e-9,
)
stop_if_failed("3.4")

# 3.5
print("\n--- 3.5 output bias ---")
prior = recipe["output_prior"]["value"]
bias = float(np.log(prior / (1 - prior)))
mb = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
torch.nn.init.constant_(mb.segmentation_head[0].bias, bias)
check(
    f"output conv bias == {bias:.4f}",
    abs(float(mb.segmentation_head[0].bias.item()) - bias) < 1e-6,
    f"{float(mb.segmentation_head[0].bias.item()):.6f}",
)
check("recipe bias is -5.30", abs(bias - (-5.300016908492776)) < 1e-9, f"{bias:.6f}")
with torch.no_grad():
    out = torch.sigmoid(mb(ds.to_encoder_input(torch.rand(2, 1, 64, 64)))).mean().item()
check(
    f"untrained mean sigmoid {out:.4f} ~ prior {prior:.4f}",
    abs(out - prior) < 0.02,
    f"|{out:.4f}-{prior:.4f}| = {abs(out-prior):.4f}",
)
stop_if_failed("3.5")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed  (3.1-3.5)")
