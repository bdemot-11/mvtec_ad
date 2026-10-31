"""Is the damage of a simulated shift or rotation only the strip it uncovers? (mvtec_ad_robustness.ipynb)

A shift or a rotation fills the uncovered strip or corners with a mirrored copy of the part. This scores the unchanged, the
shifted (32 px) and the rotated (8 degrees) test images with and without a frame around the image ignored, for the template
(frames of 40 and 60 px) and for the neural method (frames of 3 and 4 patches, 54 and 73 px), thresholds from the unchanged
held-out good images scored the same way.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import helpers as h

TEMPLATE_FRAMES_PX = (0, 40, 60)
NEURAL_FRAMES_PATCHES = (None, 3, 4)  # None: the margin the method chose for itself
OBJECTS = ("transistor", "pill", "metal_nut")


def _images(cat, idx, fn):
    return [fn(h.to_float(cat.X[i]), np.random.default_rng([0, int(i)])) for i in idx]


def _template_scores(ref, imgs, frames):
    """Template scores for every frame size at once: {frame: scores}."""
    out = {f: [] for f in frames}
    for im in imgs:
        g = h._align_ecc(h._gray(im), ref.tmpl)
        z = cv2.GaussianBlur(np.abs(g - ref.mean) / ref.std, (h.BLUR, h.BLUR), 0)
        for f in frames:
            out[f].append(np.percentile(z[f:-f, f:-f] if f else z, h.PCTL))
    return {f: np.array(v) for f, v in out.items()}


def run(csv_path="shift_frame_check.csv"):
    conditions = {"unchanged": lambda im, rng: im, "shift 32 px": h.make_variation("shift", 32),
                  "rotation 8 degrees": h.make_variation("rotate", 8)}
    rows = []
    for ctg in OBJECTS:
        cat = h.load_category(ctg)
        val_idx, _ = h.held_out_split(cat)
        idx = h.select(cat, "test")
        is_def = cat.labels[idx] != "good"
        val_imgs = _images(cat, val_idx, conditions["unchanged"])
        test_imgs = {n: _images(cat, idx, f) for n, f in conditions.items()}

        def add(method, label, by_condition, val_scores):
            thr = np.quantile(val_scores, 0.95, method="higher")
            for name, s in by_condition.items():
                rows.append({"method": method, "category": ctg, "frame ignored": label, "condition": name,
                             "auroc": roc_auc_score(is_def, s), "good rejected": float((s[~is_def] > thr).mean())})

        ref = h.fit_template_reference(cat)
        val_t = _template_scores(ref, val_imgs, TEMPLATE_FRAMES_PX)
        test_t = {n: _template_scores(ref, im, TEMPLATE_FRAMES_PX) for n, im in test_imgs.items()}
        for f in TEMPLATE_FRAMES_PX:
            add("golden template", f"{f} px", {n: test_t[n][f] for n in test_t}, val_t[f])

        nref = h.fit_reference(cat)
        val_maps = h.knn_maps(nref.memory, val_imgs)
        test_maps = {n: h.knn_maps(nref.memory, im) for n, im in test_imgs.items()}
        for m in NEURAL_FRAMES_PATCHES:
            margin = nref.margin if m is None else m
            label = f"{margin} patches" + (" (own)" if m is None else f" ({margin * 512 // 28} px)")
            add("neural network", label, {n: h.border_scores(mp, margin) for n, mp in test_maps.items()}, h.border_scores(val_maps, margin))
        print(ctg, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(csv_path, index=False)
    return out


def load(csv_path="shift_frame_check.csv"):
    return pd.read_csv(csv_path) if Path(csv_path).exists() else run(csv_path)
