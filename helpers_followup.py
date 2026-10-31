"""Follow-up to the robustness study (robustness_followup.ipynb). Uses helpers.py, which is not changed.

Two questions about the nearest-neighbour method (Wide ResNet50 layer 2, border margin from held-out good images):
- Test A: does a brightness normalisation in front of the feature extraction help? Every image (memory, held-out, test) goes through
  BrightnessNormaliser first: a gain and offset (optionally also a linear shading) fitted to the mean of the good training images.
- Test B: does a memory that also contains changed copies of the good training images help (refit), compared with only setting the
  border margin and the threshold again on changed held-out good images?

Everything is repeated for several seeds. The seed changes the 80/20 split of the good training images, the random subset of memory
patches, the random draws of the changes of the test and held-out images (the images themselves are fixed) and those of the copies.
Seed 0 is the split and the draws of the baseline runs (robustness_batch.csv), so the plain method at seed 0 must reproduce them.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import helpers as h

TARGETS = (0.05, 0.01)
ALL_FAMILIES = {**h.FAMILIES, **h.GEOMETRY}
COPY_SEED = 1000  # the random draws of the changed training copies come from seed + COPY_SEED, those of the test and held-out images from seed


# --- brightness normalisation -------------------------------------------------------------------------------------

class BrightnessNormaliser:
    """Fits the grey image to the mean grey image of the good training images, grey = a * mean + b (+ c * x + d * y with shading),
    and removes the fit from all three colour channels: (image - b - c * x - d * y) / a. No alignment. The same fit is applied to
    the training, held-out and test images. The result is not clipped. A nearly black or flat image (a < 0.05) is only offset."""

    def __init__(self, mean_gray, shading=False):
        hh, ww = mean_gray.shape
        yy, xx = np.mgrid[:hh, :ww].astype("float32")
        self.xx, self.yy = xx / ww - 0.5, yy / hh - 0.5
        self.shading = shading
        cols = [mean_gray.ravel(), np.ones(hh * ww, "float32")] + ([self.xx.ravel(), self.yy.ravel()] if shading else [])
        self.pinv = np.linalg.pinv(np.stack(cols, 1).astype("float64"))  # (2 or 4, pixels)

    def __call__(self, im):
        im = h.to_float(im)
        coef = self.pinv @ cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).ravel().astype("float64")
        a, off = (coef[0] if coef[0] >= 0.05 else 1.0), coef[1]
        if self.shading:
            off = off + coef[2] * self.xx + coef[3] * self.yy
        return ((im - (off[..., None] if self.shading else off)) / a).astype("float32")


def fit_normaliser(X_fit, shading=False):
    """The normaliser for the mean grey image of the fit images X_fit (uint8 or float RGB)."""
    return BrightnessNormaliser(np.mean([h._gray(im) for im in X_fit], 0), shading)


# --- fitting ------------------------------------------------------------------------------------------------------

def copy_specs(families):
    """The changed copies for the memory: every strength (all three levels) of each family in `families`, as (family, level, function)."""
    return [(f, lab, h.make_variation(f, p)) for f in families for lab, p in h.family(f)[1]]


def _margin(maps, tol=1.05, max_frac=0.1):
    """The border margin from held-out maps: the first ring within `tol` of the interior level, as in helpers.fit_reference."""
    prof = h.ring_profile(maps)
    return min(int(np.argmax(prof <= tol * np.median(prof[len(prof) // 2:]))), int(max_frac * maps.shape[1]))


@dataclass
class NNModel:
    """The nearest-neighbour model of one category: memory (rows), optional normaliser, border margin and the scores of the held-out
    good images (unchanged), which fix the threshold. Same role as helpers.Reference."""
    memory: torch.Tensor
    norm: object
    margin: int
    val_scores: np.ndarray
    n_memory: int
    n_images: int       # images (originals and copies) the memory was drawn from
    fit_s: float

    def maps(self, imgs):
        if self.norm is not None:
            imgs = [self.norm(im) for im in imgs]
        return h.knn_maps(self.memory, imgs)


def fit_nn(cat, seed=0, shading=None, copies=(), copies_only=False, max_patches=40000, tol=1.05, max_frac=0.1, chunk=32):
    """Memory from 80% of the good training images (split `seed`), margin and threshold basis from the other 20%, as helpers.fit_reference.
    shading: None = no normalisation, False = gain and offset, True = gain, offset and linear shading (fitted to the fit images).
    copies: copy_specs(...); the memory then also holds those changed copies of the fit images (only the copies if copies_only).
    The memory is a random subset of at most max_patches patches of all images it is drawn from."""
    t0 = time.perf_counter()
    val_idx, fit_idx = h.held_out_split(cat, seed)
    norm = None if shading is None else fit_normaliser(cat.X[fit_idx], shading)
    rng_copy = lambda i: np.random.default_rng([COPY_SEED + seed, int(i)])

    def images():
        if not copies_only:
            for i in fit_idx:
                yield cat.X[i]
        for _fam, _lab, fn in copies:
            for i in fit_idx:
                yield fn(h.to_float(cat.X[i]), rng_copy(i))

    n_images = (0 if copies_only else len(fit_idx)) + len(copies) * len(fit_idx)
    total = n_images * h.GRID ** 2
    keep = np.sort(np.random.default_rng(seed).choice(total, size=min(max_patches, total), replace=False))
    rows, offset, buf = [], 0, []

    def flush():
        nonlocal offset
        if not buf:
            return
        f = h.cnn_features([norm(im) for im in buf] if norm is not None else buf).reshape(-1, 512)
        lo, hi = np.searchsorted(keep, [offset, offset + len(f)])
        rows.append(f[keep[lo:hi] - offset])
        offset += len(f)
        buf.clear()

    for im in images():
        buf.append(im)
        if len(buf) == chunk:
            flush()
    flush()
    memory = torch.from_numpy(np.concatenate(rows))
    model = NNModel(memory, norm, 0, None, len(memory), n_images, 0.0)
    val_maps = model.maps(cat.X[val_idx])
    model.margin = _margin(val_maps, tol, max_frac)
    model.val_scores = h.border_scores(val_maps, model.margin)
    model.fit_s = time.perf_counter() - t0
    return model


# --- scoring the conditions ---------------------------------------------------------------------------------------

def changed_maps(model, X, idx, fn, seed, chunk=32):
    """Nearest-neighbour maps (N, 28, 28) of the images X[idx] after the change fn; the random draws are seeded per image index
    (the same as helpers.score_augmented)."""
    out = []
    for s in range(0, len(idx), chunk):
        out.append(model.maps([fn(h.to_float(X[i]), np.random.default_rng([seed, int(i)])) for i in idx[s:s + chunk]]))
    return np.concatenate(out)


def evaluate(model, cat, conds, seed, targets=TARGETS, store=None):
    """One row per condition (family, level, function) for the fixed test images of `cat`, like helpers.run_batch, and
    - 'fixed': margin and threshold from the unchanged held-out good images (caught@, frr@),
    - 'again': threshold from the held-out good images changed in the same way, margin as fixed (recal_caught@, recal_frr@; the
      "set again" of the first study),
    - 'reset': margin and threshold both from the changed held-out good images (reset_caught@, reset_frr@).
    store (dict): filled with the test and held-out scores per condition (…__test, …__val with the fixed margin; …__test_reset,
    …__val_reset with the margin set again)."""
    val_idx, _ = h.held_out_split(cat, seed)
    idx = h.select(cat, "test")
    is_def = cat.labels[idx] != "good"
    rows = []
    for fam, lab, fn in conds:
        mt, mv = changed_maps(model, cat.X, idx, fn, seed), changed_maps(model, cat.X, val_idx, fn, seed)
        s, sv = h.border_scores(mt, model.margin), h.border_scores(mv, model.margin)
        mc = _margin(mv)
        s2, sv2 = h.border_scores(mt, mc), h.border_scores(mv, mc)
        row = {"family": fam, "level": lab, "category": cat.name, "seed": seed, "auroc": roc_auc_score(is_def, s),
               "good_median": float(np.median(s[~is_def])), "margin": model.margin, "margin_reset": mc,
               "n_memory": model.n_memory, "n_images": model.n_images, "fit_s": model.fit_s}
        for t in targets:
            row[f"caught@{t:.0%}"], row[f"frr@{t:.0%}"] = h.operating_point(model.val_scores, s[~is_def], s[is_def], t)
            row[f"recal_caught@{t:.0%}"], row[f"recal_frr@{t:.0%}"] = h.operating_point(sv, s[~is_def], s[is_def], t)
            row[f"reset_caught@{t:.0%}"], row[f"reset_frr@{t:.0%}"] = h.operating_point(sv2, s2[~is_def], s2[is_def], t)
        rows.append(row)
        if store is not None:
            store[f"{fam}__{lab}__test"], store[f"{fam}__{lab}__val"] = s, sv
            store[f"{fam}__{lab}__test_reset"], store[f"{fam}__{lab}__val_reset"] = s2, sv2
    if store is not None:
        store["is_defective"], store["ref_val"] = is_def, model.val_scores
    return rows


def run_followup(csv_path, scores_dir, method, make_model, conds, categories, seeds, log=print):
    """For every seed and category: fit make_model(cat, seed), score `conds` and append the rows to csv_path (with the column
    'method'). A finished (method, seed, category) is skipped, so an interrupted run continues. The scores of every image go to
    scores_dir/<method>_s<seed>_<category>.npz. Returns the whole table."""
    csv_path, scores_dir = Path(csv_path), Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    table = pd.read_csv(csv_path, dtype={"level": str}) if csv_path.exists() else pd.DataFrame()
    for seed in seeds:
        for ctg in categories:
            if len(table) and ((table.method == method) & (table.seed == seed) & (table.category == ctg)).sum() >= len(conds):
                continue
            t0 = time.perf_counter()
            cat = h.load_category(ctg)
            model = make_model(cat, seed)
            t_fit = time.perf_counter() - t0
            store = {}
            rows = [dict(r, method=method) for r in evaluate(model, cat, conds, seed, store=store)]
            np.savez(scores_dir / f"{method}_s{seed}_{ctg}.npz", **store)
            if len(table):
                table = table[~((table.method == method) & (table.seed == seed) & (table.category == ctg))]
            table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
            table.to_csv(csv_path, index=False)
            log(f"{method:22s} seed {seed} {ctg:11s} fit {t_fit:5.0f} s, total {time.perf_counter() - t0:5.0f} s, memory {model.n_memory} patches", flush=True)
    return table


# --- summaries ----------------------------------------------------------------------------------------------------

def strongest_conditions(families=None):
    """The unchanged images and the strongest level of every family: (family, level, function)."""
    return h.conditions(ALL_FAMILIES if families is None else families, True)


def seed_summary(df, columns=("auroc", "caught@5%", "frr@5%", "recal_caught@5%", "recal_frr@5%", "reset_caught@5%", "reset_frr@5%"),
                 categories=None):
    """Per method and family: the mean over the categories at each seed, then the mean and the standard deviation over the seeds
    (the spread that a difference has to exceed to count). Returns a table with the columns (column, 'mean'/'std')."""
    d = df if categories is None else df[df.category.isin(categories)]
    per_seed = d.groupby(["method", "family", "seed"])[list(columns)].mean()
    g = per_seed.groupby(["method", "family"]).agg(["mean", "std"])
    return g


def paired_difference(df, method, base, column, categories=None):
    """Per family, `method` minus `base` in `column`: the mean over the categories at each seed, then the mean and the std over the
    seeds of those paired differences, and the number of seeds in which the difference has the sign of the mean."""
    d = df if categories is None else df[df.category.isin(categories)]
    a = d[d.method == method].groupby(["family", "seed"])[column].mean()
    b = d[d.method == base].groupby(["family", "seed"])[column].mean()
    diff = (a - b).dropna()
    out = diff.groupby("family").agg(["mean", "std", "count"])
    out["same_sign"] = diff.groupby("family").apply(lambda x: int((np.sign(x) == np.sign(x.mean())).sum()))
    return out
