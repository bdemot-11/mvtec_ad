"""Measurements and plots for the robustness notebook (mvtec_ad_robustness.ipynb).

Two things are shown for the simulated variations from helpers.py:
- how big the change is, measured on the pixels (needs no model): measure_dose, plot_dose, plot_differences;
- what it does to the scores, with the fixed threshold as a line: plot_scores (needs robustness_scores/*.npz).
"""
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers as h

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
GOOD, DEFECT = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# --- how big is the change ----------------------------------------------------------------------------------------

def image_stats(old, new):
    """What a variation changed in one image (float RGB in 0..1): the size of the change, and the properties it should move."""
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var()
    clipped = lambda x: float(((x <= 0) | (x >= 1)).mean())
    diff = np.abs(new - old).max(-1)
    return {"mean abs difference": float(np.abs(new - old).mean()),
            "changed area": float((diff > 0.02).mean()),  # share of pixels that changed visibly; spots change little of the image
            "brightness": float(new.mean() / old.mean()),
            "contrast": float(new.std() / old.std()),
            "red": float(new[..., 0].mean() / old[..., 0].mean()),
            "blue": float(new[..., 2].mean() / old[..., 2].mean()),
            "sharpness": float(lap(new) / lap(old)),
            "clipped": clipped(new) - clipped(old)}


def measure_dose(n_per_category=20, seed=0):
    """Stats of every variation on a sample of good test images per category, drawn with the same per-image seeds as the scoring."""
    rows = []
    for ctg in h.CATEGORIES:
        cat = h.load_category(ctg)
        good = h.select(cat, "test", good=True)
        sample = np.random.default_rng(seed).choice(good, size=min(n_per_category, len(good)), replace=False)
        for fam, (title, levels, _) in h.FAMILIES.items():
            for lab, param in levels:
                fn = h.make_variation(fam, param)
                for i in sample:
                    old = h.to_float(cat.X[i])
                    new = fn(old, np.random.default_rng([0, int(i)]))
                    rows.append({"family": fam, "level": lab, "category": ctg, "image": int(i), **image_stats(old, new)})
    return pd.DataFrame(rows)


def plot_dose(dose, metric="mean abs difference"):
    """One row per variation: the median of `metric` over the sample, with the 5th to 95th percentile as a bar."""
    names, med, lo, hi = [], [], [], []
    for fam, (title, levels, _) in h.FAMILIES.items():
        for lab, _p in levels:
            v = dose[(dose.family == fam) & (dose.level == lab)][metric]
            names.append(f"{title.split(' (')[0]}: {lab}")
            med.append(v.median()); lo.append(v.quantile(0.05)); hi.append(v.quantile(0.95))
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(names) + 1), facecolor=SURFACE)
    _style(ax)
    ax.hlines(y, lo, hi, color=GOOD, lw=2, alpha=0.5)
    ax.scatter(med, y, s=40, color=GOOD, edgecolors=SURFACE, linewidths=1, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=8)
    ax.set_xlabel(f"{metric} (median, bar = 5th to 95th percentile over {dose.groupby(['family', 'level']).size().iloc[0]} good images)",
                  color=INK2, fontsize=8)
    ax.set_xlim(left=0)
    if metric == "changed area":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    return fig


def plot_differences(cat, i, families=None):
    """The absolute difference to the original for every variation at the three strengths, amplified. One scale per row."""
    families = h.FAMILIES if families is None else families
    fams = list(families)
    fig, ax = plt.subplots(len(fams), 3, figsize=(5.5, 1.75 * len(fams)), facecolor=SURFACE)
    old = h.to_float(cat.X[i])
    for r, fam in enumerate(fams):
        title, levels, _ = families[fam]
        diffs = [np.abs(h.make_variation(fam, p)(old, np.random.default_rng([0, int(i)])) - old).mean(-1) for _, p in levels]
        top = max(np.percentile(d, 99.5) for d in diffs) or 1.0
        for c, ((lab, _), d) in enumerate(zip(levels, diffs)):
            ax[r, c].imshow(np.clip(d / top, 0, 1), cmap="gray_r", vmin=0, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
            ax[r, c].set_title(lab, fontsize=8, color=INK2)
        ax[r, 0].set_ylabel(textwrap.fill(title, 16), fontsize=8, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig.suptitle("difference to the original (amplified, one scale per row)", fontsize=9, color=INK2, y=0.998)
    return fig


# --- what does it do to the scores --------------------------------------------------------------------------------

def plot_scores(ctg, family, target=0.05, scores_dir="robustness_scores", method=None):
    """Scores of the good and the defective test images for the unchanged images and the three strengths of one variation.
    Dashed line: the threshold fixed on the unchanged held-out good images. Tick in each row: the threshold set again on
    held-out good images changed in the same way. In the row labels: share of good test images rejected by each."""
    d = np.load(Path(scores_dir) / f"{ctg}.npz")
    is_def, ref_val = d["is_defective"], d["ref_val"]
    title, levels, _ = h.family(family)
    conds = [("none", "-", "unchanged")] + [(family, lab, lab) for lab, _ in levels]
    thr = np.quantile(ref_val, 1 - target, method="higher")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 1.15 * len(conds) + 1.4), facecolor=SURFACE)
    _style(ax)
    labels = []
    for row, (fam, lab, name) in enumerate(conds):
        s, sv = d[f"{fam}__{lab}__test"], d[f"{fam}__{lab}__val"]
        y0 = len(conds) - 1 - row
        for mask, col in ((~is_def, GOOD), (is_def, DEFECT)):
            ax.scatter(s[mask], y0 + rng.uniform(-0.25, 0.25, mask.sum()), s=30, color=col, alpha=0.85,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
        rc = np.quantile(sv, 1 - target, method="higher")
        ax.plot([rc, rc], [y0 - 0.38, y0 + 0.38], color=INK, lw=2, zorder=4)
        labels.append(f"{name}\nrejects {(s[~is_def] > thr).mean():.0%} of good ({(s[~is_def] > rc).mean():.0%} recalibrated)")
    ax.axvline(thr, color=INK, lw=1.2, ls="--", zorder=2)
    ax.set_yticks(np.arange(len(conds))[::-1])
    ax.set_yticklabels(labels, color=INK, fontsize=8)
    ax.set_ylim(-0.6, len(conds) - 0.4)
    ax.set_xlabel("anomaly score (higher = more abnormal)", color=INK2, fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GOOD, label="good"),
               plt.Line2D([], [], marker="o", ls="", color=DEFECT, label="defective"),
               plt.Line2D([], [], color=INK, ls="--", lw=1.2, label=f"threshold fixed for {target:.0%} false rejects"),
               plt.Line2D([], [], color=INK, lw=2, label="threshold set again on changed good images")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title(f"{ctg}: {title}" + (f" ({method})" if method else ""), loc="left", fontsize=10, color=INK, pad=38)
    plt.tight_layout()
    return fig


# --- the results at a glance --------------------------------------------------------------------------------------

def plot_matrices(batch, target=0.05, families=None):
    """Variation x category, at the strongest level of each variation: AUROC lost, share of good test images rejected by the
    fixed threshold, and by a threshold set again on changed good images. One sequential hue, values written in the cells."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    families = h.FAMILIES if families is None else families
    fams, cats = list(families), h.CATEGORIES
    strongest = {f: families[f][1][-1][0] for f in fams}
    base = batch[batch.family == "none"].set_index("category")

    def get(col):
        return np.array([[batch[(batch.family == f) & (batch.level == strongest[f]) & (batch.category == c)][col].iloc[0]
                          for c in cats] for f in fams])

    t = f"{target:.0%}"
    panels = [("AUROC lost", np.clip(base.loc[cats, "auroc"].values[None, :] - get("auroc"), 0, None), "{:.2f}", 0.4),
              ("good parts rejected\nfixed threshold", get(f"frr@{t}"), "{:.0%}", 1.0),
              ("good parts rejected\nthreshold set again", get(f"recal_frr@{t}"), "{:.0%}", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.42 * len(fams) + 1.6), facecolor=SURFACE, sharey=True)
    for k, (ax, (title, m, fmt, vmax)) in enumerate(zip(axes, panels)):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                ax.text(c, r, fmt.format(m[r, c]), ha="center", va="center", fontsize=8,
                        color="white" if m[r, c] / vmax > 0.45 else INK)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace("_", " ") for c in cats], fontsize=8, color=INK2, rotation=30, ha="right")
        ax.set_yticks(range(len(fams)))
        if k == 0:
            ax.set_yticklabels([f"{families[f][0].split(' (')[0]} ({strongest[f]})" for f in fams], fontsize=8, color=INK)
        ax.tick_params(length=0, labelleft=(k == 0))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    plt.tight_layout()
    return fig


def plot_compare(results, categories=("transistor", "pill", "metal_nut"), target=0.05, families=None):
    """Methods side by side at the strongest level of every variation, averaged over `categories` (bar: lowest to highest category).
    results: {method name: batch table}. Left: AUROC lost. Right: share of good parts rejected by the fixed threshold."""
    families = h.FAMILIES if families is None else families
    fams = list(families)
    strongest = {f: families[f][1][-1][0] for f in fams}
    cols = [GOOD, DEFECT, "#1baf7a"]  # slots 1 to 3 of the reference palette: the most that validate for a dot plot
    assert len(results) <= len(cols), "at most three methods"
    t = f"{target:.0%}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(fams) + 1.6), facecolor=SURFACE, sharey=True)
    y = np.arange(len(fams))[::-1]
    for ax, (title, col, xmax) in zip(axes, [("AUROC lost (unchanged minus changed)", "auroc", None), ("good parts rejected, fixed threshold", f"frr@{t}", 1.0)]):
        _style(ax)
        for k, (name, b) in enumerate(results.items()):
            b = b[b.category.isin(categories)]
            base = b[b.family == "none"].set_index("category")[col]
            vals = []
            for f in fams:
                v = b[(b.family == f) & (b.level == strongest[f])].set_index("category")[col]
                vals.append(base.loc[v.index] - v if col == "auroc" else v)
            mean = np.array([v.mean() for v in vals]); lo = np.array([v.min() for v in vals]); hi = np.array([v.max() for v in vals])
            off = (k - (len(results) - 1) / 2) * 0.26
            ax.hlines(y + off, lo, hi, color=cols[k], lw=2, alpha=0.4)
            ax.scatter(mean, y + off, s=42, color=cols[k], edgecolors=SURFACE, linewidths=1, zorder=3, label=name)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.tick_params(labelleft=False)
        if xmax:
            ax.set_xlim(-0.03, xmax + 0.03)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        else:
            ax.axvline(0, color=INK2, lw=0.8)
    axes[0].tick_params(labelleft=True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{families[f][0].split(' (')[0]} ({strongest[f]})" for f in fams], fontsize=8, color=INK)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    plt.tight_layout()
    return fig


# --- combining the two methods ------------------------------------------------------------------------------------

def fusion_table(dirs=(("robustness_scores", "robustness_scores_template"),
                       ("robustness_scores_geometry", "robustness_scores_geometry_template")), target=0.05):
    """Combine the saved scores of the neural method and the template, image by image, under every condition.
    Each score is first scaled with the mean and std of the held-out good images of its own method. Rules: the neural score,
    the template score, their mean, their maximum (reject if either says so) and their minimum (reject only if both say so).
    The threshold is fixed on the held-out good images of the fused score; 'rfrr' and 'rcaught' use the changed ones."""
    from sklearn.metrics import roc_auc_score
    rules = {"neural": lambda a, b: a, "template": lambda a, b: b, "mean": lambda a, b: (a + b) / 2, "max": np.maximum, "min": np.minimum}
    rows = []
    for dn, dt in dirs:
        for ctg in h.CATEGORIES:
            N, T = np.load(Path(dn) / f"{ctg}.npz"), np.load(Path(dt) / f"{ctg}.npz")
            is_def = N["is_defective"]
            zn = lambda x: (x - N["ref_val"].mean()) / N["ref_val"].std()
            zt = lambda x: (x - T["ref_val"].mean()) / T["ref_val"].std()
            for k in sorted({k[:-6] for k in N.files if k.endswith("__test")}):
                fam, lab = k.split("__")
                sn, st, vn, vt = zn(N[k + "__test"]), zt(T[k + "__test"]), zn(N[k + "__val"]), zt(T[k + "__val"])
                for name, f in rules.items():
                    s, ref, sv = f(sn, st), f(zn(N["ref_val"]), zt(T["ref_val"])), f(vn, vt)
                    thr, rc = np.quantile(ref, 1 - target, method="higher"), np.quantile(sv, 1 - target, method="higher")
                    rows.append({"category": ctg, "family": fam, "level": lab, "rule": name, "auroc": roc_auc_score(is_def, s),
                                 "frr": (s[~is_def] > thr).mean(), "caught": (s[is_def] > thr).mean(),
                                 "rfrr": (s[~is_def] > rc).mean(), "rcaught": (s[is_def] > rc).mean()})
    return pd.DataFrame(rows).drop_duplicates(["category", "family", "level", "rule"])


def summary_table(methods, families, categories=("transistor", "pill", "metal_nut"), exclude=("shift",)):
    """One row per method: the unchanged images, and over the changes at their strongest level (without those in `exclude`)
    the mean and worst share of good parts rejected by the fixed threshold, the mean and worst AUROC change and the mean share
    of defects caught when the threshold is set again. Returns (unchanged, over the changes)."""
    strongest = {f: v[1][-1][0] for f, v in families.items()}
    unchanged, over = {}, {}
    for name, d in methods.items():
        d = d[d.category.isin(categories)]
        base = d[d.family == "none"].set_index("category")
        keep = np.array([strongest.get(f) == l for f, l in zip(d.family, d.level)]) & (d.family != "none").values & ~d.family.isin(exclude).values
        x = d[keep].assign(auroc_change=lambda t: t.auroc - t.category.map(base.auroc))
        g = x.groupby("family").agg(rejected=("frr@5%", "mean"), auroc_change=("auroc_change", "mean"), caught_again=("recal_caught@5%", "mean"))
        unchanged[name] = base[["auroc", "caught@5%", "frr@5%"]].mean()
        over[name] = {"mean rejected": g.rejected.mean(), "worst rejected": g.rejected.max(), "mean AUROC change": g.auroc_change.mean(),
                      "worst AUROC change": g.auroc_change.min(), "mean caught, threshold set again": g.caught_again.mean()}
    return pd.DataFrame(unchanged).T, pd.DataFrame(over).T
