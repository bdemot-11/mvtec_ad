"""Threshold flow with a few reject samples (mvtec_ad_robustness.ipynb), simulated on the saved scores.

The flow on a line: the threshold is first set from good parts alone. Then known rejects are collected (k samples), the
threshold is tuned so that they are caught without too many good parts rejected, and the result is checked on parts that were
not used for tuning. Here the reject samples are defective test images; everything is evaluated on the remaining test images.

The tuning rule minimises  c * (share of the k samples missed) + (share of the held-out good images rejected),
where c is how much worse an escaped defect is than a falsely rejected good part (c = 1: equal, c = 5: five times worse).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import helpers as h


def tune_threshold(good_scores, samples, c):
    """Threshold (reject if score > threshold) with the lowest cost on the held-out good scores and the reject samples.
    Among equal costs the highest threshold, so the fewest good parts are rejected."""
    cand = np.unique(np.r_[good_scores, samples])
    cand = np.r_[cand.min() - 1e-9, cand]  # the first one rejects everything
    miss = (samples[None, :] <= cand[:, None]).mean(1)
    over = (good_scores[None, :] > cand[:, None]).mean(1)
    cost = c * miss + over
    return cand[cost == cost.min()].max()


def simulate(ctg, condition, ks=(1, 2, 5, 10, 20), costs=(1, 5), n_draws=200, scores_dir="robustness_scores", target=0.05, seed=0):
    """Results of the flow for one category and condition (family, level), many random draws of the reject samples.

    Two scenarios for where the k reject samples come from: 'seen types' (drawn from all defect types, the check is on the
    other defect images) and 'one type' (all from one defect type, the check is on the defects of the other types only: a defect
    nobody has seen yet). Three rules, all checked on the same defect images: 'fixed' (threshold from the unchanged good images,
    no samples), 'set again' (from good images changed in the same way, no samples) and 'tuned' (with the k samples).
    Returns one row per draw, scenario, rule and cost ratio c: share of defects caught, share of good test parts rejected, cost."""
    d = np.load(Path(scores_dir) / f"{ctg}.npz")
    fam, lab = condition
    s, sv, ref_val, is_def = d[f"{fam}__{lab}__test"], d[f"{fam}__{lab}__val"], d["ref_val"], d["is_defective"]
    types = h.test_labels(ctg)
    defects = np.where(is_def)[0]
    good = s[~is_def]
    thr_fixed = np.quantile(ref_val, 1 - target, method="higher")
    thr_again = np.quantile(sv, 1 - target, method="higher")
    rng = np.random.default_rng(seed)
    rows = []

    def add(scenario, rule, k, c, thr, eval_defects):
        caught, frr = float((s[eval_defects] > thr).mean()), float((good > thr).mean())
        rows.append({"category": ctg, "family": fam, "level": lab, "scenario": scenario, "rule": rule, "k": k, "c": c,
                     "caught": caught, "rejected": frr, "cost": c * (1 - caught) + frr})

    for k in ks:
        for _ in range(n_draws):
            sample = rng.choice(defects, size=k, replace=False)
            one_type = rng.choice(np.unique(types[defects]))
            of_type = defects[types[defects] == one_type]
            sample_t = rng.choice(of_type, size=min(k, len(of_type)), replace=False)
            for scenario, samples, eval_defects in (("seen types", sample, np.setdiff1d(defects, sample)),
                                                    ("one type", sample_t, defects[types[defects] != one_type])):
                for c in costs:
                    add(scenario, "fixed", k, c, thr_fixed, eval_defects)
                    add(scenario, "set again", k, c, thr_again, eval_defects)
                    add(scenario, "tuned", k, c, tune_threshold(sv, s[samples], c), eval_defects)
    return pd.DataFrame(rows)


def plot_flow(flow, c=5):
    """Cost against the number of reject samples, one panel per condition, mean over the categories.
    cost = c * (share of defects missed) + (share of good parts rejected), checked on parts not used for tuning."""
    import matplotlib.pyplot as plt
    import robustness_viz as viz
    flow = flow[flow.c == c].assign(condition=lambda d: d.family + " " + d.level)
    per = flow.groupby(["condition", "scenario", "rule", "k", "category"]).cost.mean().reset_index()
    m = per.groupby(["condition", "scenario", "rule", "k"]).cost.mean().reset_index()
    conds = list(dict.fromkeys(flow.condition))
    ks = sorted(flow.k.unique())
    xs = np.arange(len(ks) + 1)  # position 0 = no samples
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.6), facecolor=viz.SURFACE)
    for ax, cond in zip(axes, conds):
        viz._style(ax)
        ax.grid(axis="x", visible=False); ax.grid(axis="y", color=viz.GRID, lw=0.8)
        d = m[m.condition == cond]
        base = d[d.scenario == "seen types"]
        for rule, ls, col in (("fixed", ":", viz.INK2), ("set again", "--", viz.INK)):
            v = base[base.rule == rule].set_index("k").cost
            ax.plot(xs, [v.iloc[0]] + list(v.loc[ks]), ls=ls, color=col, lw=1.5, label=f"threshold {rule} ({'no samples'})")
        for scenario, col in (("seen types", viz.GOOD), ("one type", viz.DEFECT)):
            v = d[(d.scenario == scenario) & (d.rule == "tuned")].set_index("k").cost
            v0 = d[(d.scenario == scenario) & (d.rule == "set again")].set_index("k").cost.iloc[0]
            ax.plot(xs, [v0] + list(v.loc[ks]), color=col, lw=2, marker="o", ms=7, mfc=col, mec=viz.SURFACE,
                    label=f"tuned with samples ({scenario})")
        ax.set_xticks(xs); ax.set_xticklabels([0] + ks)
        ax.set_ylim(bottom=0)
        ax.set_title(cond.replace("none -", "unchanged"), loc="left", fontsize=9, color=viz.INK)
        ax.set_xlabel("reject samples", color=viz.INK2, fontsize=8)
    axes[0].set_ylabel(f"cost = {c} x missed + rejected", color=viz.INK2, fontsize=8)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=8, labelcolor=viz.INK2)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
