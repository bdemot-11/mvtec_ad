"""Runs Test A and Test B of the follow-up (see helpers_followup.py and robustness_followup.ipynb) with several worker processes.

    python robustness_followup_run.py worker 4      -> one worker (start several, each takes the next unclaimed job; resumable)
    python robustness_followup_run.py merge         -> merge the parts into robustness_batch_nn_norm.csv and robustness_batch_nn_refit.csv

Jobs (a job is one method at one seed over the three objects, a few to 60 minutes with 2 threads):
  Test A, all seeds: plain (the baseline), norm_gain (gain and offset), norm_shade (gain, offset and linear shading), 15 conditions;
  Test B, all seeds: refit_mixed (original + changed copies of darker, brighter, uneven, blur, lens_spots), and for each of those five
  families refit_only_<family> (changed copies of that family only). The baseline of Test B is the plain run of Test A.
"""
import os
import sys
import time
import traceback
from pathlib import Path

N_THREADS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "worker" else 2
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"

import cv2
import pandas as pd
import torch

import helpers as h
import helpers_followup as hf

cv2.setNumThreads(1)
torch.set_num_threads(2)

SEEDS = range(5)
OBJECTS = ["transistor", "pill", "metal_nut"]
B_FAMILIES = ["darker", "brighter", "uneven", "blur", "lens_spots"]
A_CSV, A_SCORES = "robustness_batch_nn_norm.csv", "robustness_batch_nn_norm_scores"
B_CSV, B_SCORES = "robustness_batch_nn_refit.csv", "robustness_batch_nn_refit_scores"
RUN = Path("robustness_batch_nn_run")      # parts, claims and logs of this run
PARTS, CLAIMS, DONE = RUN / "parts", RUN / "claims", RUN / "done"


def jobs():
    out = []
    for seed in SEEDS:
        for name, shading in (("plain", None), ("norm_gain", False), ("norm_shade", True)):
            out.append(dict(id=f"A_{name}_s{seed}", method=name, seed=seed, scores=A_SCORES,
                            make=lambda cat, s, sh=shading: hf.fit_nn(cat, s, shading=sh), conds=lambda: hf.strongest_conditions()))
    for seed in SEEDS:
        fams = {f: h.FAMILIES[f] for f in B_FAMILIES}
        out.append(dict(id=f"B_refit_mixed_s{seed}", method="refit_mixed", seed=seed, scores=B_SCORES,
                        make=lambda cat, s: hf.fit_nn(cat, s, copies=hf.copy_specs(B_FAMILIES)), conds=lambda fams=fams: h.conditions(fams, True)))
        for f in B_FAMILIES:
            out.append(dict(id=f"B_refit_only_{f}_s{seed}", method=f"refit_only_{f}", seed=seed, scores=B_SCORES,
                            make=lambda cat, s, f=f: hf.fit_nn(cat, s, copies=hf.copy_specs([f]), copies_only=True),
                            conds=lambda f=f: h.conditions({f: h.FAMILIES[f]}, True)))
    return out


def claim(job_id):
    try:
        os.close(os.open(CLAIMS / job_id, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def worker():
    for d in (PARTS, CLAIMS, DONE):
        d.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    for j in jobs():
        if (DONE / j["id"]).exists() or not claim(j["id"]):
            continue
        t0 = time.perf_counter()
        print(f"[{pid}] start {j['id']}", flush=True)
        try:
            hf.run_followup(PARTS / f"{j['id']}.csv", j["scores"], j["method"], j["make"], j["conds"](), OBJECTS, [j["seed"]],
                            log=lambda *a, **k: print(f"[{pid}]", *a, **k))
            (DONE / j["id"]).touch()
            print(f"[{pid}] done {j['id']} in {time.perf_counter() - t0:.0f} s", flush=True)
        except Exception:
            print(f"[{pid}] FAILED {j['id']}\n{traceback.format_exc()}", flush=True)
            (CLAIMS / j["id"]).unlink(missing_ok=True)
    print(f"[{pid}] no more jobs", flush=True)


def merge():
    read = lambda ids: pd.concat([pd.read_csv(PARTS / f"{i}.csv", dtype={"level": str}) for i in ids if (DONE / i).exists()] or [pd.DataFrame(columns=["family", "level", "method"])], ignore_index=True)
    a = read([j["id"] for j in jobs() if j["id"].startswith("A_")])
    b = read([j["id"] for j in jobs() if j["id"].startswith("B_")])
    # the baseline of Test B: the plain runs of Test A at the conditions that Test B tests
    tested = set(zip(b.family, b.level)) if len(b) else set()
    base = a[(a.method == "plain") & pd.Series([(f, l) in tested for f, l in zip(a.family, a.level)], index=a.index)].assign(method="orig_only")
    a.to_csv(A_CSV, index=False)
    pd.concat([base, b], ignore_index=True).to_csv(B_CSV, index=False)
    print(f"{A_CSV}: {len(a)} rows, {B_CSV}: {len(b) + len(base)} rows")


if __name__ == "__main__":
    {"worker": worker, "merge": merge}[sys.argv[1]]()
