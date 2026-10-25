"""Summarize per-type results produced by knn_eval.py.

Usage: python summarize_strata.py --res_dir out --cent_dir out_cent --nested lymphoid pbmc_cite spleen_lymph

Expects files named <dataset>_seed<s>.json in --res_dir (and in --cent_dir for the neighborhood analysis).
Reports (1) per-dataset stratum means at k=15, (2) the pooled nested group for every k,
(3) seed-level variation, (4) within- vs cross-lineage errors, and (5) the correlation between
the effect on non-rare types and the distance-weighted gain of their 4 nearest rare types.
"""
import argparse
import glob
import json
import os
import re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res_dir", default="out")
ap.add_argument("--cent_dir", default=None)
ap.add_argument("--nested", nargs="+", default=["lymphoid", "pbmc_cite", "spleen_lymph"])
A = ap.parse_args()
KS, STRATA = [5, 15, 30, 50], ["abundant", "intermediate", "rare"]

R = {}
for f in sorted(glob.glob(os.path.join(A.res_dir, "*_seed*.json"))):
    ds, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
    R[(ds, int(seed))] = json.load(open(f))
DS = sorted({d for d, _ in R})


def pooled(dsets, k, st, key=None):
    v = [t[f"k{k}"] for (d, _), r in R.items() if d in dsets for t in r["types"].values() if t["stratum"] == st]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean([x["scVI"] for x in v])), float(np.mean([x["scANVI"] for x in v])), len(v)


print("== per-dataset stratum means, k=15 (unsupervised, supervised, effect) ==")
for ds in DS:
    row = []
    for st in STRATA:
        u, s, n = pooled([ds], 15, st)
        row.append(f"{st[:3]} {u:.3f} {s:.3f} {s - u:+.4f} (n={n})")
    print(f"{ds:<14}", " | ".join(row))

print("\n== nested group, all k ==")
for k in KS:
    print(f"k={k:<3}", " | ".join(f"{st[:3]} {pooled(A.nested, k, st)[1] - pooled(A.nested, k, st)[0]:+.4f}" for st in STRATA))

print("\n== seed-level effect per dataset, abundant stratum, k=15 ==")
for ds in DS:
    v = []
    for (d, s), r in sorted(R.items()):
        if d == ds:
            a = [t["k15"]["delta"] for t in r["types"].values() if t["stratum"] == "abundant"]
            v.append(np.mean(a))
    print(f"{ds:<14} mean {np.mean(v):+.4f} sd {np.std(v, ddof=1) if len(v) > 1 else 0:.4f} seeds {np.round(v, 4).tolist()}")

STOP = {"cell", "cells", "positive", "negative", "like", "the", "of", "and"}
toks = lambda s: {w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if w and w not in STOP}
print("\n== error decomposition, nested group, k=15 (pooled over cells, top-6 predicted labels per type) ==")
agg = {}
for (ds, _), r in R.items():
    if ds not in A.nested:
        continue
    for name, t in r["types"].items():
        for meth, conf in t.get("confusion_k15", {}).items():
            tot = sum(conf.values())
            win = sum(c for p, c in conf.items() if p != name and toks(p) & toks(name))
            cro = sum(c for p, c in conf.items() if p != name and not toks(p) & toks(name))
            a = agg.setdefault((t["stratum"], meth), [0, 0, 0])
            a[0] += tot; a[1] += win; a[2] += cro
for st in STRATA:
    for meth in ["scVI", "scANVI"]:
        tot, win, cro = agg.get((st, meth), [0, 0, 0])
        if tot:
            print(f"{st:<13}{meth:<7} error {(win + cro) / tot:.4f} within {win / tot:.4f} cross {cro / tot:.4f}"
                  f" within% {100 * win / max(win + cro, 1):.1f}")

if A.cent_dir:
    print("\n== neighborhood analysis: effect of non-rare types vs gain of 4 nearest rare types ==")
    X, Y, G = [], [], []
    for (ds, seed), r in R.items():
        cf = os.path.join(A.cent_dir, f"{ds}_seed{seed}.json")
        if ds not in A.nested or not os.path.exists(cf):
            continue
        D = json.load(open(cf))["dist_scanvi"]
        thr = np.percentile([t["n"] for t in r["types"].values()], 33)
        big = {n: t for n, t in r["types"].items() if t["n"] > thr}
        rare = {n: t for n, t in r["types"].items() if t["n"] <= thr}
        for P, tp in big.items():
            if P not in D:
                continue
            near = sorted(((c, D[P][c]) for c in rare if c in D[P]), key=lambda z: z[1])[:4]
            if not near:
                continue
            X.append(tp["k15"]["delta"])
            Y.append(sum(rare[c]["k15"]["delta"] / (1.0 + dv) for c, dv in near))
            G.append(ds)
    X, Y, G = map(np.asarray, (X, Y, G))
    r0 = float(np.corrcoef(X, Y)[0, 1])
    rng = np.random.default_rng(0)
    groups = sorted(set(G))
    bs = []
    for _ in range(5000):
        pick = rng.choice(len(groups), len(groups))
        xi = np.concatenate([X[G == groups[i]] for i in pick])
        yi = np.concatenate([Y[G == groups[i]] for i in pick])
        if xi.std() > 0 and yi.std() > 0:
            bs.append(np.corrcoef(xi, yi)[0, 1])
    print(f"n={len(X)} r={r0:+.3f} dataset-resampled 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    for g in groups:
        m = G == g
        print(f"  {g:<14} n={m.sum():<4} r={np.corrcoef(X[m], Y[m])[0, 1]:+.3f}")
