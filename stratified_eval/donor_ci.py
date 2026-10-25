"""Donor-level effects with bootstrap confidence intervals over held-out donors.

Usage: python donor_ci.py <res_dir> [nested datasets ...]
"""
import glob
import json
import os
import sys
from collections import defaultdict
import numpy as np

src = sys.argv[1]
NESTED = sys.argv[2:] or ["lymphoid", "pbmc_cite", "spleen_lymph"]
D = defaultdict(list)  # (dataset, donor, stratum) -> per-seed mean effect
for f in sorted(glob.glob(os.path.join(src, "*_seed*.json"))):
    ds = os.path.basename(f)[:-5].rsplit("_seed", 1)[0]
    rec = json.load(open(f))
    acc = defaultdict(list)
    for t in rec["types"].values():
        for dn, v in t.get("per_donor_k15", {}).items():
            acc[(dn, t["stratum"])].append(v["scANVI"] - v["scVI"])
    for (dn, st), v in acc.items():
        D[(ds, dn, st)].append(np.mean(v))

rng = np.random.default_rng(0)


def boot(vals, B=10000):
    vals = np.asarray(vals)
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(B)]
    return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


for st in ["abundant", "intermediate", "rare"]:
    print(f"\n== {st} ==")
    pool = []
    for ds in sorted({k[0] for k in D}):
        vals = [np.mean(v) for (d, dn, s), v in D.items() if d == ds and s == st]
        if len(vals) >= 2:
            m, lo, hi = boot(vals)
            print(f"{ds:<14} donors={len(vals):<3} effect={m:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
            if ds in NESTED:
                pool += vals
    if len(pool) >= 2:
        m, lo, hi = boot(pool)
        print(f"{'nested pool':<14} donors={len(pool):<3} effect={m:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
