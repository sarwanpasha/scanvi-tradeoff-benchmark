"""Stratum effects (mean +- sd over seeds) for the training variants.

Usage: python summarize_variants.py <res_dir>
Result files are named <dataset>_<variant>_seed<s>.json, e.g. lymphoid_balanced_seed0.json
or spleen_lymph_flat_seed2.json, as written by knn_eval.py on the outputs of
train_scanvi_balanced.py and train_scpoli.py.
"""
import glob
import json
import os
import sys
from collections import defaultdict
import numpy as np

R = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_seed*.json"))):
    head, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
    ds, variant = head.rsplit("_", 1)
    rec = json.load(open(f))
    for k in ["k5", "k15", "k30", "k50"]:
        by = defaultdict(list)
        for t in rec["types"].values():
            by[t["stratum"]].append(t[k]["delta"])
        for st, v in by.items():
            R[(ds, variant, st)][k].append(float(np.mean(v)))
print("effect = supervised - unsupervised, per-type mean within stratum")
for key in sorted(R):
    v = np.array(R[key]["k15"])
    other = " ".join(f"{k}={np.mean(R[key][k]):+.4f}" for k in ["k5", "k30", "k50"])
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"{key[0]:<13} {key[1]:<9} {key[2]:<13} k15 {v.mean():+.4f} +- {sd:.4f} "
          f"seeds {np.round(v, 4).tolist()} | {other}")
