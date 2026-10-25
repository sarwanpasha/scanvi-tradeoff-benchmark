"""Pairwise distances between reference cell-type centroids in both embeddings.

Usage: python centroids.py <embeddings.npz> <out.json>
"""
import json
import os
import sys
import numpy as np

path, out = sys.argv[1], sys.argv[2]
d = np.load(path, allow_pickle=True)
rl, ql = d["ref_labels"], d["query_labels"]
types, counts = np.unique(ql, return_counts=True)
cs = {str(t): d["ref_scanvi"][rl == t].mean(0) for t in np.unique(rl)}
cu = {str(t): d["ref_scvi"][rl == t].mean(0) for t in np.unique(rl)}
names = sorted(cs)
dist = lambda C: {a: {b: float(np.linalg.norm(C[a] - C[b])) for b in names} for a in names}
json.dump({"file": os.path.basename(path), "query_size": {str(t): int(c) for t, c in zip(types, counts)},
           "ref_types": names, "dist_scanvi": dist(cs), "dist_scvi": dist(cu)}, open(out, "w"), indent=1)
print("wrote", out, flush=True)
