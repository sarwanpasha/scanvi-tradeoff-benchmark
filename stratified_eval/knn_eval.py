"""Per-type kNN label-transfer accuracy for two embeddings stored in one .npz file.

Usage: python knn_eval.py <embeddings.npz> <out.json>

The 'scvi' slots hold the unsupervised embedding and the 'scanvi' slots the supervised one.
Types are assigned to abundance strata by the 33rd and 66th percentiles of query type sizes.
Types with fewer than 10 query cells are not evaluated.
"""
import json
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

KS = [5, 15, 30, 50]
MIN_CELLS = 10

path, out = sys.argv[1], sys.argv[2]
d = np.load(path, allow_pickle=True)
rv, ra, qv, qa = d["ref_scvi"], d["ref_scanvi"], d["query_scvi"], d["query_scanvi"]
rl, ql = d["ref_labels"], d["query_labels"]
qb = d["query_batches_obs"] if "query_batches_obs" in d.files else None
if qb is not None and np.asarray(qb).shape != np.asarray(ql).shape:
    qb = None

types, counts = np.unique(ql, return_counts=True)
size = {t: int(c) for t, c in zip(types, counts)}
p33, p66 = (float(np.percentile(list(size.values()), q)) for q in (33, 66))


def stratum(t):
    s = size[t]
    return "rare" if s <= p33 else ("abundant" if s > p66 else "intermediate")


pred = {}
for k in KS:
    pred[("scVI", k)] = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(rv, rl).predict(qv)
    pred[("scANVI", k)] = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(ra, rl).predict(qa)

rec = {"file": os.path.basename(path), "n_ref": int(len(rl)), "n_query": int(len(ql)),
       "p33": p33, "p66": p66, "has_donor": qb is not None, "types": {}}
for t in types:
    if size[t] < MIN_CELLS:
        continue
    m = ql == t
    e = {"n": size[t], "stratum": stratum(t)}
    for k in KS:
        u = float(np.mean(pred[("scVI", k)][m] == t))
        s = float(np.mean(pred[("scANVI", k)][m] == t))
        e[f"k{k}"] = {"scVI": u, "scANVI": s, "delta": s - u, "harm": u - s}
    if qb is not None:
        per = {}
        for b in np.unique(np.asarray(qb)[m]):
            mb = m & (np.asarray(qb) == b)
            if mb.sum() >= MIN_CELLS:
                per[str(b)] = {"n": int(mb.sum()),
                               "scVI": float(np.mean(pred[("scVI", 15)][mb] == t)),
                               "scANVI": float(np.mean(pred[("scANVI", 15)][mb] == t))}
        e["per_donor_k15"] = per
    conf = {}
    for meth in ["scVI", "scANVI"]:
        uu, cc = np.unique(pred[(meth, 15)][m], return_counts=True)
        top = np.argsort(-cc)[:6]
        conf[meth] = {str(uu[j]): int(cc[j]) for j in top}
    e["confusion_k15"] = conf
    rec["types"][str(t)] = e

json.dump(rec, open(out, "w"), indent=1, default=float)
print("wrote", out, "types", len(rec["types"]), flush=True)
