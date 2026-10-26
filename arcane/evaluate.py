"""Evaluate kNN-U, kNN-S, scANVI classifier, CellTypist, baselines and ARCANE for one run.

Usage: python evaluate.py <tag> <seed>   (reads emb/<tag>_seed<s>_{full,cf0,cf1}.npz)
Writes res/<tag>_seed<s>.json (summary), res/<tag>_seed<s>_types.csv (per-type metrics) and
res/<tag>_seed<s>_cells.npz (per-cell outputs).
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from arcane import knn_evidence, hard, run_arcane
from metrics import per_type, auroc

tag, seed = sys.argv[1], int(sys.argv[2])
K, ALPHA = 15, 0.1
os.makedirs("res", exist_ok=True)
L = lambda s: dict(np.load(f"emb/{tag}_seed{seed}_{s}.npz", allow_pickle=True))
F, cal = L("full"), [L("cf0"), L("cf1")]
classes = np.array(sorted(set(F["y_ref"])))
T = len(classes)
ci = {c: i for i, c in enumerate(classes)}
y_tgt = F["y_tgt"]
known = np.isin(y_tgt, classes)
y = np.array([ci.get(c, -1) for c in y_tgt])

res = run_arcane(F["ref_U"], F["ref_S"], F["y_ref"], F["tgt_U"], F["tgt_S"], cal, classes, K, ALPHA)
res_acc = run_arcane(F["ref_U"], F["ref_S"], F["y_ref"], F["tgt_U"], F["tgt_S"], cal, classes, K, ALPHA,
                     objective="accuracy")
# in-sample calibration (no cross-fitting): leave-one-out kNN inside the full reference embedding
insample = [dict(ref_U=F["ref_U"], ref_S=F["ref_S"], y_ref=F["y_ref"], tgt_U=F["ref_U"], tgt_S=F["ref_S"],
                 y_tgt=F["y_ref"])]
from arcane import Arbiter
PU_r, _ = knn_evidence(F["ref_U"], F["y_ref"], F["ref_U"], classes, K, exclude_self=True)
PS_r, _ = knn_evidence(F["ref_S"], F["y_ref"], F["ref_S"], classes, K, exclude_self=True)
yr = np.array([ci[c] for c in F["y_ref"]])
arb_in = Arbiter(T).fit(hard(PU_r), hard(PS_r), yr)
pred = {"kNN-U": res["yU"], "kNN-S": res["yS"], "Average": res["yavg"], "ARCANE": res["label"],
        "ARCANE-acc": res_acc["label"], "ARCANE-insample": arb_in.predict(res["PU"], res["PS"])}
# scANVI native classifier
pc = list(F["scanvi_classes"])
sc_lab = np.array(pc)[F["scanvi_prob"].argmax(1)]
pred["scANVI-clf"] = np.array([ci.get(c, -1) for c in sc_lab])
# concatenated embedding kNN
Zr = np.hstack([StandardScaler().fit_transform(F["ref_U"]), StandardScaler().fit_transform(F["ref_S"])])
sU, sS = StandardScaler().fit(F["ref_U"]), StandardScaler().fit(F["ref_S"])
Zr = np.hstack([sU.transform(F["ref_U"]), sS.transform(F["ref_S"])])
Zt = np.hstack([sU.transform(F["tgt_U"]), sS.transform(F["tgt_S"])])
pred["Concat"] = hard(knn_evidence(Zr, F["y_ref"], Zt, classes, K)[0])
ct = f"emb/{tag}_seed{seed}_celltypist.npz"
if os.path.exists(ct):
    C = np.load(ct, allow_pickle=True)
    m = pd.Series(C["pred"], index=C["tgt_names"]).reindex(F["tgt_names"]).values
    pred["CellTypist"] = np.array([ci.get(c, -1) for c in m])

n_ref = pd.Series(F["y_ref"]).value_counts().reindex(classes).fillna(0).values
n_q = np.array([(y == t).sum() for t in range(T)])
rows, summ = [], {"tag": tag, "seed": seed, "n_classes": int(T), "n_query": int(len(y)),
                  "n_known": int(known.sum()), "methods": {}}
ev = np.where(n_q >= 10)[0]
for mname, yh in pred.items():
    rec, prec, f1, _ = per_type(y[known], yh[known], T)
    summ["methods"][mname] = dict(BA=float(np.nanmean(rec[ev])), macroF1=float(np.nanmean(f1[ev])),
                                  acc=float((yh[known] == y[known]).mean()))
    for t in ev:
        rows.append(dict(tag=tag, seed=seed, type=classes[t], n_query=int(n_q[t]), n_ref=int(n_ref[t]),
                         method=mname, recall=rec[t], precision=prec[t], f1=f1[t]))
# conformal summaries on known types
S = res["sets"]
cov = [float(S[known][y[known] == t, t].mean()) for t in ev]
summ["conformal"] = dict(alpha=ALPHA, mean_cov=float(np.mean(cov)), min_cov=float(np.min(cov)),
                         per_type_cov=dict(zip(classes[ev].tolist(), cov)),
                         set_size=float(S[known].sum(1).mean()), singleton=float((S[known].sum(1) == 1).mean()),
                         empty_known=float((S[known].sum(1) == 0).mean()),
                         empty_novel=float((S[~known].sum(1) == 0).mean()) if (~known).any() else None,
                         unknown_known=float(res["unknown"][known].mean()),
                         unknown_novel=float(res["unknown"][~known].mean()) if (~known).any() else None)
# novelty detection for held-out types
if (~known).any():
    nov = ~known
    PU, PS = res["PU"], res["PS"]
    pmax = lambda P: 1 - P.max(1)
    scores = {"ARCANE": res["novelty"], "ARCANE set p-value": 1.0 - res["conformal"].pvalues(res["score"]).max(1), "1-max kNN-S": pmax(PS), "1-max kNN-U": pmax(PU),
              "1-max average": pmax(0.5 * (PU + PS)), "1-max scANVI": 1 - F["scanvi_prob"].max(1),
              "kNN distance U": res["dU"], "kNN distance S": res["dS"],
              "U-S disagreement": (res["yU"] != res["yS"]).astype(float)}
    summ["novelty_auroc"] = {k: float(auroc(v, nov)) for k, v in scores.items()}
    summ["novel_types"] = sorted(set(y_tgt[nov]))
    summ["novel_assigned_by"] = {m: pd.Series(classes[p[nov]]).value_counts().head(3).to_dict()
                                 for m, p in pred.items() if (p[nov] >= 0).all()}
# hierarchy-consistent output: if the conformal set holds several fine types with one parent, report the parent
par = dict(zip(F["y_ref"], F["parent_ref"]))
if any(par[c] != c for c in classes):
    lab_fine = classes[res["label"]]
    out_lab, at_parent = [], []
    for i in range(len(y)):
        members = classes[S[i]]
        parents = {par[c] for c in members}
        if len(members) > 1 and len(parents) == 1:
            out_lab.append(parents.pop()); at_parent.append(True)
        else:
            out_lab.append(lab_fine[i]); at_parent.append(False)
    out_lab, at_parent = np.array(out_lab), np.array(at_parent)
    true_par = F["parent_tgt"]
    correct = np.where(at_parent, out_lab == true_par, out_lab == y_tgt)
    summ["hierarchy"] = dict(frac_parent=float(at_parent[known].mean()),
                             hier_acc=float(correct[known].mean()),
                             fine_acc_arcane=float((lab_fine == y_tgt)[known].mean()),
                             parent_acc_of_parent_calls=float(correct[known & at_parent].mean()) if (known & at_parent).any() else None,
                             fine_acc_on_parent_calls=float((lab_fine == y_tgt)[known & at_parent].mean()) if (known & at_parent).any() else None)
pd.DataFrame(rows).to_csv(f"res/{tag}_seed{seed}_types.csv", index=False)
json.dump(summ, open(f"res/{tag}_seed{seed}.json", "w"), indent=1)
np.savez_compressed(f"res/{tag}_seed{seed}_cells.npz", names=F["tgt_names"], y=y_tgt, cond=F["cond_tgt"],
                    batch=F["b_tgt"], novelty=res["novelty"], unknown=res["unknown"], set_size=S.sum(1), classes=classes,
                    scanvi_maxprob=F["scanvi_prob"].max(1), knn_dist_U=res["dU"], knn_dist_S=res["dS"],
                    pmax_S=res["PS"].max(1), pmax_U=res["PU"].max(1),
                    **{f"pred_{k}": v for k, v in pred.items()})
print(tag, seed, {k: round(v["BA"], 3) for k, v in summ["methods"].items()}, flush=True)
