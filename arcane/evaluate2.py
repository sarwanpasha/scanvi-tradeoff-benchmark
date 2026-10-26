"""Evaluate ARCANE (symmetric cross-fitting) and baselines for one run.

Usage: python evaluate2.py <tag> <seed> [k]
Reads emb/<tag>_seed<s>_{full,cf0,cf1}.npz (cf targets = held-out fold cells followed by query cells)
and, if present, emb/<tag>_seed<s>_celltypist.npz. Writes res2/<tag>_seed<s>[_k<k>].json, _types.csv,
_cells.npz.
"""
import json
import os
import sys
import time
import zlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from arcane import knn_evidence, hard, run_arcane, run_arcane_sym, _fold_p
from metrics import per_type, auroc

tag, seed = sys.argv[1], int(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 15
ALPHA, ALPHA_U = 0.1, 0.05
OUT = "res2"
os.makedirs(OUT, exist_ok=True)
suffix = "" if K == 15 else f"_k{K}"
t0 = time.time()
L = lambda s: dict(np.load(f"emb/{tag}_seed{seed}_{s}.npz", allow_pickle=True))
F, folds = L("full"), [L("cf0"), L("cf1")]

# partial labels: only labelled reference cells serve as neighbours and calibration cells
partial = None
if "_partial-" in tag:
    partial = float(tag.split("_partial-")[1].split("_")[0])


def restrict(d):
    if partial is None or "labeled_ref" not in d:
        return d
    m = d["labeled_ref"].astype(bool)
    d = dict(d)
    for key in ("ref_U", "ref_S", "y_ref", "parent_ref", "b_ref"):
        if key in d:
            d[key] = d[key][m]
    return d


F = restrict(F)
folds = [restrict(f) for f in folds]
lab_mask = None
if partial is not None:
    lab_mask = lambda names: np.array([zlib.crc32(str(n).encode()) % 1000 < partial * 1000 for n in names])

classes = np.array(sorted(set(F["y_ref"])))
T = len(classes)
ci = {c: i for i, c in enumerate(classes)}
y_tgt = F["y_tgt"]
known = np.isin(y_tgt, classes)
y = np.array([ci.get(c, -1) for c in y_tgt])

res = run_arcane_sym(F, folds, classes, K, ALPHA, ALPHA_U, labeled_mask=lab_mask)
res_acc = run_arcane_sym(F, folds, classes, K, ALPHA, ALPHA_U, objective="accuracy", labeled_mask=lab_mask)
# previous (asymmetric) calibration for comparison: held-out fold only, full model for query
cal_old = []
for f in folds:
    m = f["tgt_role"] == "cal"
    cal_old.append(dict(ref_U=f["ref_U"], ref_S=f["ref_S"], y_ref=f["y_ref"], tgt_U=f["tgt_U"][m],
                        tgt_S=f["tgt_S"][m], y_tgt=f["y_tgt"][m]))
res_old = run_arcane(F["ref_U"], F["ref_S"], F["y_ref"], F["tgt_U"], F["tgt_S"], cal_old, classes, K, ALPHA)

pred = {"kNN-U": res["yU"], "kNN-S": res["yS"], "Average": res["yavg"], "ARCANE": res["label"],
        "ARCANE-acc": res_acc["label"]}
pc = list(F["scanvi_classes"])
prob = F["scanvi_prob"]
pred["scANVI-clf"] = np.array([ci.get(c, -1) for c in np.array(pc)[prob.argmax(1)]])
# Saerens et al. EM label-shift correction of the scANVI posterior
prior_ref = pd.Series(F["y_ref"]).value_counts(normalize=True).reindex(pc).fillna(1e-6).values
w = prior_ref.copy()
for _ in range(100):
    post = prob * (w / prior_ref)
    post /= post.sum(1, keepdims=True)
    w_new = post.mean(0)
    if np.abs(w_new - w).max() < 1e-6:
        break
    w = w_new
pred["scANVI-EM"] = np.array([ci.get(c, -1) for c in np.array(pc)[post.argmax(1)]])
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
ev = np.where(n_q >= 10)[0]
pi = n_q[ev] / n_q[ev].sum()
order = np.argsort(pi)
pi_rev = np.empty_like(pi)
pi_rev[order] = pi[order[::-1]]          # reversed abundance ranking: heavy label shift
rows, summ = [], {"tag": tag, "seed": seed, "k": K, "n_classes": int(T), "n_query": int(len(y)),
                  "n_known": int(known.sum()), "methods": {}}
for mname, yh in pred.items():
    rec, prec, f1, _ = per_type(y[known], yh[known], T)
    summ["methods"][mname] = dict(BA=float(np.nanmean(rec[ev])), macroF1=float(np.nanmean(f1[ev])),
                                  acc=float((yh[known] == y[known]).mean()),
                                  acc_reversed=float(np.nansum(pi_rev * rec[ev])))
    for t in ev:
        rows.append(dict(tag=tag, seed=seed, type=classes[t], n_query=int(n_q[t]), n_ref=int(n_ref[t]),
                         method=mname, recall=rec[t], precision=prec[t], f1=f1[t]))

S = res["sets"]


def conf_summary(sets, unknown, unknown_type, known=known):
    cov = [float(sets[known][y[known] == t, t].mean()) for t in ev if (y[known] == t).sum() >= 20]
    return dict(mean_cov=float(np.mean(cov)), min_cov=float(np.min(cov)), frac_types_cov085=float(np.mean(np.array(cov) >= 0.85)),
                per_type_cov=dict(zip([c for t, c in zip(ev, classes[ev]) if (y[known] == t).sum() >= 20], cov)),
                set_size=float(sets[known].sum(1).mean()), singleton=float((sets[known].sum(1) == 1).mean()),
                empty=float((sets[known].sum(1) == 0).mean()),
                unknown_known=float(unknown[known].mean()),
                unknown_novel=float(unknown[~known].mean()) if (~known).any() else None,
                unknown_type_known=float(unknown_type[known].mean()) if unknown_type is not None else None,
                unknown_type_novel=float(unknown_type[~known].mean()) if (unknown_type is not None and (~known).any()) else None)


summ["conformal"] = conf_summary(S, res["unknown"], res["unknown_type"])
summ["conformal_old"] = conf_summary(res_old["sets"], res_old["unknown"], None)
summ["conformal_bonf"] = conf_summary(res["sets_bonf"], res["unknown_bonf"], res["unknown_type_bonf"])
summ["conformal_rand"] = conf_summary(res["sets_rand"], res["unknown_bonf"], res["unknown_type_bonf"])
# recalibration of the unknown flag on control donors of the query study (label-free): each control cell
# is scored against the other control donors (leave one donor out), patient cells against all controls;
# every control donor contributes the same number of cells
cond = np.array([str(c) for c in F["cond_tgt"]]) if "cond_tgt" in F else None
donor = np.array([str(b) for b in F["b_tgt"]]) if "b_tgt" in F else None
p_ctrl = None
if cond is not None and (cond == "normal").any() and (cond != "normal").any():
    ctrl = cond == "normal"
    nov_full = res["novelty"]
    rngc = np.random.default_rng(seed)
    dons = np.unique(donor[ctrl])
    per = int(max(50, min(300, np.percentile([(donor == d).sum() for d in dons], 10))))
    pick = {d: rngc.choice(np.where(donor == d)[0], min(per, (donor == d).sum()), replace=False) for d in dons}
    allcal = np.concatenate([nov_full[pick[d]] for d in dons])
    p_ctrl = np.ones(len(nov_full))
    p_ctrl[~ctrl] = _fold_p(allcal, nov_full[~ctrl])
    for d in dons:
        cal_d = np.concatenate([nov_full[pick[e]] for e in dons if e != d])
        m = donor == d
        p_ctrl[m] = _fold_p(cal_d, nov_full[m])
    u_ctrl = p_ctrl <= ALPHA_U
    don_fa = pd.Series(u_ctrl[ctrl]).groupby(donor[ctrl]).mean()
    summ["conformal_ctrl"] = conf_summary(S, res["unknown"], res["unknown_type"], known=known & ctrl)
    summ["conformal_bonf_ctrl"] = conf_summary(res["sets_bonf"], res["unknown_bonf"], res["unknown_type_bonf"], known=known & ctrl)
    summ["conformal_old_ctrl"] = conf_summary(res_old["sets"], res_old["unknown"], None, known=known & ctrl)
    summ["control_recal"] = dict(n_ctrl_donors=int(len(dons)), cells_per_donor=per,
                                 fa_ctrl_lodo=float(u_ctrl[ctrl].mean()), fa_ctrl_donor_q90=float(don_fa.quantile(0.9)),
                                 flag_patient=float(u_ctrl[~ctrl].mean()),
                                 excess_patient=float((u_ctrl[~ctrl].mean() - ALPHA_U) / (1 - ALPHA_U)),
                                 ref_cal_fa_ctrl=float(res["unknown"][ctrl].mean()),
                                 ref_cal_flag_patient=float(res["unknown"][~ctrl].mean()),
                                 bonf_fa_ctrl=float(res["unknown_bonf"][ctrl].mean()),
                                 bonf_flag_patient=float(res["unknown_bonf"][~ctrl].mean()))
summ["calibration_counts"] = dict(zip(classes.tolist(), res["cal_counts"].tolist()))
if (~known).any():
    nov = ~known
    PU, PS = res["PU"], res["PS"]
    pmax = lambda P: 1 - P.max(1)
    scores = {"ARCANE": 1 - res["p_unknown"], "ARCANE U only": 1 - res["p_unknown_U"],
              "ARCANE S only": 1 - res["p_unknown_S"], "ARCANE Bonferroni": 1 - res["p_unknown_bonf"],
              "ARCANE type Bonferroni": 1 - res["p_unknown_type_bonf"], "ARCANE full-model distance": res["novelty"],
              "set p-value": 1.0 - res["pvals"].max(1), "1-max kNN-S": pmax(PS), "1-max kNN-U": pmax(PU),
              "1-max scANVI": 1 - prob.max(1), "kNN distance U": res["dU"], "kNN distance S": res["dS"],
              "U-S disagreement": (res["yU"] != res["yS"]).astype(float)}
    summ["novelty_auroc"] = {k: float(auroc(v, nov)) for k, v in scores.items()}
    summ["novel_types"] = sorted(set(y_tgt[nov]))
# hierarchy: report the parent when the set holds several fine types that share one parent
par = dict(zip(F["y_ref"], F["parent_ref"]))
if any(par[c] != c for c in classes):
    lab_fine = classes[res["label"]]
    at_parent = np.zeros(len(y), bool)
    out_lab = lab_fine.astype(object).copy()
    for i in range(len(y)):
        members = classes[S[i]]
        parents = {par[c] for c in members}
        if len(members) > 1 and len(parents) == 1:
            out_lab[i] = parents.pop()
            at_parent[i] = True
    true_par = F["parent_tgt"]
    fine_ok = lab_fine == y_tgt
    par_ok = np.array([par.get(c, "") for c in lab_fine]) == true_par
    hier_ok = np.where(at_parent, out_lab == true_par, fine_ok)
    summ["hierarchy"] = dict(frac_parent=float(at_parent[known].mean()), hier_acc=float(hier_ok[known].mean()),
                             fine_acc=float(fine_ok[known].mean()), parent_acc_of_fine_label=float(par_ok[known].mean()),
                             fine_acc_on_parent_calls=float(fine_ok[known & at_parent].mean()) if (known & at_parent).any() else None,
                             parent_acc_on_parent_calls=float((out_lab == true_par)[known & at_parent].mean()) if (known & at_parent).any() else None,
                             fine_acc_on_fine_calls=float(fine_ok[known & ~at_parent].mean()))
summ["runtime_min"] = {s: (L(s)["runtime"] / 60).round(2).tolist() if "runtime" in L(s) else None for s in ("full", "cf0", "cf1")}
summ["gpu_peak_gb"] = {s: float(L(s).get("gpu_peak_gb", np.nan)) for s in ("full", "cf0", "cf1")}
summ["ram_peak_gb"] = {s: float(L(s).get("ram_peak_gb", np.nan)) for s in ("full", "cf0", "cf1")}
summ["eval_minutes"] = (time.time() - t0) / 60
pd.DataFrame(rows).to_csv(f"{OUT}/{tag}_seed{seed}{suffix}_types.csv", index=False)
json.dump(summ, open(f"{OUT}/{tag}_seed{seed}{suffix}.json", "w"), indent=1)
if K == 15:
    np.savez_compressed(f"{OUT}/{tag}_seed{seed}_cells.npz", names=F["tgt_names"], y=y_tgt, cond=F["cond_tgt"],
                        batch=F["b_tgt"], novelty=res["novelty"], p_unknown=res["p_unknown"], unknown=res["unknown"],
                        unknown_type=res["unknown_type"], p_unknown_U=res["p_unknown_U"], p_unknown_S=res["p_unknown_S"],
                        set_size=S.sum(1), classes=classes, pvals=res["pvals"].astype(np.float32), sets=S,
                        scanvi_maxprob=prob.max(1), knn_dist_U=res["dU"], knn_dist_S=res["dS"],
                        pmax_S=res["PS"].max(1), pmax_U=res["PU"].max(1),
                        p_unknown_bonf=res["p_unknown_bonf"], p_unknown_type_bonf=res["p_unknown_type_bonf"],
                        sets_bonf=res["sets_bonf"], set_size_bonf=res["sets_bonf"].sum(1),
                        p_unknown_ctrl=p_ctrl if p_ctrl is not None else np.ones(len(y)),
                        label_arcane=classes[res["label"]],
                        **{f"pred_{k}": classes[np.clip(v, 0, T - 1)] for k, v in pred.items()})
print(tag, seed, K, {k: round(v["BA"], 3) for k, v in summ["methods"].items()}, flush=True)
