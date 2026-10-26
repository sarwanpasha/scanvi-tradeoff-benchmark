"""Aggregate evaluation outputs into result tables (printed and saved under summary/)."""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

os.makedirs("summary", exist_ok=True)
J = [json.load(open(f)) for f in sorted(glob.glob("res/*_seed*.json"))]
TY = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("res/*_types.csv"))], ignore_index=True)
TY["ds"] = TY.tag.str.split("_ho-|_ds-", regex=True).str[0]
TY["setting"] = np.where(TY.tag.str.contains("_ho-"), "holdout", np.where(TY.tag.str.contains("_ds-"), "downsample", "main"))
TY.to_csv("summary/all_types.csv", index=False)
METHODS = ["kNN-U", "kNN-S", "scANVI-clf", "CellTypist", "Average", "Concat", "ARCANE-acc", "ARCANE-insample", "ARCANE"]

# 1. main table: BA and macro-F1, mean +- sd over seeds
rows = []
for j in J:
    if "_ho-" in j["tag"] or "_ds-" in j["tag"]:
        continue
    for m, v in j["methods"].items():
        rows.append(dict(ds=j["tag"], seed=j["seed"], method=m, **v))
M = pd.DataFrame(rows)
M.to_csv("summary/main_methods.csv", index=False)
tab = M.groupby(["ds", "method"]).agg(BA=("BA", "mean"), BA_sd=("BA", "std"), F1=("macroF1", "mean"),
                                      F1_sd=("macroF1", "std"), acc=("acc", "mean"), n=("seed", "count"))
print("== main: balanced accuracy and macro-F1 ==")
print(tab.round(4).to_string())

# 2. per-type no-harm and worst-type change relative to max(kNN-U, kNN-S)
main = TY[TY.setting == "main"]
piv = main.pivot_table(index=["ds", "seed", "type", "n_ref", "n_query"], columns="method", values="recall").reset_index()
best = piv[["kNN-U", "kNN-S"]].max(1)
out = []
for m in [c for c in METHODS if c in piv]:
    d = piv[m] - best
    out.append(dict(method=m, no_harm=(d >= -0.01).mean(), mean_vs_best=d.mean(), worst=d.min(),
                    gain_vs_U=(piv[m] - piv["kNN-U"]).mean(), types=len(d)))
NH = pd.DataFrame(out)
print("\n== per-type recall relative to the better of kNN-U and kNN-S (main runs, type x seed) ==")
print(NH.round(4).to_string(index=False))
NH.to_csv("summary/no_harm.csv", index=False)

# 3. stratified by reference abundance tertile (within dataset): recall, precision, F1
def strat(g):
    q33, q66 = np.percentile(g.n_ref, 33), np.percentile(g.n_ref, 66)
    return np.where(g.n_ref <= q33, "rare", np.where(g.n_ref > q66, "abundant", "intermediate"))
main = main.copy()
main["stratum_ref"] = main.groupby(["ds", "seed", "method"], group_keys=False).apply(lambda g: pd.Series(strat(g), index=g.index))
ST = main.groupby(["stratum_ref", "method"])[["recall", "precision", "f1"]].mean().unstack(0)
print("\n== stratified by reference abundance (mean over dataset x seed x type) ==")
print(ST.round(4).to_string())
ST.to_csv("summary/strata_ref.csv")

# 4. paired tests ARCANE vs others on per-type F1 (type x seed pairs)
pf = main.pivot_table(index=["ds", "seed", "type"], columns="method", values="f1")
print("\n== paired Wilcoxon on per-type F1, ARCANE minus method ==")
for m in [c for c in METHODS if c in pf and c != "ARCANE"]:
    d = (pf["ARCANE"] - pf[m]).dropna()
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    print(f"{m:<16} mean diff {d.mean():+.4f}  wins {(d > 0).sum()} ties {(d == 0).sum()} losses {(d < 0).sum()}  p={p:.2e}")

# 5. conformal coverage
C = pd.DataFrame([dict(ds=j["tag"], seed=j["seed"], **{k: v for k, v in j["conformal"].items() if k != "per_type_cov"})
                  for j in J])
print("\n== conformal (alpha=0.1) ==")
cols = [c for c in ["mean_cov", "min_cov", "set_size", "singleton", "empty_known", "empty_novel", "unknown_known", "unknown_novel"] if c in C]
print(C.groupby("ds")[cols].mean().round(3).to_string())
C.to_csv("summary/conformal.csv", index=False)
cov_rows = [dict(ds=j["tag"], seed=j["seed"], type=t, cov=c) for j in J for t, c in j["conformal"]["per_type_cov"].items()]
pd.DataFrame(cov_rows).to_csv("summary/conformal_per_type.csv", index=False)

# 6. novelty detection
N = pd.DataFrame([dict(tag=j["tag"], seed=j["seed"], **j["novelty_auroc"]) for j in J if "novelty_auroc" in j])
if len(N):
    print("\n== novelty AUROC for held-out types ==")
    print(N.groupby("tag").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
    N.to_csv("summary/novelty.csv", index=False)
    HO = TY[TY.setting == "holdout"].groupby(["tag", "method"])["recall"].mean().unstack()
    print("\n== holdout runs: mean recall on known types ==")
    print(HO.round(3).to_string())

# 7. downsampling: recall/precision of the downsampled type
D = TY[TY.setting == "downsample"].copy()
if len(D):
    D["target"] = D.tag.str.extract(r"_ds-(.*)-\d+$")[0]
    D = D[D.type == D.target]
    print("\n== downsampled type: recall / F1 by method ==")
    print(D.groupby(["tag", "method"])[["recall", "precision", "f1", "n_ref"]].mean().round(3).to_string())
    D.to_csv("summary/downsample.csv", index=False)
print("\nAGG_DONE")
