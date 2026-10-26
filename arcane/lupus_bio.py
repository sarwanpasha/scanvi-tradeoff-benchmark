"""Lupus analysis on the Perez et al. data (reference = healthy donors, query = held-out healthy and SLE donors).

Uses res/lupus_seed*_cells.npz and data/std/lupus.h5ad. Reports annotation accuracy by condition,
fractions of cells called unknown by condition and type, the correlation between the novelty
score and an interferon-stimulated gene (ISG) score, and genes enriched in flagged stimulated cells.
"""
import glob
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import spearmanr

ISG = ["ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "IFI44L", "IFI44",
       "RSAD2", "LY6E", "IFITM3", "XAF1", "HERC5", "EPSTI1", "STAT1", "IRF7", "ISG20", "SAMD9L", "CMPK2"]
a = ad.read_h5ad("data/std/lupus.h5ad")
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
genes = [g for g in ISG if g in a.var_names]
sc.tl.score_genes(a, genes, score_name="ISG")
out = {"isg_genes_used": genes, "seeds": {}}
rows = []
for f in sorted(glob.glob("res/lupus_seed*_cells.npz")):
    seed = int(f.split("seed")[1].split("_")[0])
    C = np.load(f, allow_pickle=True)
    df = pd.DataFrame({"name": C["names"], "y": C["y"], "cond": C["cond"], "novelty": C["novelty"],
                       "set_size": C["set_size"], "batch": C["batch"]})
    classes = C["classes"]
    for k in C.files:
        if k.startswith("pred_"):
            p = C[k]
            df[k[5:]] = np.where(p >= 0, classes[np.clip(p, 0, len(classes) - 1)], "none")
    df["isg"] = a.obs.loc[df.name, "ISG"].values
    df["flag"] = C["unknown"].astype(bool) if "unknown" in C.files else (df.set_size == 0)
    df["empty_set"] = df.set_size == 0
    for extra in ["scanvi_maxprob", "knn_dist_U", "knn_dist_S", "pmax_S", "pmax_U"]:
        if extra in C.files:
            df[extra] = C[extra]
    res = {}
    for cond, g in df.groupby("cond"):
        res[cond] = {m: float((g[m] == g.y).mean()) for m in ["kNN-U", "kNN-S", "scANVI-clf", "ARCANE", "Average"] if m in g}
        res[cond]["flagged"] = float(g.flag.mean())
    stim = df[df.cond != df.cond.iloc[0]] if df.cond.nunique() > 1 else df
    stim_name = [c for c in df.cond.unique() if "lupus" in c.lower()]
    s = df[df.cond.isin(stim_name)]
    res["rho_novelty_isg_sle"] = float(spearmanr(s.novelty, s.isg)[0])
    res["rho_novelty_isg_all"] = float(spearmanr(df.novelty, df.isg)[0])
    res["flag_by_type"] = df.groupby(["y", "cond"]).flag.mean().unstack().round(3).to_dict()
    from scipy.stats import mannwhitneyu
    sle = df[df.cond.isin(stim_name)]
    res["isg_flagged_vs_kept_sle"] = {t: dict(n_flag=int(g.flag.sum()), isg_flag=float(g[g.flag].isg.mean()),
                                              isg_kept=float(g[~g.flag].isg.mean()),
                                              p=float(mannwhitneyu(g[g.flag].isg, g[~g.flag].isg).pvalue))
                                      for t, g in sle.groupby("y") if g.flag.sum() >= 20 and (~g.flag).sum() >= 20}
    don = df.groupby(["batch", "cond"]).agg(flag=("flag", "mean"), isg=("isg", "mean")).reset_index()
    res["donor_table"] = don.round(4).values.tolist()
    res["rho_donor_flag_isg"] = float(spearmanr(don.flag, don.isg)[0])
    # compare alternative uncertainty scores as predictors of the ISG score in SLE cells
    res["rho_isg_sle_by_score"] = {k: float(spearmanr(sle[k], sle.isg)[0]) for k in
                                   ["novelty", "scanvi_maxprob", "knn_dist_U", "knn_dist_S", "pmax_S", "pmax_U"] if k in sle}
    out["seeds"][seed] = res
    df["seed"] = seed
    rows.append(df)
D = pd.concat(rows)
D.to_csv("summary/lupus_cells.csv", index=False)
# genes enriched in flagged vs unflagged stimulated cells (seed 0), within the most common type
s0 = D[(D.seed == D.seed.min()) & D.cond.str.lower().str.contains("lupus")]
top_type = s0.y.value_counts().index[0]
sub = a[s0.name.values].copy()
sub.obs["flag"] = pd.Categorical(np.where(s0.flag.values, "flagged", "kept"))
sub.obs["type"] = s0.y.values
de = {}
for t in s0.y.value_counts().index[:4]:
    x = sub[sub.obs.type == t].copy()
    if x.obs.flag.nunique() < 2 or (x.obs.flag == "flagged").sum() < 20:
        continue
    sc.tl.rank_genes_groups(x, "flag", groups=["flagged"], reference="kept", method="wilcoxon")
    r = sc.get.rank_genes_groups_df(x, "flagged").head(15)
    de[t] = r[["names", "logfoldchanges", "pvals_adj"]].round(4).values.tolist()
out["de_flagged_vs_kept_stim"] = de
json.dump(out, open("summary/lupus_bio.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1)[:3000])
for sd, r in out["seeds"].items():
    print(sd, {k: v for k, v in r.items() if k != "flag_by_type"})
