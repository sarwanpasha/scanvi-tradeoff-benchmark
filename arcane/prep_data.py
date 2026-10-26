"""Download public datasets and write standardized AnnData files to data/std/<name>.h5ad.

Standardized obs columns: label (fine type), parent (coarse type or label), batch, role
(ref/query), fold (0/1 for reference cells, -1 otherwise), condition (default 'none').
X holds raw counts. Usage: python prep_data.py <name>
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

os.makedirs("data/std", exist_ok=True)
name = sys.argv[1]


def assign_folds(a, ref_batches):
    """Split reference batches into two folds with similar cell counts (greedy)."""
    sizes = a.obs.loc[a.obs.batch.isin(ref_batches), "batch"].value_counts()
    load, fold = [0, 0], {}
    for b, n in sizes.sort_values(ascending=False).items():
        f = int(load[1] < load[0])
        fold[b] = f
        load[f] += n
    a.obs["fold"] = a.obs.batch.map(fold).fillna(-1).astype(int)


def finish(a, ref, qry, label, batch, parent=None, condition=None):
    a.obs["label"] = a.obs[label].astype(str)
    a.obs["batch"] = a.obs[batch].astype(str)
    a.obs["parent"] = a.obs[parent].astype(str) if parent else a.obs["label"]
    a.obs["condition"] = a.obs[condition].astype(str) if condition else "none"
    a = a[~a.obs.label.isin(["nan", "NaN", "None", "NA", "", "unknown", "Unknown"])].copy()
    a.obs["role"] = np.where(a.obs.batch.isin(ref), "ref", np.where(a.obs.batch.isin(qry), "query", "none"))
    a = a[a.obs.role != "none"].copy()
    assign_folds(a, ref)
    if "counts" in a.layers:
        a.X = a.layers["counts"]
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    keep = ["label", "parent", "batch", "role", "fold", "condition"]
    a = ad.AnnData(X=a.X, obs=a.obs[keep].copy(), var=pd.DataFrame(index=a.var_names.astype(str)))
    a.var_names_make_unique()
    a.write_h5ad(f"data/std/{name}.h5ad", compression="gzip")
    print(name, a.shape, a.obs.role.value_counts().to_dict(), "labels", a.obs.label.nunique(),
          "parents", a.obs.parent.nunique(), "folds", a.obs.groupby("fold").size().to_dict(), flush=True)


if name == "pbmc_fine":
    import scvi
    a = scvi.data.pbmc_seurat_v4_cite_seq(save_path="data/raw/")
    finish(a, ["P1", "P5", "P6", "P7", "P8"], ["P2", "P3", "P4"], "celltype.l2", "donor", parent="celltype.l1")
elif name == "spleen_lymph":
    import scvi
    a = scvi.data.spleen_lymph_cite_seq(save_path="data/raw/")
    finish(a, ["SLN111-D1", "SLN208-D1"], ["SLN111-D2", "SLN208-D2"], "cell_types", "batch")
elif name == "heart":
    import scvi
    a = scvi.data.heart_cell_atlas_subsampled(save_path="data/raw/")
    print(a.obs.columns.tolist(), flush=True)
    bcol = "donor" if "donor" in a.obs else "cell_source"
    donors = a.obs[bcol].value_counts()
    ref, qry, tot, run = [], [], donors.sum(), 0
    for d, n in donors.items():  # largest donors first; about 60% of cells to the reference
        (ref if run < 0.6 * tot else qry).append(d)
        if d in ref:
            run += n
    finish(a, ref, qry, "cell_type", bcol)
elif name == "lupus":
    a = ad.read_h5ad("data/raw/lupus.h5ad")
    a.obs["donor"] = a.obs.donor_id.astype(str)
    ref = sorted(a.obs.loc[a.obs.role_hint == "ref", "donor"].unique())
    qry = sorted(a.obs.loc[a.obs.role_hint == "query", "donor"].unique())
    counts = a.obs.cell_type.value_counts()
    a = a[a.obs.cell_type.isin(counts.index[counts >= 50])].copy()
    finish(a, ref, qry, "cell_type", "donor", condition="disease")
elif name == "hlca":
    a = ad.read_h5ad("data/raw/hlca_core.h5ad", backed="r")
    obs = a.obs
    print(obs.columns.tolist(), flush=True)
    lab = "ann_finest_level" if "ann_finest_level" in obs else "cell_type"
    par = "ann_level_2" if "ann_level_2" in obs else None
    bcol = "dataset" if "dataset" in obs else "study"
    sizes = obs[bcol].value_counts()
    chosen = sizes.index[:8].tolist()
    rng = np.random.default_rng(0)
    idx = []
    for b in chosen:
        ii = np.where(obs[bcol].values == b)[0]
        idx += sorted(rng.choice(ii, min(len(ii), 30000), replace=False).tolist())
    a = a[np.sort(idx)].to_memory()
    if "raw" in dir(a) and a.raw is not None:
        a = ad.AnnData(X=a.raw.X, obs=a.obs, var=a.raw.var)
    if "feature_name" in a.var:
        a.var_names = a.var["feature_name"].astype(str)
    finish(a, chosen[:5], chosen[5:8], lab, bcol, parent=par)
else:
    raise SystemExit("unknown dataset " + name)
