"""Shared dataset configuration and loading for the stratified evaluation."""
import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc

LAB, BAT, COARSE = "cell_type", "batch_key", "coarse_type"

# Reference and query batches are disjoint. Paths are relative to the working directory.
DATASETS = {
    "lymphoid": dict(h5ad="data/lymphoid_sc.h5ad", tissue="Tonsil", label="Subset_Broad", batch="Donor",
                     ref=["BCP5", "BCP6"], query=["BCP2", "BCP3", "BCP4", "BCP8", "BCP9"]),
    "pbmc_cite": dict(h5ad="data/pbmc_cite_full.h5ad", tissue=None, label="celltype.l2", batch="donor",
                      ref=["P1", "P5", "P6", "P7", "P8"], query=["P2", "P3", "P4"], coarse="celltype.l1"),
    "spleen_lymph": dict(h5ad="data/spleen_lymph_full.h5ad", tissue=None, label="cell_types", batch="batch",
                         ref=["SLN111-D1", "SLN208-D1"], query=["SLN111-D2", "SLN208-D2"]),
}


def load_split(name, n_hvg=2000, dense=False):
    """Return (reference, query) AnnData objects restricted to reference HVGs, using raw counts."""
    c = DATASETS[name]
    a = ad.read_h5ad(c["h5ad"])
    a.obs_names_make_unique()
    if c["tissue"]:
        a = a[a.obs["Tissue"].astype(str) == c["tissue"]].copy()
    if "counts" in a.layers:
        a.X = a.layers["counts"].copy()
    a.obs[LAB] = a.obs[c["label"]].astype(str)
    a.obs[BAT] = a.obs[c["batch"]].astype(str)
    a = a[~a.obs[LAB].isin(["nan", "NaN", "None", "NA", ""])].copy()
    if "coarse" in c:
        a.obs[COARSE] = a.obs[c["coarse"]].astype(str)
    else:  # lineage prefix of the subset name, e.g. "B" for "B_GC_DZ"
        a.obs[COARSE] = a.obs[LAB].str.split("[_ ]", regex=True).str[0]
    ref = a[a.obs[BAT].isin(c["ref"])].copy()
    qry = a[a.obs[BAT].isin(c["query"])].copy()
    del a
    sc.pp.highly_variable_genes(ref, n_top_genes=n_hvg, flavor="seurat_v3")
    hvg = ref.var_names[ref.var["highly_variable"]]
    ref, qry = ref[:, hvg].copy(), qry[:, hvg].copy()
    if dense:
        for x in (ref, qry):
            x.X = (x.X.toarray() if sp.issparse(x.X) else np.asarray(x.X)).astype(np.float32)
    return ref, qry


def save_embeddings(path, ref, qry, z_ref_u, z_qry_u, z_ref_s, z_qry_s):
    """Save unsupervised (slot 'scvi') and supervised (slot 'scanvi') embeddings with metadata."""
    np.savez_compressed(path, ref_scvi=z_ref_u, query_scvi=z_qry_u, ref_scanvi=z_ref_s, query_scanvi=z_qry_s,
                        ref_labels=ref.obs[LAB].values.astype(str), query_labels=qry.obs[LAB].values.astype(str),
                        ref_batches=ref.obs[BAT].values.astype(str),
                        query_batches_obs=qry.obs[BAT].values.astype(str), batch_key=BAT, label_key=LAB)
