"""Fetch a donor-balanced subset of the Perez et al. (2022) lupus PBMC atlas from CELLxGENE Census.

Reference: healthy donors. Query: held-out healthy donors and patients with systemic lupus
erythematosus (SLE). Raw counts are kept.
"""
import numpy as np
import cellxgene_census

DID = "218acb0f-9f2f-4f76-b90b-15a4b7c7f629"
rng = np.random.default_rng(0)
with cellxgene_census.open_soma(census_version="2025-11-08") as c:
    obs = cellxgene_census.get_obs(c, "homo_sapiens", value_filter=f"dataset_id == '{DID}'",
                                   column_names=["soma_joinid", "donor_id", "disease", "cell_type"])
    print(obs.disease.value_counts().to_string(), flush=True)
    per = obs.groupby(["disease", "donor_id"]).size().rename("n").reset_index()
    per = per[per.n >= 1500]
    healthy = per[per.disease == "normal"].sample(frac=1, random_state=0).donor_id.tolist()[:14]
    sle = per[per.disease != "normal"].sample(frac=1, random_state=0).donor_id.tolist()[:8]
    keep = []
    for d in healthy + sle:
        ids = obs.loc[obs.donor_id == d, "soma_joinid"].values
        keep += rng.choice(ids, min(3500, len(ids)), replace=False).tolist()
    keep = np.sort(np.array(keep))
    print("donors", len(healthy), len(sle), "cells", len(keep), flush=True)
    a = cellxgene_census.get_anndata(c, "homo_sapiens", obs_coords=keep,
                                     obs_column_names=["donor_id", "disease", "cell_type"],
                                     var_column_names=["feature_name"])
a.var_names = a.var.feature_name.astype(str).values
a.var.index.name = None
a.var_names_make_unique()
a.obs["role_hint"] = np.where(a.obs.donor_id.isin(healthy[:10]), "ref", "query")
a.uns["healthy_ref"] = healthy[:10]
a.write_h5ad("data/raw/lupus.h5ad")
print("LUPUS_OK", a.shape, a.obs.groupby(["role_hint", "disease"]).size().to_dict(), flush=True)
