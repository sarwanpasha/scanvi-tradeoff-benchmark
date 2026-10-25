"""Train scVI, scANVI, and class-balanced scANVI on a reference and map the query with scArches surgery.

Writes two embedding files per seed:
  <ds>_scanvi_seed<s>.npz    scVI vs ordinary scANVI
  <ds>_balanced_seed<s>.npz  scVI vs class-balanced scANVI
"""
import argparse
import os
import numpy as np
import scvi
from common import LAB, BAT, load_split, save_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--outdir", default="emb_balanced")
ap.add_argument("--max_epochs", type=int, default=200)
ap.add_argument("--surgery_epochs", type=int, default=100)
ap.add_argument("--n_latent", type=int, default=10)
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
scvi.settings.seed = A.seed

ref, qry = load_split(A.ds)
ep = int(min(A.max_epochs, max(20, round(min(20000 / ref.n_obs * 400, 400)))))
print(A.ds, "ref", ref.shape, "query", qry.shape, "scVI epochs", ep, flush=True)


def fit_scvi(r):
    scvi.model.SCVI.setup_anndata(r, batch_key=BAT, labels_key=LAB)
    m = scvi.model.SCVI(r, n_latent=A.n_latent)
    m.train(max_epochs=ep)
    return m


def fit_scanvi(m):
    s = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category="Unknown", labels_key=LAB)
    s.train(max_epochs=max(20, ep // 10))
    return s


def project(cls, model, q):
    qq = q.copy()
    qq.obs[LAB] = "Unknown"
    cls.prepare_query_anndata(qq, model)
    qm = cls.load_query_data(qq, model)
    qm.train(max_epochs=A.surgery_epochs, plan_kwargs=dict(weight_decay=0.0))
    return qm.get_latent_representation()


def balanced_index(labels, seed):
    """Oversample every class smaller than the median class size to the median (with replacement)."""
    rng = np.random.default_rng(seed)
    u, cnt = np.unique(labels, return_counts=True)
    n_star = int(np.median(cnt))
    idx = []
    for t in u:
        I = np.where(labels == t)[0]
        idx += I.tolist() if len(I) >= n_star else rng.choice(I, n_star, replace=True).tolist()
    return np.sort(idx), n_star


m_v = fit_scvi(ref.copy())
z_rv, z_qv = m_v.get_latent_representation(), project(scvi.model.SCVI, m_v, qry)
m_a = fit_scanvi(m_v)
z_ra, z_qa = m_a.get_latent_representation(), project(scvi.model.SCANVI, m_a, qry)

idx, n_star = balanced_index(ref.obs[LAB].values, A.seed)
ref_b = ref[idx].copy()
ref_b.obs_names_make_unique()
print("n_star", n_star, "balanced reference", ref_b.n_obs, flush=True)
m_b = fit_scanvi(fit_scvi(ref_b))
z_rb = m_b.get_latent_representation(adata=ref)  # evaluate on the original reference
z_qb = project(scvi.model.SCANVI, m_b, qry)

save_embeddings(f"{A.outdir}/{A.ds}_scanvi_seed{A.seed}.npz", ref, qry, z_rv, z_qv, z_ra, z_qa)
save_embeddings(f"{A.outdir}/{A.ds}_balanced_seed{A.seed}.npz", ref, qry, z_rv, z_qv, z_rb, z_qb)
print("saved", A.ds, A.seed, flush=True)
