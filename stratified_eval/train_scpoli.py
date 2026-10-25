"""Train scPoli with and without cell-type labels (same architecture) and map the query.

With --hier, the supervised model uses two label levels (coarse, fine).
Output: <ds>_<flat|hier>_seed<s>.npz with unsupervised scPoli in the 'scvi' slots and
supervised scPoli in the 'scanvi' slots, so the same evaluation script applies.
"""
import anndata
anndata.read = anndata.read_h5ad  # compatibility shim for scArches with recent anndata
import argparse
import os
import numpy as np
import torch
from scarches.models.scpoli import scPoli
from common import LAB, BAT, COARSE, load_split, save_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--outdir", default="emb_scpoli")
ap.add_argument("--hier", action="store_true")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--pretrain", type=int, default=40)
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
np.random.seed(A.seed)
torch.manual_seed(A.seed)

ref, qry = load_split(A.ds, dense=True)
print(A.ds, A.seed, "ref", ref.shape, "query", qry.shape, "fine", ref.obs[LAB].nunique(),
      "coarse", ref.obs[COARSE].nunique(), "hier", A.hier, flush=True)


def run(cell_type_keys):
    kw = dict(condition_keys=[BAT], embedding_dims=5, latent_dim=10, recon_loss="nb")
    if cell_type_keys:
        kw["cell_type_keys"] = cell_type_keys
    m = scPoli(adata=ref, **kw)
    m.train(n_epochs=A.epochs, pretraining_epochs=A.pretrain, eta=5)
    q = scPoli.load_query_data(adata=qry, reference_model=m, labeled_indices=[])
    q.train(n_epochs=A.epochs, pretraining_epochs=A.pretrain, eta=10)
    return m.get_latent(ref, mean=True), q.get_latent(qry, mean=True)


zr_u, zq_u = run(None)
zr_s, zq_s = run([COARSE, LAB] if A.hier else [LAB])
tag = "hier" if A.hier else "flat"
save_embeddings(f"{A.outdir}/{A.ds}_{tag}_seed{A.seed}.npz", ref, qry, zr_u, zq_u, zr_s, zq_s)
print("saved", A.ds, tag, A.seed, flush=True)
