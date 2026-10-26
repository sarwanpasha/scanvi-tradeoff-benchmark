"""Train scVI and scANVI on a reference and map target cells with scArches surgery.

--split full : train on all reference cells, target = query cells
--split cf0/cf1 : train on reference fold 1/0, target = reference fold 0/1 (cross-fitting)
--holdout T : remove type T from all reference cells (it stays in the query)
--downsample T:N : keep N reference cells of type T
Output npz: embeddings of training reference cells and target cells, labels, batches,
scANVI class probabilities on targets, target obs names.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", default="full", choices=["full", "cf0", "cf1"])
ap.add_argument("--holdout", default=None)
ap.add_argument("--downsample", default=None)
ap.add_argument("--n_latent", type=int, default=10)
ap.add_argument("--max_epochs", type=int, default=200)
ap.add_argument("--surgery_epochs", type=int, default=0, help="0: min(100, max(30, round(4e6/n_target)))")
ap.add_argument("--outdir", default="emb")
A = ap.parse_args()
os.makedirs(A.outdir, exist_ok=True)
tag = A.ds + (f"_ho-{A.holdout}" if A.holdout else "") + (f"_ds-{A.downsample}" if A.downsample else "")
tag = tag.replace(" ", "~").replace("/", "-").replace(":", "-")
out = f"{A.outdir}/{tag}_seed{A.seed}_{A.split}.npz"
if os.path.exists(out):
    raise SystemExit("exists " + out)
scvi.settings.seed = A.seed
torch.set_float32_matmul_precision("high")
t0 = time.time()

a = ad.read_h5ad(f"data/std/{A.ds}.h5ad")
ref = a[a.obs.role == "ref"].copy()
if A.holdout:
    A.holdout = A.holdout.replace("~", " ")
    ref = ref[ref.obs.label != A.holdout].copy()
if A.downsample:
    typ, n = A.downsample.replace("~", " ").rsplit(":", 1)
    rng = np.random.default_rng(A.seed)
    ii = np.where(ref.obs.label.values == typ)[0]
    drop = np.setdiff1d(ii, rng.choice(ii, min(int(n), len(ii)), replace=False))
    ref = ref[np.setdiff1d(np.arange(ref.n_obs), drop)].copy()
if A.split == "full":
    train, tgt = ref, a[a.obs.role == "query"].copy()
else:
    f = int(A.split[-1])
    train, tgt = ref[ref.obs.fold != f].copy(), ref[ref.obs.fold == f].copy()
del a
sc.pp.highly_variable_genes(train, n_top_genes=2000, flavor="seurat_v3")
hvg = train.var_names[train.var.highly_variable]
train, tgt = train[:, hvg].copy(), tgt[:, hvg].copy()
ep = int(min(A.max_epochs, max(20, round(8e6 / train.n_obs))))
sep = A.surgery_epochs or int(min(100, max(30, round(4e6 / tgt.n_obs))))
bs = 512 if ref.n_obs > 50000 else 128  # larger minibatches for the 2 largest references
print(tag, A.split, "train", train.shape, "target", tgt.shape, "epochs", ep, "surgery", sep, "batch", bs, flush=True)

scvi.model.SCVI.setup_anndata(train, batch_key="batch", labels_key="label")
mU = scvi.model.SCVI(train, n_latent=A.n_latent)
mU.train(max_epochs=ep, batch_size=bs, enable_progress_bar=False)
mS = scvi.model.SCANVI.from_scvi_model(mU, unlabeled_category="Unknown", labels_key="label")
mS.train(max_epochs=max(20, ep // 10), batch_size=bs, enable_progress_bar=False)


def surgery(cls, model):
    q = tgt.copy()
    q.obs["label"] = "Unknown"
    cls.prepare_query_anndata(q, model)
    qm = cls.load_query_data(q, model)
    qm.train(max_epochs=sep, batch_size=bs, plan_kwargs=dict(weight_decay=0.0), enable_progress_bar=False)
    return qm, q


qU, _ = surgery(scvi.model.SCVI, mU)
qS, qa = surgery(scvi.model.SCANVI, mS)
prob = qS.predict(qa, soft=True)
np.savez_compressed(
    out,
    ref_U=mU.get_latent_representation().astype(np.float32),
    ref_S=mS.get_latent_representation().astype(np.float32),
    tgt_U=qU.get_latent_representation().astype(np.float32),
    tgt_S=qS.get_latent_representation().astype(np.float32),
    y_ref=train.obs.label.values.astype(str), y_tgt=tgt.obs.label.values.astype(str),
    parent_tgt=tgt.obs.parent.values.astype(str), parent_ref=train.obs.parent.values.astype(str),
    b_ref=train.obs.batch.values.astype(str), b_tgt=tgt.obs.batch.values.astype(str),
    cond_tgt=tgt.obs.condition.values.astype(str), tgt_names=tgt.obs_names.values.astype(str),
    scanvi_prob=prob.values.astype(np.float32), scanvi_classes=np.array(prob.columns, dtype=str),
)
print("saved", out, "minutes %.1f" % ((time.time() - t0) / 60), flush=True)
