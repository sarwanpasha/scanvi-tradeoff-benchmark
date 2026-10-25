"""scIB metrics for the unsupervised and supervised embeddings in one .npz file.

Usage: python scib_eval.py <embeddings.npz> <out.csv>
"""
import sys
import anndata as ad
import numpy as np
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

f, out = sys.argv[1], sys.argv[2]
d = np.load(f, allow_pickle=True)
lab = np.concatenate([d["ref_labels"], d["query_labels"]]).astype(str)
bat = np.concatenate([d["ref_batches"], d["query_batches_obs"]]).astype(str)
A = ad.AnnData(np.zeros((len(lab), 1), dtype=np.float32))
A.obs["lab"], A.obs["bat"] = pd.Categorical(lab), pd.Categorical(bat)
A.obsm["scVI"] = np.vstack([d["ref_scvi"], d["query_scvi"]]).astype(np.float32)
A.obsm["scANVI"] = np.vstack([d["ref_scanvi"], d["query_scanvi"]]).astype(np.float32)
bm = Benchmarker(A, batch_key="bat", label_key="lab", embedding_obsm_keys=["scVI", "scANVI"],
                 pre_integrated_embedding_obsm_key="scVI", bio_conservation_metrics=BioConservation(),
                 batch_correction_metrics=BatchCorrection(pcr_comparison=False), n_jobs=8)
bm.benchmark()
bm.get_results(min_max_scale=False).to_csv(out)
print("done", f, flush=True)
