"""Download the two CITE-seq datasets through scvi-tools (RNA counts are used)."""
import sys
import scvi

which = sys.argv[1]
if which == "pbmc_cite":
    a = scvi.data.pbmc_seurat_v4_cite_seq(save_path="data/")
elif which == "spleen_lymph":
    a = scvi.data.spleen_lymph_cite_seq(save_path="data/")
else:
    raise SystemExit("usage: python download_data.py {pbmc_cite|spleen_lymph}")
a.write_h5ad(f"data/{which}_full.h5ad")
print("saved", which, a.shape)
