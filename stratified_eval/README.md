# Abundance-stratified evaluation of label-supervised reference mapping

These scripts measure how label supervision changes kNN label-transfer accuracy for each cell type
when held-out query batches are mapped onto a labeled reference. Cell types are grouped into
abundance tertiles of the query (rare, intermediate, abundant), and effects are reported as
supervised minus unsupervised accuracy, with seed-level and donor-level uncertainty.

## Scripts

| Script | Purpose |
|---|---|
| `common.py` | Dataset splits (reference and query batches), loading, HVG selection, saving embeddings |
| `download_data.py` | Downloads the PBMC CITE-seq and spleen/lymph node CITE-seq datasets via scvi-tools |
| `train_scanvi_balanced.py` | scVI, scANVI, and class-balanced scANVI (median oversampling) with scArches query surgery |
| `train_scpoli.py` | scPoli with and without labels; `--hier` adds a coarse label level |
| `knn_eval.py` | Per-type kNN accuracy (k = 5, 15, 30, 50), strata, per-donor accuracy, top confusions |
| `centroids.py` | Reference centroid distances used by the neighborhood analysis |
| `summarize_strata.py` | Stratum means, k sweep, seed variation, lineage error split, neighborhood correlation |
| `donor_ci.py` | Donor-level effects with bootstrap 95% confidence intervals |
| `scib_eval.py`, `scib_summary.py` | scIB metrics for both embeddings and their differences |
| `summarize_variants.py` | Stratum effects for the balanced and scPoli variants |
| `run_slurm_array.sbatch` | Generic Slurm array template that runs one command line per array task |

All embedding files share one format: the unsupervised embedding is stored in the `ref_scvi` and
`query_scvi` arrays and the supervised embedding in `ref_scanvi` and `query_scanvi`, together with
labels and query batches. Every evaluation script therefore works for scANVI, class-balanced scANVI,
and scPoli alike.

## Example

```bash
pip install -r requirements.txt
mkdir -p data logs emb out
python download_data.py spleen_lymph
python train_scanvi_balanced.py --ds spleen_lymph --seed 0 --outdir emb
python train_scpoli.py --ds spleen_lymph --seed 0 --outdir emb
for f in emb/*.npz; do python knn_eval.py "$f" "out/$(basename "${f%.npz}").json"; done
python summarize_variants.py out
```

The tonsil data (`data/lymphoid_sc.h5ad`) must contain the columns `Tissue`, `Donor`, and `Subset_Broad`.
Run from this folder, or adjust the relative paths in `common.py`.
