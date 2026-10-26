# ARCANE

ARCANE (arbitration of complementary embeddings for annotation with error control) transfers cell-type labels from an annotated single-cell reference to a query by combining two views of the same cells: an unsupervised scVI embedding and a supervised scANVI embedding.

Using cells from reference batches that were held out during training (cross-fitting), ARCANE

1. arbitrates between the kNN labels of the two views with the rule that maximizes balanced accuracy, which does not depend on the cell-type composition of the query,
2. returns class-conditional (Mondrian) conformal prediction sets, and
3. calls cells unknown when they lie far from the reference in both views, using a scale-free kNN distance and a conformal threshold.

## Files

| File | Purpose |
|---|---|
| `arcane.py` | The method: kNN evidence, balanced arbitration, conformal sets, novelty score and unknown calls |
| `prep_data.py` | Writes standardized AnnData files (labels, batches, reference/query roles, cross-fitting folds) |
| `lupus_download.py`, `hlca_download.py` | Fetch the lupus and HLCA data from CELLxGENE Census |
| `train.py` | Trains scVI and scANVI on the full reference or on one fold and maps targets with scArches surgery |
| `celltypist_run.py` | CellTypist baseline |
| `evaluate.py` | Runs ARCANE and all baselines for one dataset and seed and writes per-type metrics |
| `aggregate.py` | Summary tables over datasets and seeds |
| `lupus_bio.py` | Relates unknown calls in lupus to an interferon-stimulated gene score |
| `make_tasks.py`, `run_all.sh` | Task lists and an end-to-end run |

## Usage

```python
from arcane import run_arcane
out = run_arcane(ref_U, ref_S, y_ref, tgt_U, tgt_S, cal, classes, k=15, alpha=0.1, alpha_unknown=0.05)
out["label"], out["sets"], out["unknown"], out["novelty"]
```

`cal` is a list with one dictionary per cross-fitting fold holding the embeddings of the training fold (`ref_U`, `ref_S`, `y_ref`) and of the held-out fold (`tgt_U`, `tgt_S`, `y_tgt`).

## Requirements

Python 3.10 or later, scvi-tools 1.3.3, scanpy, anndata, scikit-learn, numpy, pandas, scipy, celltypist, cellxgene-census.
