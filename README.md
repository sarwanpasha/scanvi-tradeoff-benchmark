# scanvi-tradeoff-benchmark

Code and benchmark for the paper **"When Supervision Backfires: A Structural Rare-versus-Abundant Trade-off in Label-Supervised Single-Cell Integration"** (KDD 2026 Datasets and Benchmarks Track).

This repository releases:
- A cross-dataset transfer benchmark for evaluating supervised single-cell integration methods (six reference/query splits across mouse and human tissues),
- A composition-stratified evaluation suite that separates abundant, rare, and intermediate cell types under the scArches reference-mapping protocol,
- Six pre-registered intervention experiments testing removability of the abundant-type cost,
- Causal-control scripts (label shuffle, selective label ablation, supervision dose-response),
- A readout-robustness sweep over k-nearest-neighbour values.

## Contents

```
scanvi-tradeoff-benchmark/
├── README.md                       # this file
├── requirements.txt                # pinned Python dependencies (scvi-tools, numpy, scanpy)
├── environment.yml                 # conda alternative
├── src/
│   ├── pilot_leakage.py            # dataset loaders (5 immune-tissue datasets + pancreas)
│   ├── cross_dataset_arm.py        # scVI / scANVI training + scArches reference mapping
│   ├── extra_methods.py            # supervised baselines: sup_lda, sup_ncm, balanced_scanvi
│   ├── method_A_detector.py        # Intervention 1: per-type risk prediction
│   ├── method_B_correction.py      # Intervention 2: per-cell directional correction
│   ├── diagnose_alignment.py       # Intervention 4: query-side manifold alignment
│   ├── capacity_test.py            # Intervention 5: latent capacity scaling
│   ├── tradeoff_test.py            # coupling diagnostic (Section 5: cost is dual of benefit)
│   ├── stats_tightening.py         # bootstrap CI + permutation test for coupling
│   ├── diagnose_theory.py          # causal mechanism dissection (interferon-state ablation)
│   └── k_sensitivity.py            # readout robustness over k ∈ {5, 15, 30, 50}
├── scripts/
│   ├── run_main_benchmark.sh       # SLURM template: trains scVI+scANVI on all datasets
│   ├── run_interventions.sh        # SLURM template: runs all six interventions
│   ├── run_causal_controls.sh      # SLURM template: shuffled labels + ablation + dose-response
│   └── run_pancreas.sh             # SLURM template: pancreas (uses cell_ranger HVG variant)
├── configs/
│   └── datasets.yaml               # dataset URLs, label_keys, batch_keys, citations
├── results/
│   ├── revision_results.json       # all numerical results reported in the paper
│   └── k_sensitivity.json          # k-sweep raw output
└── figures/
    └── (regeneration scripts for paper figures; see Reproducing the Paper below)
```

## Quick start

### 1. Environment

We pin numpy to 1.26.x because scvi-tools 1.4.x is incompatible with numpy 2.x. We also pin scikit-learn and statsmodels because the cell_ranger HVG variant used for the pancreas dataset requires statsmodels.

```bash
# Option A: pip + a fresh virtualenv (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate scanvi-tradeoff
```

Verify the install:

```bash
python -c "import scvi, numpy, sklearn, scanpy, statsmodels; \
           print(f'scvi {scvi.__version__}, numpy {numpy.__version__}, \
                   sklearn {sklearn.__version__}, scanpy {scanpy.__version__}, \
                   statsmodels {statsmodels.__version__}')"
```

You should see numpy 1.26.x, scvi 1.4.x, sklearn 1.3.x, scanpy 1.11.x.

### 2. Data

All datasets are public. The loaders in `src/pilot_leakage.py` download them on first use to a local `data/` directory. Sources and citations are listed in `configs/datasets.yaml`. Some scIB-hosted datasets (notably pancreas) sit behind AWS WAF challenges that may block automated `curl`/`wget` downloads from headless servers; in that case, download the file in a browser and place it in `data/` manually (the loader looks for files by name and skips the download if present).

| Dataset | Source | Cells | Cell types |
|---------|--------|------:|------:|
| Spleen-LN (mouse) | scIB | 30,474 | 27 |
| Lymphoid (tonsil) | HCA | 29,705 | 35 |
| PBMC (fine) | CITE-seq | 152,094 | 30 |
| PBMC (coarse) | 10x | 11,990 | 9 |
| Heart | HCA | 18,641 | 11 |
| Pancreas | scIB | 16,175 | 13 |

### 3. Reproducing the main benchmark

The main result (Table 2 in the paper) is the composition-stratified retention pattern. To reproduce it on a single GPU:

```bash
mkdir -p embeddings results
for ds in spleen_lymph lymphoid pbmc_cite pbmc heart; do
  for seed in 0 1 2; do
    python -c "import cross_dataset_arm as c; c.run_once('$ds', seed=$seed, save_embeddings_dir='embeddings')"
  done
done

# Aggregate per-stratum retention
python src/aggregate_main_benchmark.py --emb_dir embeddings --out results/main_benchmark.json
```

Expected wall time: about 30 minutes per (dataset, seed) on an A100 — roughly 7-8 hours for the full benchmark across 5 immune tissues × 3 seeds. The pancreas run is configured separately because it requires a different highly-variable-gene method (`cell_ranger` instead of `seurat_v3`) due to the dataset's mixed droplet/plate-based technology composition; see `scripts/run_pancreas.sh`.

### 4. Reproducing the six interventions

```bash
# Interventions 1-2: per-type prediction, per-cell correction
python src/method_A_detector.py    --emb_dir embeddings --out results/intervention_1.json
python src/method_B_correction.py  --emb_dir embeddings --out results/intervention_2.json

# Intervention 3: readout replacement (run within extra_methods.py)
# Intervention 4: manifold alignment
python src/diagnose_alignment.py   --emb_dir embeddings --out results/intervention_4.json

# Intervention 5: latent capacity scaling (re-trains at d ∈ {5, 10, 20, 50})
python src/capacity_test.py --datasets spleen_lymph lymphoid pbmc_cite \
                            --dims 5 10 20 50 --seeds 0 1 2 \
                            --out results/intervention_5.json

# Intervention 6: class-balanced training + supervised baselines
python src/extra_methods.py --datasets pbmc_cite spleen_lymph lymphoid \
                            --methods sup_lda sup_ncm balanced_scanvi \
                            --seeds 0 1 2 --out results/intervention_6.json
```

### 5. Reproducing the coupling diagnostic (Section 5)

```bash
# Per-type harm vs neighbourhood rare-type gain; bootstrap CI + permutation p
python src/tradeoff_test.py    --emb_dir embeddings --out results/coupling_raw.json
python src/stats_tightening.py --in results/coupling_raw.json --out results/coupling_stats.json
```

Expected numbers: Pearson r=0.55, 95% bootstrap CI [0.17, 0.79], permutation p=0.011 (n=21 type instances).

### 6. Reproducing the causal dissection (Section 6)

```bash
# Mouse spleen: shuffled labels, selective ablation, dose-response
python src/diagnose_theory.py --dataset spleen_lymph \
                              --controls shuffle ablate_isg ablate_control dose_response \
                              --out results/causal.json
```

### 7. Readout-robustness sweep (Section 3.5)

```bash
python src/k_sensitivity.py --emb_dir embeddings --ks 5 15 30 50 \
                            --out results/k_sensitivity.json
```

### 8. Adding a new method to the benchmark

The benchmark is designed to be extensible. To evaluate a new label-supervised integration method, implement a function in `src/extra_methods.py` with this signature:

```python
def run_my_method(adata_ref, adata_query, label_key, batch_key, seed):
    """
    adata_ref: AnnData with labels (obs[label_key]) and batches (obs[batch_key])
    adata_query: AnnData without labels; will be label-transferred
    Returns: (ref_latent: np.ndarray, query_latent: np.ndarray)
    """
    # ... your training + reference-mapping logic ...
    return ref_latent, query_latent
```

Then add it to the method dispatch at the top of `extra_methods.py` and re-run the per-stratum retention reporting on its embeddings.

## Computational requirements

- **GPU**: A single A100 (40 or 80 GB) for the main benchmark and the GPU-bound interventions. V100 also works with slightly longer wall times.
- **CPU**: 16 cores, 96 GB RAM recommended for PBMC-fine (152k cells). All other datasets fit in 64 GB.
- **Storage**: ~10 GB for input datasets, ~2 GB for saved embeddings.
- **Total runtime**: ~10 GPU-hours for the full main benchmark + interventions, plus ~2 CPU-hours for diagnostics.

## Reproducibility notes

- All experiments use three random seeds (0, 1, 2). Per-seed and per-dataset variance is reported in `results/revision_results.json`.
- The benchmark protocol holds out whole batches (donors or technical batches), not cells, to ensure transfer is genuinely out-of-distribution.
- The supervised contrastive learning comparison in Section 5 is *qualitative*; we do not implement a contrastive integration method ourselves but reference Khosla et al. (NeurIPS 2020) and Graf et al. (ICML 2021).
- Figures in the paper are regenerated from the JSON files in `results/` via the scripts in `figures/`. The Figure 3 scatter plot is illustrative at the measured correlation; the actual point coordinates are saved in `results/coupling_raw.json`.

## Known limitations of this implementation

- The pancreas dataset is preprocessed differently from the immune-tissue datasets (cell_ranger HVG instead of seurat_v3 HVG) because mixed droplet/plate-based sequencing technologies produce a multimodal library-size distribution that breaks the seurat_v3 LOESS mean-variance fit. This is documented inline in `cross_dataset_arm.py` and is consistent with the standard scIB/openproblems convention for this dataset.
- scVI/scANVI training is non-deterministic in some respects even with fixed seeds (PyTorch CUDA kernels are not bit-reproducible across hardware). Reported numbers should reproduce to ±0.01 retention on the same hardware family.
- The class-balanced scANVI implementation (Intervention 6) uses inverse-frequency resampling of the reference for training and the original-frequency reference for evaluation. We do not modify scANVI's internal loss; the rebalancing is done at the data level.

## Citation

If you use this benchmark, please cite:

```bibtex
@inproceedings{authors2026supervision,
  title={When Supervision Backfires: A Structural Rare-versus-Abundant Trade-off in Label-Supervised Single-Cell Integration},
  author={...},
  booktitle={Proceedings of the 32nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (Datasets and Benchmarks Track)},
  year={2026}
}
```

Please also cite the underlying datasets and methods (scVI, scANVI, scArches, scIB) as appropriate; citations are listed in the paper.

## License

Code: MIT License (see LICENSE).
Datasets: please honour the licenses of the underlying public sources, listed in `configs/datasets.yaml`.

## Contact

Issues and questions: please open a GitHub issue on this repository. For correspondence not appropriate for a public issue tracker, see the paper for author email.
