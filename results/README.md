# Results

All numbers reported in the paper are stored here as JSON for reproducibility and inspection.

| File | Description | Paper reference |
|------|-------------|-----------------|
| `revision_results.json` | Consolidated summary of all key statistics | (used to write the paper) |
| `k_sensitivity.json` | Per-stratum retention at k ∈ {5, 15, 30, 50} | Table 4 (Section 3.5) |
| `pancreas_results.json` | Per-seed per-stratum retention on pancreas | Table 3 (Section 3.4) |
| `intervention_6.json` | sup_lda, sup_ncm, balanced_scanvi raw outputs | Table 5 (Section 4.6) |
| `coupling_stats.json` | Pearson r, bootstrap CI, permutation p | Section 5 |
| `causal.json` | Shuffled-label, ablation, dose-response | Section 6 / Figs 4-5 |

Files are produced by the scripts in `../src/` and `../scripts/`. See the main README for the command sequence.
