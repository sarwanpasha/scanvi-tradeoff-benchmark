# Preparing this repo for public release

This document is **for you** (the author). Delete it before pushing to GitHub.

The README, requirements, configs, SLURM scripts, and one source file (`src/k_sensitivity.py`) were prepared in this directory. The remaining source files — the actual training, intervention, and diagnostic logic — live on your cluster and need to be copied here with cluster-specific identifiers stripped.

## Step 1: Copy your cluster source files into src/

From your cluster, copy these files into `src/` of this repo. Filenames must match exactly because the SLURM scripts and README reference them:

```
src/pilot_leakage.py          # dataset loaders
src/cross_dataset_arm.py      # scVI/scANVI training + scArches mapping
src/extra_methods.py          # sup_lda, sup_ncm, balanced_scanvi
src/method_A_detector.py      # Intervention 1 (rename if your cluster file is "method_A_riskscore.py")
src/method_B_correction.py    # Intervention 2
src/diagnose_alignment.py     # Intervention 4
src/capacity_test.py          # Intervention 5
src/tradeoff_test.py          # coupling diagnostic
src/stats_tightening.py       # bootstrap CI + permutation p
src/diagnose_theory.py        # causal controls (shuffle/ablate/dose-response)
src/aggregate_main_benchmark.py   # NEW: aggregates per-stratum across (dataset, seed)
```

If `aggregate_main_benchmark.py` doesn't exist on the cluster, write a small wrapper that pools the per-stratum dictionaries already produced by your per-run scripts. The pattern is the same as `k_sensitivity.py`'s `per_stratum_retention` function.

## Step 2: Sanitize each file

For every `.py` file you copy, find-and-replace these cluster-specific strings:

| Find | Replace with |
|------|--------------|
| `/work/nvme/bhkb/Label_Leakage` | (remove; use relative paths) |
| `/u/sali4` | (remove; use relative or `~`) |
| `/home/sali4` | (remove) |
| `bhkb-delta-gpu` | (remove from SLURM if it appears) |
| `sali4@dt-login*` | (remove from comments) |
| `Label_Leakage` (as folder name) | rename to `scanvi-tradeoff` if used as a path prefix |
| Hardcoded data paths like `data/spleen.h5ad` | move to a config or accept `--data_dir` |

A one-shot scrub:

```bash
cd src/
for f in *.py; do
  sed -i 's|/work/nvme/bhkb/Label_Leakage/||g; \
          s|/u/sali4/||g; \
          s|/home/sali4/||g; \
          s|bhkb-delta-gpu|YOUR_ACCOUNT|g; \
          s|sali4|user|g' "$f"
done
grep -rn "sali4\|bhkb\|/work/nvme\|Label_Leakage" src/  # should print nothing
```

After scrubbing, run each file through `ast.parse` to confirm syntax wasn't broken by the substitution:

```bash
for f in src/*.py; do python -c "import ast; ast.parse(open('$f').read())" && echo "OK $f"; done
```

## Step 3: Check for accidental personally identifying material

Things that commonly leak through:

- Hardcoded NCSA Delta hostnames in comments (`dt-login01`, `gpub`, etc.)
- Git config remnants (`.git/config`, author email — gitignored, but double-check)
- Logbook or note files (you mentioned `RESULTS_LOGBOOK.md` and `journal.txt` in prior sessions — leave those out of the repo; they're for you, not the world)
- Email addresses in module headers
- Comments referencing institutional resources by name

```bash
grep -rni "ncsa\|delta\|illinois\|gpub\|@.*\.edu" src/ scripts/ configs/ README.md
```

## Step 4: Add a small `aggregate_main_benchmark.py` if you don't have one

This file is referenced by `scripts/run_main_benchmark.sh`. Minimal version:

```python
# src/aggregate_main_benchmark.py
"""Aggregate per-stratum retention across all (dataset, seed) embeddings."""
import argparse, glob, os, json, numpy as np
from k_sensitivity import load_embedding, knn_predict, per_stratum_retention

def aggregate(emb_dir, out, k=15):
    files = sorted(glob.glob(os.path.join(emb_dir, "*_seed*.npz")))
    files = [f for f in files if not any(tag in os.path.basename(f)
             for tag in ["_shuffled", "_withISG", "_ablISG", "_dim"])]
    agg = {"scVI": {s: [] for s in ["abundant", "intermediate", "rare"]},
           "scANVI": {s: [] for s in ["abundant", "intermediate", "rare"]}}
    for f in files:
        d = load_embedding(f)
        for meth, ref, q in [("scVI", d["ref_scvi"], d["q_scvi"]),
                             ("scANVI", d["ref_scanvi"], d["q_scanvi"])]:
            pred = knn_predict(ref, d["rl"], q, k=k)
            for s, v in per_stratum_retention(pred, d["ql"]).items():
                agg[meth][s].append(v)
    summary = {meth: {s: float(np.nanmean(v)) for s, v in d.items()}
               for meth, d in agg.items()}
    json.dump(summary, open(out, "w"), indent=2)
    print("saved ->", out, "\n", summary)

def per_dataset(emb_dir, dataset, out, k=15):
    """Like aggregate() but for a single dataset (used for pancreas)."""
    files = sorted(glob.glob(os.path.join(emb_dir, f"{dataset}_seed*.npz")))
    out_dict = {}
    for f in files:
        seed = os.path.basename(f).split("seed")[1].split(".")[0]
        d = load_embedding(f)
        out_dict[f"seed{seed}"] = {
            meth: per_stratum_retention(knn_predict(ref, d["rl"], q, k=k), d["ql"])
            for meth, ref, q in [("scVI", d["ref_scvi"], d["q_scvi"]),
                                 ("scANVI", d["ref_scanvi"], d["q_scanvi"])]
        }
    json.dump(out_dict, open(out, "w"), indent=2)
    print("saved ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="embeddings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    if args.dataset:
        per_dataset(args.emb_dir, args.dataset, args.out)
    else:
        aggregate(args.emb_dir, args.out)
```

Save that as `src/aggregate_main_benchmark.py`.

## Step 5: Copy results JSON files

```bash
# From your cluster, copy these to results/ of this repo:
results/main_benchmark.json        # Table 2
results/k_sensitivity.json         # Table 3
results/intervention_1.json
results/intervention_2.json
results/intervention_4.json
results/intervention_5.json
results/intervention_6.json
results/coupling_stats.json        # Section 5: r, CI, p
results/causal.json                # Section 6: shuffle, ablations, dose-response
results/pancreas_results.json      # pancreas
results/revision_results.json      # the consolidated summary (already in this dir)
```

## Step 6: Initialize git and push

```bash
cd scanvi-tradeoff-benchmark
rm PREPARE_REPO.md           # remove this internal file before pushing
git init
git add .
git commit -m "Initial public release"
gh repo create scanvi-tradeoff-benchmark --public --source=. --push
```

Or via the GitHub web UI: create the repo, then push.

## Step 7: Sanity-check that the repo runs on a fresh machine

If you have a colleague or a clean VM, do a five-minute test:

```bash
git clone https://github.com/YOUR_USERNAME/scanvi-tradeoff-benchmark.git
cd scanvi-tradeoff-benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "from src.k_sensitivity import per_stratum_retention; print('OK')"
```

If that imports cleanly, the public release is in shape.
