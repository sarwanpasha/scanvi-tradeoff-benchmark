#!/bin/bash
#SBATCH --job-name=scanvi_pancreas
#SBATCH --partition=gpu                  # CHANGE to your GPU partition
#SBATCH --account=YOUR_ACCOUNT           # CHANGE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/scanvi_pancreas_%j.out
#SBATCH --error=logs/scanvi_pancreas_%j.err

set -euo pipefail

# Activate your environment (see run_main_benchmark.sh comment).
python -c "import numpy; assert numpy.__version__.startswith('1.26'), \
           'NUMPY PIN BROKEN: '+numpy.__version__"

mkdir -p embeddings results logs

# Pancreas is special: it uses cell_ranger HVG instead of seurat_v3 because
# the mixed droplet+plate-based technology composition (inDrop, CelSeq2,
# smartseq2, fluidigmc1) produces a 5-order-of-magnitude library size range
# that breaks seurat_v3's LOESS mean-variance fit. cross_dataset_arm.py
# detects dataset=='pancreas' and dispatches to cell_ranger automatically.

echo "=== pancreas $(date) ==="
python -c "
import sys; sys.path.insert(0, 'src')
import cross_dataset_arm as c
for seed in [0, 1, 2]:
    c.run_once('pancreas', seed=seed, save_embeddings_dir='embeddings')
"

# Compute per-stratum retention for pancreas
python -c "
import sys; sys.path.insert(0, 'src')
from aggregate_main_benchmark import per_dataset
per_dataset('embeddings', 'pancreas', out='results/pancreas_results.json')
"
echo "=== done $(date) ==="
