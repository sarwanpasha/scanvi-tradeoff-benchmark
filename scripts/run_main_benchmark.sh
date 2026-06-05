#!/bin/bash
#SBATCH --job-name=scanvi_main
#SBATCH --partition=gpu                  # CHANGE to your GPU partition
#SBATCH --account=YOUR_ACCOUNT           # CHANGE to your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/scanvi_main_%j.out
#SBATCH --error=logs/scanvi_main_%j.err

# ============================================================================
# Main benchmark: train scVI + scANVI on five immune-tissue datasets, 3 seeds each.
# Pancreas is run separately via run_pancreas.sh because it requires a different
# HVG variant.
# ============================================================================

set -euo pipefail

# Activate your Python environment (edit for your setup)
# Example: source /path/to/venv/bin/activate
# or:      conda activate scanvi-tradeoff

# Guard the numpy pin (scvi-tools 1.4.x is incompatible with numpy 2.x)
python -c "import numpy; assert numpy.__version__.startswith('1.26'), \
           'NUMPY PIN BROKEN: '+numpy.__version__"

mkdir -p embeddings results logs

echo "=== main benchmark $(date) ==="

for ds in spleen_lymph lymphoid pbmc_cite pbmc heart; do
  for seed in 0 1 2; do
    echo "--- $ds seed=$seed $(date) ---"
    python -c "
import sys; sys.path.insert(0, 'src')
import cross_dataset_arm as c
c.run_once('$ds', seed=$seed, save_embeddings_dir='embeddings')
"
  done
done

# Aggregate composition-stratified retention across all (dataset, seed)
python -c "
import sys; sys.path.insert(0, 'src')
from aggregate_main_benchmark import aggregate
aggregate('embeddings', out='results/main_benchmark.json')
"

echo "=== done $(date) ==="
