#!/bin/bash
#SBATCH --job-name=scanvi_causal
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/scanvi_causal_%j.out
#SBATCH --error=logs/scanvi_causal_%j.err

set -euo pipefail
python -c "import numpy; assert numpy.__version__.startswith('1.26'), \
           'NUMPY PIN BROKEN: '+numpy.__version__"

mkdir -p results logs

# Causal mechanism dissection on mouse spleen:
#   - shuffle: shuffled reference labels (negative control for label-driven displacement)
#   - ablate_isg: selectively merge interferon-state label into parent
#   - ablate_control: same but with an unrelated control label
#   - dose_response: train at 10%, 25%, 50%, 75%, 100% reference label fractions
echo "=== causal controls $(date) ==="
python src/diagnose_theory.py --dataset spleen_lymph \
                              --controls shuffle ablate_isg ablate_control dose_response \
                              --out results/causal.json

# Three transfer tests for the dataset-specificity of the mechanism:
#   - cell_cycle: substitute cell-cycle label for ISG label
#   - derived_isg_pbmc: derive an ISG signature in resting human PBMC
#   - kang_native: Kang et al. interferon-beta stimulated PBMC
python src/diagnose_theory.py --transfer_tests cell_cycle derived_isg_pbmc kang_native \
                              --out results/causal_transfer.json

echo "=== done $(date) ==="
