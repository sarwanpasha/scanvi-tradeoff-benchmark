#!/bin/bash
#SBATCH --job-name=scanvi_int
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=96g
#SBATCH --time=10:00:00
#SBATCH --output=logs/scanvi_int_%j.out
#SBATCH --error=logs/scanvi_int_%j.err

set -euo pipefail
python -c "import numpy; assert numpy.__version__.startswith('1.26'), \
           'NUMPY PIN BROKEN: '+numpy.__version__"

mkdir -p results logs

# Intervention 1: per-type harm predictor (leave-one-dataset-out)
echo "=== Intervention 1: per-type predictor $(date) ==="
python src/method_A_detector.py --emb_dir embeddings --out results/intervention_1.json

# Intervention 2: per-cell correction
echo "=== Intervention 2: per-cell correction $(date) ==="
python src/method_B_correction.py --emb_dir embeddings --out results/intervention_2.json

# Intervention 4: manifold alignment diagnostic (no training needed, uses saved embeddings)
echo "=== Intervention 4: manifold alignment $(date) ==="
python src/diagnose_alignment.py --emb_dir embeddings --out results/intervention_4.json

# Intervention 5: latent capacity scaling (RE-TRAINS at d=5,10,20,50; expensive)
echo "=== Intervention 5: capacity scaling $(date) ==="
python src/capacity_test.py --datasets spleen_lymph lymphoid pbmc_cite \
                            --dims 5 10 20 50 --seeds 0 1 2 \
                            --out results/intervention_5.json

# Intervention 6: supervised baselines + class-balanced scANVI
echo "=== Intervention 6: supervised baselines + balanced training $(date) ==="
python src/extra_methods.py --datasets pbmc_cite spleen_lymph lymphoid \
                            --methods sup_lda sup_ncm balanced_scanvi \
                            --seeds 0 1 2 --out results/intervention_6.json

echo "=== all interventions done $(date) ==="
