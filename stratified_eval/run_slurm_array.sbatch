#!/bin/bash
# Generic Slurm array template. Each line of $TASKFILE holds the arguments of one run, e.g.
#   train_scpoli.py --ds lymphoid --seed 0 --hier
# Submit with:
#   sbatch --array=0-<N-1> --export=ALL,TASKFILE=tasks.txt run_slurm_array.sbatch
#SBATCH --job-name=stratified_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1 SCIPY_ARRAY_API=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKFILE}")
python3 ${LINE}
