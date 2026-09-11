#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

# Submit from the repository root, after creating logs/.
# Use the cluster's existing ExpoComm environment; do not install dependencies.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source /usr/itetnas04/data-scratch-01/pparsons/data/conda/etc/profile.d/conda.sh
conda activate /usr/itetnas04/data-scratch-01/pparsons/data/conda_envs/ExpoComm
export OMP_NUM_THREADS=1
python -u src/tiger_deer_experiment.py "$@"
