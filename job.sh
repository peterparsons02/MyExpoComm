#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/itetnas04/data-scratch-01/pparsons/data/MyExpoComm/logs/%x_%j.out
#SBATCH --error=/usr/itetnas04/data-scratch-01/pparsons/data/MyExpoComm/logs/%x_%j.err
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=tikgpu10,tikgpu[06-09],tikgpu03

ALG_CONFIG=$1
ENV_CONFIG=$2
PROFILE=$3

if [ -z "$ALG_CONFIG" ] || [ -z "$ENV_CONFIG" ] || [ -z "$PROFILE" ]; then
  echo "ERROR: Missing arguments."
  echo "Usage: sbatch --job_name=<JOB_NAME> job_AdvPursuit.sh <ALG_CONFIG> <ENV_CONFIG> <PROFILE>"
  echo "Example: sbatch --job_name=advP_HybridComm_2M job_AdvPursuit.sh HybridComm MAgent_AdvPursuit advP_HybridComm_2M"
  exit 1
fi

# --- CONFIGURATION ---
# We use the absolute path resolved from your 'net_scratch' symlink
# This ensures SLURM can find it even if it doesn't load your home aliases
DIRECTORY=/usr/itetnas04/data-scratch-01/pparsons/data/MyExpoComm

# Name of your conda environment (as seen in your prompt)
CONDA_ENV_NAME=/usr/itetnas04/data-scratch-01/pparsons/data/conda_envs/ExpoComm

# Ensure log directory exists
mkdir -p ${DIRECTORY}/logs

# --- SETUP (Do not modify) ---
set -o errexit
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then echo 'Failed to create temp directory' >&2; exit 1; fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR
cd "${TMPDIR}" || exit 1

echo "JOB ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Project Directory: ${DIRECTORY}"
echo "--------------------------------"
echo "Algorithm:   ${ALG_CONFIG}"
echo "Environment: ${ENV_CONFIG}"
echo "Profile:     ${PROFILE}"
echo "--------------------------------"

source /usr/itetnas04/data-scratch-01/pparsons/data/conda/etc/profile.d/conda.sh

# Activate using the absolute path to your environment
conda activate /usr/itetnas04/data-scratch-01/pparsons/data/conda_envs/ExpoComm

echo "Conda activated: ${CONDA_DEFAULT_ENV}"

# Move to the code directory
cd ${DIRECTORY}

# --- EXECUTION ---
echo "Starting Run..."

# Added the extra print to help you verify reward settings in the logs
python src/main.py \
    --config=${ALG_CONFIG} \
    --env-config=${ENV_CONFIG} \
    --profile=${PROFILE}

echo "Finished at: $(date)"
exit 0
