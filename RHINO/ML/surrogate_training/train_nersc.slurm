#!/bin/bash
#SBATCH --job-name=rhino-train
#SBATCH --account=m3239
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --licenses=scratch,cfs
#SBATCH --output=rhino-%j.out
#SBATCH --error=rhino-%j.err

set -euo pipefail

if [[ -z "${FEATURES:-}" ]]; then
    echo "Set FEATURES to the immutable rhino_features.csv before submission." >&2
    echo "Example: sbatch --export=ALL,FEATURES=/global/cfs/.../rhino_features.csv $0" >&2
    exit 2
fi

module load python
conda activate IFE_AmSC

REPOSITORY="${SLURM_SUBMIT_DIR}"
RUN_ROOT="${RUN_ROOT:-${PSCRATCH}/rhino-runs}"
RUN_DIR="${RUN_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cd "${REPOSITORY}"

srun python RHINO/ML/surrogate_training/trainSurrogate.py \
    --features "${FEATURES}" \
    --outdir "${RUN_DIR}" \
    --epochs "${EPOCHS:-200}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --hidden-dim "${HIDDEN_DIM:-100}" \
    --hidden-layers "${HIDDEN_LAYERS:-3}" \
    --lr "${LEARNING_RATE:-1e-4}" \
    --seed "${SEED:-42}"

srun python RHINO/ML/surrogate_training/testSurrogate.py \
    --model "${RUN_DIR}/rhino_surrogate.pt" \
    --splits "${RUN_DIR}/split_indices.json" \
    --outdir "${RUN_DIR}"

echo "Portable run bundle: ${RUN_DIR}"
