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

TRAIN_MLFLOW_ARGS=()
TEST_MLFLOW_ARGS=()
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    DATASET_MANIFEST="${DATASET_MANIFEST:-$(dirname "${FEATURES}")/dataset_manifest.json}"
    if [[ ! -f "${DATASET_MANIFEST}" ]]; then
        echo "Dataset manifest not found: ${DATASET_MANIFEST}" >&2
        exit 2
    fi

    MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-rhino-surrogate}"
    MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-perlmutter-${SLURM_JOB_ID}}"
    TRAIN_MLFLOW_ARGS=(
        --mlflow
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}"
        --mlflow-experiment "${MLFLOW_EXPERIMENT}"
        --mlflow-run-name "${MLFLOW_RUN_NAME}"
        --dataset-manifest "${DATASET_MANIFEST}"
        --skip-model-registration
    )
    TEST_MLFLOW_ARGS=(
        --mlflow
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}"
    )
fi

srun python RHINO/ML/surrogate_training/trainSurrogate.py \
    --features "${FEATURES}" \
    --outdir "${RUN_DIR}" \
    --epochs "${EPOCHS:-200}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --hidden-dim "${HIDDEN_DIM:-100}" \
    --hidden-layers "${HIDDEN_LAYERS:-3}" \
    --lr "${LEARNING_RATE:-1e-4}" \
    --seed "${SEED:-42}" \
    "${TRAIN_MLFLOW_ARGS[@]}"

srun python RHINO/ML/surrogate_training/testSurrogate.py \
    --model "${RUN_DIR}/rhino_surrogate.pt" \
    --splits "${RUN_DIR}/split_indices.json" \
    --outdir "${RUN_DIR}" \
    "${TEST_MLFLOW_ARGS[@]}"

echo "Portable run bundle: ${RUN_DIR}"
