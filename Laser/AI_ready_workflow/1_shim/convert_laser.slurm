#!/usr/bin/env bash
#SBATCH --account=m3239
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=laser-bp5
#SBATCH --output=laser-bp5-%j.out
#SBATCH --error=laser-bp5-%j.err

# Discover the complete inventory in one batch so HDF5 rollover segments with
# the same run number/date can be written into one openPMD Series.

set -uo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
LASER_DATA_ROOT=${LASER_DATA_ROOT:-/global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser}
LASER_BP_OUT_DIR=${LASER_BP_OUT_DIR:-$LASER_DATA_ROOT/bp_output}
LASER_CONDA_ENV=${LASER_CONDA_ENV:-IFE_AmSC}

# sbatch executes a copy of this file from /var/spool/slurmd, so BASH_SOURCE
# cannot be used to locate repository files. Resolve them from the directory
# where sbatch was invoked, while supporting both repository layouts in use.
if [[ -n "${LASER_SHIM_DIR:-}" ]]; then
    : # Respect an explicit override.
elif [[ -f "$SUBMIT_DIR/Laser/shim/laserWrite_multi.py" ]]; then
    LASER_SHIM_DIR="$SUBMIT_DIR/Laser/shim"
elif [[ -f "$SUBMIT_DIR/LASER/laserWrite_multi.py" ]]; then
    LASER_SHIM_DIR="$SUBMIT_DIR/LASER"
elif [[ -f "$SUBMIT_DIR/laserWrite_multi.py" ]]; then
    LASER_SHIM_DIR="$SUBMIT_DIR"
else
    LASER_SHIM_DIR="$SUBMIT_DIR/Laser/shim"
fi

if [[ ! -d "$LASER_DATA_ROOT" ]]; then
    echo "ERROR: LASER_DATA_ROOT does not exist: $LASER_DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$LASER_SHIM_DIR/laserWrite_multi.py" ]]; then
    echo "ERROR: laserWrite_multi.py not found under LASER_SHIM_DIR: $LASER_SHIM_DIR" >&2
    echo "Set LASER_SHIM_DIR explicitly with sbatch --export if the shim is elsewhere." >&2
    exit 1
fi

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LASER_CONDA_ENV"

mkdir -p "$LASER_BP_OUT_DIR"

echo "LASER source root : $LASER_DATA_ROOT"
echo "BP5 output        : $LASER_BP_OUT_DIR"
echo "Conda environment : $LASER_CONDA_ENV"

python "$LASER_SHIM_DIR/laserWrite_multi.py" \
    "$LASER_DATA_ROOT" \
    --out-dir "$LASER_BP_OUT_DIR" \
    --skip-existing
