# -----------------------------------------------------------------------------
# RHINO Campaign Workflow Configuration
#
# Central configuration file for RHINO campaign archive and index creation.
#
# This file defines:
#   - RHINO data locations
#   - campaign storage locations
#   - archive naming conventions
#   - date-based campaign grouping
#   - input directories containing ADIOS/openPMD outputs
#
# These parameters are separated from the workflow logic so that:
#   - campaign scripts remain reusable
#   - dataset locations can change without modifying workflow code
#   - multiple RHINO campaign configurations can be supported easily
#
# Used by:
#   create_archives.sh
#   create_index.sh
#
# RHINO / IFE_AmSC
# -----------------------------------------------------------------------------

RHINO_DATA_ROOT="/global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino"

CAMPAIGN_STORE="/global/homes/b/bhowmic/campaign-store/IFE"     #needs to be updated as per user's campaign store location

CAMPAIGN_NAMESPACE="IFE"

ARCHIVE_PREFIX="rhino"

CAMPAIGN_INDEX="rhino.acx"

# Uncompressed TAR files are written here so hpc_campaign can index their
# member offsets for direct access. TAR_STORAGE_HOST is the short, unique host
# name recorded in the campaign archive for this filesystem location.
TAR_OUTPUT_DIR="$RHINO_DATA_ROOT"
TAR_PREFIX="rhino"
TAR_STORAGE_SYSTEM="fs"
TAR_STORAGE_HOST="NERSC"

INPUT_DIRS=(
  "surrogate_bp_output/2026-04-29"
  "surrogate_bp_output/2026-04-30"
  "surrogate_bp_output/2026-05-01"
)
