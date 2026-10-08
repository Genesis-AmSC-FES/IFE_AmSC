#!/usr/bin/env bash
# Laser HPC Campaign configuration. Every value can be overridden by exporting
# the same variable before invoking create_campaign.sh.

LASER_DATA_ROOT=${LASER_DATA_ROOT:-/global/cfs/cdirs/m3239/2026_FES-AmSC/data/laser}
BP5_INPUT_DIR=${BP5_INPUT_DIR:-$LASER_DATA_ROOT/bp_output}

CAMPAIGN_STORE=${CAMPAIGN_STORE:-/global/homes/b/bhowmic/campaign-store/IFE}
PUBLISH_CAMPAIGN_STORE=${PUBLISH_CAMPAIGN_STORE:-/global/cfs/cdirs/m3239/2026_FES-AmSC/campaigns/IFE}
ARCHIVE_PREFIX=${ARCHIVE_PREFIX:-laser}
CAMPAIGN_INDEX=${CAMPAIGN_INDEX:-laser.acx}

# Datasets are grouped chronologically. A new campaign archive begins whenever
# the next dataset date falls outside this many-day window. Set an optional
# positive MAX_DATASETS_PER_ARCHIVE to split a busy window into smaller parts;
# zero means that the date window alone controls grouping.
CAMPAIGN_WINDOW_DAYS=${CAMPAIGN_WINDOW_DAYS:-10}
MAX_DATASETS_PER_ARCHIVE=${MAX_DATASETS_PER_ARCHIVE:-0}

# Uncompressed TARs, their SHA-256 files, member manifests, and TAR indexes are
# written here. hpc_campaign records this location as the archival replica.
TAR_OUTPUT_DIR=${TAR_OUTPUT_DIR:-$LASER_DATA_ROOT/campaign_tar}
TAR_PREFIX=${TAR_PREFIX:-laser}
TAR_STORAGE_SYSTEM=${TAR_STORAGE_SYSTEM:-fs}
TAR_STORAGE_HOST=${TAR_STORAGE_HOST:-NERSC}
