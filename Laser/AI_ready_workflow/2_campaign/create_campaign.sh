#!/usr/bin/env bash
# Build date-windowed Laser campaign archives and the .acx index.
# TAR creation and archival-replica registration are temporarily disabled.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/config_campaign.sh"

dry_run=false
rebuild_tars=false
window_days="$CAMPAIGN_WINDOW_DAYS"
max_datasets="$MAX_DATASETS_PER_ARCHIVE"

usage() {
    cat <<'EOF'
Usage: bash create_campaign.sh [options]

Options:
  --dry-run                 Print and validate the complete plan only.
  --rebuild-tars            Reserved; TAR creation is currently disabled.
  --window-days N           Start a new campaign every N calendar days.
  --max-datasets N          Optional maximum datasets per campaign (0=unlimited).
  -h, --help                Show this help text.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run) dry_run=true ;;
        --rebuild-tars) rebuild_tars=true ;;
        --window-days)
            shift
            (( $# > 0 )) || { echo "ERROR: --window-days needs a value" >&2; exit 2; }
            window_days="$1"
            ;;
        --max-datasets)
            shift
            (( $# > 0 )) || { echo "ERROR: --max-datasets needs a value" >&2; exit 2; }
            max_datasets="$1"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ "$window_days" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: window days must be a positive integer" >&2
    exit 2
}
[[ "$max_datasets" =~ ^[0-9]+$ ]] || {
    echo "ERROR: max datasets must be zero or a positive integer" >&2
    exit 2
}
[[ -d "$LASER_DATA_ROOT" ]] || {
    echo "ERROR: LASER_DATA_ROOT does not exist: $LASER_DATA_ROOT" >&2
    exit 1
}
[[ -d "$BP5_INPUT_DIR" ]] || {
    echo "ERROR: BP5_INPUT_DIR does not exist: $BP5_INPUT_DIR" >&2
    exit 1
}
[[ "$CAMPAIGN_INDEX" == *.acx && "$CAMPAIGN_INDEX" != */* ]] || {
    echo "ERROR: CAMPAIGN_INDEX must be a filename ending in .acx" >&2
    exit 2
}

if [[ "$dry_run" == false ]]; then
    command -v hpc_campaign >/dev/null || {
        echo "ERROR: hpc_campaign is not available on PATH" >&2
        exit 1
    }
fi

archive_command=(
    python "$SCRIPT_DIR/create_archives.py"
    --data-root "$LASER_DATA_ROOT"
    --bp5-input-dir "$BP5_INPUT_DIR"
    --campaign-store "$CAMPAIGN_STORE"
    --tar-output-dir "$TAR_OUTPUT_DIR"
    --archive-prefix "$ARCHIVE_PREFIX"
    --tar-prefix "$TAR_PREFIX"
    --window-days "$window_days"
    --max-datasets "$max_datasets"
    --storage-system "$TAR_STORAGE_SYSTEM"
    --storage-host "$TAR_STORAGE_HOST"
)
[[ "$rebuild_tars" == true ]] && archive_command+=(--rebuild-tars)
[[ "$dry_run" == true ]] && archive_command+=(--dry-run)

printf 'Archive command:'
printf ' %q' "${archive_command[@]}"
printf '
'
"${archive_command[@]}"

index_path="$CAMPAIGN_STORE/$CAMPAIGN_INDEX"
index_command=(
    python "$SCRIPT_DIR/create_index.py"
    --campaign-store "$CAMPAIGN_STORE"
    --archive-prefix "$ARCHIVE_PREFIX"
    --index "$CAMPAIGN_INDEX"
)

if [[ "$dry_run" == true ]]; then
    printf 'Index command:'
    printf ' %q' "${index_command[@]}"
    printf '
'
else
    "${index_command[@]}"
fi

echo
echo "Laser campaign workflow complete."
echo "Campaign store : $CAMPAIGN_STORE"
echo "Shared store   : $PUBLISH_CAMPAIGN_STORE (publish separately)"
echo "TAR replicas   : disabled temporarily"
echo "Index          : $index_path"
