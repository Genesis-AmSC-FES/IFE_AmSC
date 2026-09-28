#!/usr/bin/env bash

# Create RHINO TAR files, campaign archives, and TAR-backed replicas.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/config_campaign.sh"

dry_run=false
rebuild_tars=false

usage() {
    cat <<'EOF'
Usage: bash create_archives.sh [--dry-run] [--rebuild-tars]

  --dry-run       Print commands without creating or changing files.
  --rebuild-tars  Replace TAR, checksum, and TAR index files.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            dry_run=true
            ;;
        --rebuild-tars)
            rebuild_tars=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

run_cmd() {
    printf '  Command :'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$dry_run" == false ]]; then
        "$@"
    fi
}

if [[ ! -d "$RHINO_DATA_ROOT" ]]; then
    echo "ERROR: RHINO data root does not exist: $RHINO_DATA_ROOT" >&2
    exit 1
fi
if (( ${#INPUT_DIRS[@]} == 0 )); then
    echo "ERROR: INPUT_DIRS must contain at least one directory." >&2
    exit 1
fi

cd "$RHINO_DATA_ROOT"
data_root_abs=$(realpath "$RHINO_DATA_ROOT")
tar_output_abs=$(realpath -m "$TAR_OUTPUT_DIR")

echo "--------------------------------------------------"
echo "RHINO Campaign Archive Creation"
echo "--------------------------------------------------"
echo "Data root          : $RHINO_DATA_ROOT"
echo "Campaign store     : $CAMPAIGN_STORE"
echo "Archive prefix     : $ARCHIVE_PREFIX"
echo "TAR output         : $TAR_OUTPUT_DIR"
echo "TAR prefix         : $TAR_PREFIX"
echo "TAR storage        : $TAR_STORAGE_SYSTEM / $TAR_STORAGE_HOST"
echo "--------------------------------------------------"

shopt -s nullglob
files=()
for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ "$input_dir" = /* || "$input_dir" == ".." || "$input_dir" == ../* \
        || "$input_dir" == */../* || "$input_dir" == */.. ]]; then
        echo "ERROR: INPUT_DIRS entries must stay under RHINO_DATA_ROOT: $input_dir" >&2
        exit 1
    fi
    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: BP5 input directory does not exist: $RHINO_DATA_ROOT/$input_dir" >&2
        exit 1
    fi
    input_abs=$(realpath "$input_dir")
    if [[ "$input_abs/" != "$data_root_abs/"* ]]; then
        echo "ERROR: INPUT_DIRS entry escapes RHINO_DATA_ROOT: $input_dir" >&2
        exit 1
    fi
    if [[ "$tar_output_abs/" == "$input_abs/"* ]]; then
        echo "ERROR: TAR_OUTPUT_DIR cannot be inside an input directory: $input_dir" >&2
        exit 1
    fi
    echo "Scanning directory: $RHINO_DATA_ROOT/$input_dir"
    datasets=("$input_dir"/*.bp5)
    files+=("${datasets[@]}")
done
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
    echo "ERROR: No .bp5 datasets were discovered." >&2
    exit 1
fi

mapfile -t files < <(printf '%s\n' "${files[@]}" | sort)
declare -A dated_input_dirs
for dataset in "${files[@]}"; do
    dated_input_dirs["$(dirname "$dataset")"]=1
done
estimated_archives=${#dated_input_dirs[@]}
echo "Discovered ${#files[@]} dataset(s)."
echo "Estimated archive count: $estimated_archives"

if [[ "$dry_run" == false ]]; then
    mkdir -p "$TAR_OUTPUT_DIR"
fi

declare -A tar_by_dir
declare -A used_tar_paths

# Create or verify one uncompressed TAR and TAR index per input directory.
for input_dir in "${INPUT_DIRS[@]}"; do
    date_name=$(basename "$input_dir")
    tar_path="$TAR_OUTPUT_DIR/${TAR_PREFIX}-${date_name}.tar"
    checksum_path="${tar_path}.sha256"
    tar_index_path="${tar_path}.idx"

    if [[ -n "${used_tar_paths[$tar_path]:-}" ]]; then
        echo "ERROR: Multiple INPUT_DIRS produce the same TAR name: $tar_path" >&2
        exit 1
    fi
    used_tar_paths["$tar_path"]=1
    tar_by_dir["$input_dir"]="$tar_path"

    echo
    echo "Preparing TAR for $input_dir:"
    create_tar=false
    if [[ "$rebuild_tars" == true || ! -f "$tar_path" ]]; then
        create_tar=true
        run_cmd tar -cf "$tar_path" "$input_dir"
        echo "  Checksum: $checksum_path"
        if [[ "$dry_run" == false ]]; then
            (
                cd "$TAR_OUTPUT_DIR"
                sha256sum "$(basename "$tar_path")" > "$(basename "$checksum_path")"
            )
        fi
    else
        echo "  Reusing  : $tar_path"
        if [[ ! -f "$checksum_path" ]]; then
            echo "ERROR: Existing TAR has no checksum: $checksum_path" >&2
            echo "Use --rebuild-tars to replace it." >&2
            exit 1
        fi
        if [[ "$dry_run" == true ]]; then
            echo "  Verify   : $checksum_path"
        else
            (
                cd "$TAR_OUTPUT_DIR"
                sha256sum --check "$(basename "$checksum_path")"
            )
        fi
    fi

    if [[ "$create_tar" == true || ! -f "$tar_index_path" \
        || "$tar_path" -nt "$tar_index_path" ]]; then
        run_cmd hpc_campaign taridx "$tar_path" "$tar_index_path"
    else
        echo "  Reusing  : $tar_index_path"
    fi
done

declare -A archive_started

# Add live BP5 datasets to one dated campaign archive per input directory.
for dataset in "${files[@]}"; do
    input_dir=$(dirname "$dataset")
    date_name=$(basename "$input_dir")
    archive="${ARCHIVE_PREFIX}-${date_name}.aca"
    name=$(basename "$dataset" .bp5)
    run_id=$(grep -oP '\d{2}-\d{2}-\d{2}.*' <<< "$name" || true)
    if [[ -z "$run_id" ]]; then
        run_id="$name"
    fi

    echo
    echo "Adding dataset:"
    echo "  File    : $dataset"
    echo "  Run ID  : $run_id"
    echo "  Archive : $archive"

    command=(
        hpc_campaign manager
        --campaign_store "$CAMPAIGN_STORE"
        "$archive"
    )
    if [[ -z "${archive_started[$archive]:-}" ]]; then
        command+=(--truncate)
        archive_started["$archive"]=1
    fi
    command+=(data "$dataset" --name "$run_id")
    run_cmd "${command[@]}"
done

# Supplying the TAR index makes hpc_campaign attach every matching BP5 replica.
for input_dir in "${INPUT_DIRS[@]}"; do
    date_name=$(basename "$input_dir")
    archive="${ARCHIVE_PREFIX}-${date_name}.aca"
    if [[ -z "${archive_started[$archive]:-}" ]]; then
        continue
    fi
    tar_path="${tar_by_dir[$input_dir]}"
    tar_index_path="${tar_path}.idx"

    echo
    echo "Registering TAR replicas:"
    echo "  Archive : $archive"
    echo "  TAR     : $tar_path"
    run_cmd hpc_campaign manager \
        --campaign_store "$CAMPAIGN_STORE" \
        "$archive" \
        add-archival-storage \
        "$TAR_STORAGE_SYSTEM" \
        "$TAR_STORAGE_HOST" \
        "$TAR_OUTPUT_DIR" \
        "$(basename "$tar_path")" \
        "$tar_index_path"
done

echo
echo "--------------------------------------------------"
echo "Campaign archive creation complete."
echo "Campaign archives : $estimated_archives"
echo "TAR files         : ${#INPUT_DIRS[@]}"
echo "Grouping          : one campaign archive per date"
echo "--------------------------------------------------"
