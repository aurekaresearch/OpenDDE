#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DATA_BASE_URL="https://aureka-s3-opendde.s3.us-west-2.amazonaws.com"
DEFAULT_SEARCH_DATABASE_URL="https://storage.googleapis.com/alphafold-databases/v3.0"

COMMON_FILES=(
    "components.cif"
    "components.cif.rdkit_mol.pkl"
    "release_date_cache.json"
    "obsolete_to_successor.json"
)

SEARCH_DATABASE_FILES=(
    "pdb_seqres_2022_09_28.fasta.zst"
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst"
)

OPENDDE_ROOT="${OPENDDE_ROOT_DIR:-}"
DATA_BASE_URL="${OPENDDE_DATA_BASE_URL:-$DEFAULT_DATA_BASE_URL}"
DEPENDENCY_URL="${OPENDDE_DEPENDENCY_URL:-}"
COMMON_URL="${OPENDDE_COMMON_URL:-}"
COMMON_URL_EXPLICIT=0
if [[ -n "$COMMON_URL" ]]; then
    COMMON_URL_EXPLICIT=1
elif [[ -n "${OPENDDE_DEPENDENCY_URL:-}" ]]; then
    COMMON_URL="$OPENDDE_DEPENDENCY_URL"
fi
SEARCH_DATABASE_URL="${OPENDDE_SEARCH_DATABASE_URL:-$DEFAULT_SEARCH_DATABASE_URL}"
MODEL_NAME="${OPENDDE_MODEL_NAME:-}"
MODEL_SOURCE="${OPENDDE_MODEL_SOURCE:-${OPENDDE_MODEL_URL:-}}"
CHECKPOINT_NAME="${OPENDDE_CHECKPOINT:-}"
MODEL_MANIFEST_PATH="${OPENDDE_MODEL_MANIFEST:-${REPO_ROOT}/opendde/config/model_manifest.json}"
MODEL_MANIFEST_TOOL="${REPO_ROOT}/opendde/config/model_manifest.py"
if [[ -n "${OPENDDE_PYTHON:-}" ]]; then
    PYTHON_COMMAND="$OPENDDE_PYTHON"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_COMMAND="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_COMMAND="python3"
fi
DOWNLOAD_COMMON=1
DOWNLOAD_SEARCH_DATABASE=1
DOWNLOAD_MODEL=1
VERIFY_RELEASED_MODEL=1
FORCE=0

info() { printf '\033[1;34m[INFO]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$1"; }
err() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }

TEMP_FILES=()
cleanup_temp_files() {
    if [[ "${#TEMP_FILES[@]}" -gt 0 ]]; then
        rm -f -- "${TEMP_FILES[@]}"
    fi
}
trap cleanup_temp_files EXIT

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Download the files needed by the inference-only OpenDDE runtime into
OPENDDE_ROOT_DIR:

  common/          CCD/cache metadata used by JSON parsing and featurization
  search_database/ Template and RNA-MSA search databases used by prep/mt
  checkpoint/      Model checkpoint used by opendde pred

Options:
  --root DIR                 Data root. Defaults to OPENDDE_ROOT_DIR.
  --base-url URL             Bulk tarball root used only as a search database
                             fallback. Defaults to OPENDDE_DATA_BASE_URL or
                             ${DEFAULT_DATA_BASE_URL}.
  --dependency-url URL       Root for model checkpoint files. Defaults to
                             OPENDDE_DEPENDENCY_URL or the manifest source.
                             Also used for common files unless --common-url is set.
  --common-url URL           Root for common runtime files. Defaults to
                             OPENDDE_COMMON_URL, OPENDDE_DEPENDENCY_URL if set,
                             otherwise <manifest-source>/common.
  --search-database-url URL  Root for hmmsearch/nhmmer search databases.
                             Defaults to OPENDDE_SEARCH_DATABASE_URL or
                             ${DEFAULT_SEARCH_DATABASE_URL}.
  --model-name NAME          Model name. Defaults to manifest.default_model; unknown
                             names retain the legacy <dependency-url>/<name>.pt lookup.
  --checkpoint FILE          Released checkpoint filename from the manifest.
                             Overrides the selected model's default checkpoint.
  --model-source PATH_OR_URL Model source file/URL. Defaults to OPENDDE_MODEL_SOURCE,
                             OPENDDE_MODEL_URL, otherwise the manifest checkpoint URL
                             for a released model or <dependency-url>/<model-name>.pt.
  --model-manifest PATH      Manifest used to verify released checkpoints. Defaults
                             to OPENDDE_MODEL_MANIFEST or ${MODEL_MANIFEST_PATH}.
  --skip-model-verification Allow a custom file named like an official checkpoint
                             without checking its released size and SHA-256.
  --skip-common              Do not download common runtime files.
  --skip-search-database     Do not download search database files.
  --skip-model               Do not install the model checkpoint.
  --force                    Re-download/re-copy even if target files exist.
  --inference_only           Kept for backward compatibility; this is the default.
  -h, --help                 Show this help message.

Examples:
  OPENDDE_ROOT_DIR=/data/opendde $0
  $0 --root /data/opendde --skip-search-database --skip-model
  $0 --root /data/opendde --model-source /path/to/opendde.pt
  $0 --root /data/opendde --model-source https://example.com/opendde_abag.pt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            OPENDDE_ROOT="${2:-}"
            shift 2
            ;;
        --base-url)
            DATA_BASE_URL="${2:-}"
            shift 2
            ;;
        --dependency-url)
            DEPENDENCY_URL="${2:-}"
            if [[ "$COMMON_URL_EXPLICIT" -eq 0 ]]; then
                COMMON_URL="$DEPENDENCY_URL"
            fi
            shift 2
            ;;
        --common-url)
            COMMON_URL="${2:-}"
            COMMON_URL_EXPLICIT=1
            shift 2
            ;;
        --search-database-url)
            SEARCH_DATABASE_URL="${2:-}"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="${2:-}"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_NAME="${2:-}"
            shift 2
            ;;
        --model-source|--model-url)
            MODEL_SOURCE="${2:-}"
            shift 2
            ;;
        --model-manifest)
            MODEL_MANIFEST_PATH="${2:-}"
            shift 2
            ;;
        --skip-model-verification)
            VERIFY_RELEASED_MODEL=0
            shift
            ;;
        --skip-common)
            DOWNLOAD_COMMON=0
            shift
            ;;
        --skip-search-database)
            DOWNLOAD_SEARCH_DATABASE=0
            shift
            ;;
        --skip-model)
            DOWNLOAD_MODEL=0
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --inference_only)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            ;;
    esac
done

[[ -n "$OPENDDE_ROOT" ]] || err "OPENDDE_ROOT_DIR is not set. Use --root DIR or export OPENDDE_ROOT_DIR=/path/to/data_root."
if [[ "$DOWNLOAD_SEARCH_DATABASE" -eq 1 ]]; then
    [[ -n "$DATA_BASE_URL" ]] || err "Data base URL must not be empty."
    [[ -n "$SEARCH_DATABASE_URL" ]] || err "Search database URL must not be empty."
fi

NEEDS_MODEL_MANIFEST=0
if [[ "$DOWNLOAD_MODEL" -eq 1 ]] \
    || [[ "$DOWNLOAD_COMMON" -eq 1 \
        && "$COMMON_URL_EXPLICIT" -eq 0 \
        && -z "$COMMON_URL" ]]; then
    NEEDS_MODEL_MANIFEST=1
fi

if [[ "$NEEDS_MODEL_MANIFEST" -eq 1 ]]; then
    [[ -f "$MODEL_MANIFEST_PATH" ]] \
        || err "Model manifest is not a regular file: $MODEL_MANIFEST_PATH"
    [[ -f "$MODEL_MANIFEST_TOOL" ]] \
        || err "Model manifest tool is not a regular file: $MODEL_MANIFEST_TOOL"
    command -v "$PYTHON_COMMAND" >/dev/null 2>&1 \
        || err "Python command is unavailable: $PYTHON_COMMAND"

    if ! MANIFEST_DEFAULTS="$(
        "$PYTHON_COMMAND" "$MODEL_MANIFEST_TOOL" resolve-model \
            --manifest "$MODEL_MANIFEST_PATH"
    )"; then
        err "Unable to resolve defaults from model manifest: $MODEL_MANIFEST_PATH"
    fi
    IFS=$'\t' read -r \
        MANIFEST_DEFAULT_MODEL \
        _MANIFEST_DEFAULT_CHECKPOINT \
        _MANIFEST_DEFAULT_CHECKPOINT_URL \
        MANIFEST_SOURCE_ROOT <<< "$MANIFEST_DEFAULTS"

    MODEL_NAME="${MODEL_NAME:-$MANIFEST_DEFAULT_MODEL}"
    DEPENDENCY_URL="${DEPENDENCY_URL:-$MANIFEST_SOURCE_ROOT}"
    COMMON_URL="${COMMON_URL:-${MANIFEST_SOURCE_ROOT}/common}"
fi

if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
    [[ -n "$MODEL_NAME" ]] || err "Model name must not be empty."
    [[ -n "$DEPENDENCY_URL" ]] || err "Dependency URL must not be empty."
fi
if [[ "$DOWNLOAD_COMMON" -eq 1 ]]; then
    [[ -n "$COMMON_URL" ]] || err "Common URL must not be empty."
fi

DATA_BASE_URL="${DATA_BASE_URL%/}"
DEPENDENCY_URL="${DEPENDENCY_URL%/}"
COMMON_URL="${COMMON_URL%/}"
SEARCH_DATABASE_URL="${SEARCH_DATABASE_URL%/}"

is_url() {
    [[ "$1" =~ ^https?:// || "$1" =~ ^file:// ]]
}

checkpoint_filename_for_source() {
    local source="$1"
    local clean_source="${source%%\?*}"
    clean_source="${clean_source%%#*}"
    clean_source="${clean_source#file://}"
    basename "$clean_source"
}

released_checkpoint_url() {
    "$PYTHON_COMMAND" "$MODEL_MANIFEST_TOOL" \
        checkpoint-url "$1" --manifest "$MODEL_MANIFEST_PATH"
}

model_checkpoint_url() {
    "$PYTHON_COMMAND" "$MODEL_MANIFEST_TOOL" \
        checkpoint-url "$1" \
        --model "$MODEL_NAME" \
        --manifest "$MODEL_MANIFEST_PATH"
}

verify_released_checkpoint() {
    local path="$1"
    local checkpoint_filename="$2"

    "$PYTHON_COMMAND" "$MODEL_MANIFEST_TOOL" \
        verify-checkpoint \
        "$path" \
        --checkpoint "$checkpoint_filename" \
        --manifest "$MODEL_MANIFEST_PATH"
}

download_url() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")" || return 1

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --connect-timeout 30 -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -O "$dest" "$url"
    else
        err "Neither curl nor wget is installed; cannot download $url"
    fi
}

decompress_zst() {
    local source="$1"
    local dest="$2"

    if command -v zstd >/dev/null 2>&1; then
        zstd -d -f -o "$dest" "$source"
    elif command -v unzstd >/dev/null 2>&1; then
        unzstd -f -o "$dest" "$source"
    else
        warn "zstd is required to decompress $source"
        return 1
    fi
}

try_copy_or_download() {
    local source="$1"
    local dest="$2"
    local expected_checkpoint="${3:-}"
    local tmp_dest=""
    local download_dest=""
    local should_decompress_zst=0
    local verification_error=""

    if [[ -f "$dest" && "$FORCE" -eq 0 ]]; then
        if [[ -z "$expected_checkpoint" ]]; then
            info "Already exists: $dest"
            return
        fi
        if verification_error="$(
            verify_released_checkpoint "$dest" "$expected_checkpoint" 2>&1
        )"; then
            info "Already exists and is verified: $dest"
            return
        fi
        warn "Existing checkpoint failed verification (${verification_error}); replacing it: $dest"
    fi

    mkdir -p "$(dirname "$dest")" || return 1
    tmp_dest="$(mktemp "${dest}.tmp.XXXXXX")" || return 1
    TEMP_FILES+=("$tmp_dest")
    download_dest="$tmp_dest"
    if [[ "$source" == *.zst && "$dest" != *.zst ]]; then
        should_decompress_zst=1
        download_dest="$(mktemp "${dest}.download.XXXXXX")" || return 1
        TEMP_FILES+=("$download_dest")
    fi

    if is_url "$source"; then
        if [[ "$source" =~ ^file:// ]]; then
            local local_path="${source#file://}"
            [[ -f "$local_path" ]] || return 1
            cp "$local_path" "$download_dest" || return 1
        else
            info "Downloading $source"
            if ! download_url "$source" "$download_dest"; then
                rm -f "$tmp_dest" "$download_dest"
                return 1
            fi
        fi
    else
        [[ -f "$source" ]] || return 1
        info "Copying $source"
        cp "$source" "$download_dest" || return 1
    fi

    if [[ "$should_decompress_zst" -eq 1 ]]; then
        if ! decompress_zst "$download_dest" "$tmp_dest"; then
            rm -f "$tmp_dest" "$download_dest"
            return 1
        fi
        rm -f "$download_dest" || return 1
    fi

    if [[ -n "$expected_checkpoint" ]] \
        && ! verify_released_checkpoint "$tmp_dest" "$expected_checkpoint"; then
        rm -f "$tmp_dest" "$download_dest"
        return 1
    fi

    mv "$tmp_dest" "$dest" || return 1
}

copy_or_download() {
    local source="$1"
    local dest="$2"
    local expected_checkpoint="${3:-}"

    if ! try_copy_or_download "$source" "$dest" "$expected_checkpoint"; then
        err "Failed to copy/download $source to $dest"
    fi
}

try_download_and_extract_tarball() {
    local tarball="$1"
    local url="${DATA_BASE_URL}/${tarball}"
    local dest="${OPENDDE_ROOT}/${tarball}"

    if ! try_copy_or_download "$url" "$dest"; then
        return 1
    fi
    info "Extracting $dest to $OPENDDE_ROOT"
    if ! tar -xzf "$dest" -C "$OPENDDE_ROOT"; then
        rm -f "$dest"
        return 1
    fi
    rm -f "$dest" || return 1
}

download_and_extract_tarball() {
    local tarball="$1"

    if ! try_download_and_extract_tarball "$tarball"; then
        err "Failed to download/extract ${DATA_BASE_URL}/${tarball}"
    fi
}

download_files_from() {
    local base_url="$1"
    local subdir="$2"
    shift 2
    local file
    local dest_file

    mkdir -p "${OPENDDE_ROOT}/${subdir}" || return 1
    for file in "$@"; do
        dest_file="$file"
        if [[ "$file" == *.zst ]]; then
            dest_file="${file%.zst}"
        fi
        if ! try_copy_or_download "${base_url}/${file}" "${OPENDDE_ROOT}/${subdir}/${dest_file}"; then
            return 1
        fi
    done
}

common_ready() {
    [[ -f "${OPENDDE_ROOT}/common/components.cif" \
        && -f "${OPENDDE_ROOT}/common/components.cif.rdkit_mol.pkl" \
        && -f "${OPENDDE_ROOT}/common/release_date_cache.json" \
        && -f "${OPENDDE_ROOT}/common/obsolete_to_successor.json" ]]
}

search_database_ready() {
    [[ -f "${OPENDDE_ROOT}/search_database/pdb_seqres_2022_09_28.fasta" \
        && -f "${OPENDDE_ROOT}/search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta" \
        && -f "${OPENDDE_ROOT}/search_database/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta" \
        && -f "${OPENDDE_ROOT}/search_database/rnacentral_active_seq_id_90_cov_80_linclust.fasta" ]]
}

info "Using OPENDDE_ROOT_DIR: $OPENDDE_ROOT"
mkdir -p "$OPENDDE_ROOT"

if [[ "$DOWNLOAD_COMMON" -eq 1 ]]; then
    if [[ "$FORCE" -eq 0 ]] && common_ready; then
        info "Common inference data already exists under ${OPENDDE_ROOT}/common"
    else
        info "Downloading common inference data files from ${COMMON_URL}"
        download_files_from "$COMMON_URL" "common" "${COMMON_FILES[@]}" \
            || err "Failed to download common inference data files from ${COMMON_URL}"
    fi
fi

if [[ "$DOWNLOAD_SEARCH_DATABASE" -eq 1 ]]; then
    if [[ "$FORCE" -eq 0 ]] && search_database_ready; then
        info "Search databases already exist under ${OPENDDE_ROOT}/search_database"
    else
        info "Downloading search database files from ${SEARCH_DATABASE_URL}"
        if ! download_files_from "$SEARCH_DATABASE_URL" "search_database" "${SEARCH_DATABASE_FILES[@]}"; then
            warn "Search database files were not all available from ${SEARCH_DATABASE_URL}; trying ${DATA_BASE_URL}/search_database.tar.gz."
            download_and_extract_tarball "search_database.tar.gz"
        fi
    fi
fi

if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
    if [[ -n "$CHECKPOINT_NAME" ]]; then
        if ! model_checkpoint_url "$CHECKPOINT_NAME" >/dev/null 2>&1; then
            err "Checkpoint is not released for model ${MODEL_NAME}: $CHECKPOINT_NAME"
        fi
        MODEL_TARGET_FILE="$CHECKPOINT_NAME"
        MODEL_SOURCE="${MODEL_SOURCE:-${DEPENDENCY_URL}/${CHECKPOINT_NAME}}"
    elif [[ -z "$MODEL_SOURCE" ]]; then
        MODEL_RESOLUTION=""
        if [[ "$MODEL_NAME" == "$MANIFEST_DEFAULT_MODEL" ]]; then
            MODEL_RESOLUTION="$MANIFEST_DEFAULTS"
        else
            MODEL_RESOLUTION="$(
                "$PYTHON_COMMAND" "$MODEL_MANIFEST_TOOL" resolve-model \
                    --model "$MODEL_NAME" \
                    --manifest "$MODEL_MANIFEST_PATH" \
                    2>/dev/null
            )" || MODEL_RESOLUTION=""
        fi
        if [[ -n "$MODEL_RESOLUTION" ]]; then
            IFS=$'\t' read -r \
                _RESOLVED_MODEL \
                MODEL_TARGET_FILE \
                _RESOLVED_CHECKPOINT_URL \
                _RESOLVED_SOURCE_ROOT <<< "$MODEL_RESOLUTION"
            MODEL_SOURCE="${DEPENDENCY_URL}/${MODEL_TARGET_FILE}"
        else
            MODEL_SOURCE="${DEPENDENCY_URL}/${MODEL_NAME}.pt"
            MODEL_TARGET_FILE="${MODEL_NAME}.pt"
        fi
    else
        MODEL_TARGET_FILE="$(checkpoint_filename_for_source "$MODEL_SOURCE")"
        if [[ -z "$MODEL_TARGET_FILE" || "$MODEL_TARGET_FILE" == "." ]]; then
            MODEL_TARGET_FILE="${MODEL_NAME}.pt"
        fi
    fi
    MODEL_EXPECTED_CHECKPOINT=""
    if [[ "$VERIFY_RELEASED_MODEL" -eq 1 ]]; then
        if released_checkpoint_url "$MODEL_TARGET_FILE" >/dev/null 2>&1; then
            MODEL_EXPECTED_CHECKPOINT="$MODEL_TARGET_FILE"
            info "Released checkpoint verification is enabled for ${MODEL_EXPECTED_CHECKPOINT}"
        fi
    fi
    copy_or_download \
        "$MODEL_SOURCE" \
        "${OPENDDE_ROOT}/checkpoint/${MODEL_TARGET_FILE}" \
        "$MODEL_EXPECTED_CHECKPOINT"
fi

if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
    CHECKPOINT_SUMMARY="checkpoint/${MODEL_TARGET_FILE}"
else
    CHECKPOINT_SUMMARY="checkpoint/ (not changed; model download skipped)"
fi

cat <<EOF

OpenDDE inference data is ready under:
  ${OPENDDE_ROOT}

Required runtime layout:
  common/components.cif
  common/components.cif.rdkit_mol.pkl
  common/release_date_cache.json
  common/obsolete_to_successor.json
  search_database/pdb_seqres_2022_09_28.fasta
  search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
  search_database/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
  search_database/rnacentral_active_seq_id_90_cov_80_linclust.fasta
  ${CHECKPOINT_SUMMARY}

Set this before inference:
  export OPENDDE_ROOT_DIR="${OPENDDE_ROOT}"
EOF
