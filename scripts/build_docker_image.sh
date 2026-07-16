#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${OPENDDE_DOCKER_IMAGE:-opendde:local}"
PLATFORM="${OPENDDE_DOCKER_PLATFORM:-linux/amd64}"
PULL=0
NO_CACHE=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build a local OpenDDE GPU image for user-managed use. This helper does not
inspect Git state, publish images, or participate in CI/release automation.

Options:
  --tag IMAGE       Local image name and tag. Defaults to ${IMAGE}.
  --platform VALUE  Docker target platform. Defaults to ${PLATFORM}.
  --pull            Always attempt to pull a newer base image.
  --no-cache        Disable Docker's build cache.
  --dry-run         Print the resolved build command without running Docker.
  -h, --help        Show this help message.

Environment overrides:
  OPENDDE_DOCKER_IMAGE, OPENDDE_DOCKER_PLATFORM
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            [[ $# -ge 2 && -n "$2" ]] \
                || { echo "ERROR: --tag requires a value" >&2; exit 2; }
            IMAGE="$2"
            shift 2
            ;;
        --platform)
            [[ $# -ge 2 && -n "$2" ]] \
                || { echo "ERROR: --platform requires a value" >&2; exit 2; }
            PLATFORM="$2"
            shift 2
            ;;
        --pull)
            PULL=1
            shift
            ;;
        --no-cache)
            NO_CACHE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

build_command=(
    docker build
    --file "${REPO_ROOT}/Dockerfile"
    --tag "$IMAGE"
    --platform "$PLATFORM"
)
[[ "$PULL" -eq 0 ]] || build_command+=(--pull)
[[ "$NO_CACHE" -eq 0 ]] || build_command+=(--no-cache)
build_command+=("$REPO_ROOT")

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${build_command[@]}"
    printf '\n'
    exit 0
fi

command -v docker >/dev/null 2>&1 \
    || { echo "ERROR: docker is not installed or not on PATH" >&2; exit 1; }

DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" "${build_command[@]}"
