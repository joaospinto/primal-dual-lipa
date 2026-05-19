# Shared bash helpers for the install scripts. Source this from each
# script with `. "$(dirname "$0")/_lib.sh"`.

set -euo pipefail

log() {
    printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*" >&2
}

die() {
    log "FATAL: $*"
    exit 1
}

# Repo root, derived from the location of this file.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Default install locations. Each can be overridden via env var so the
# Dockerfile (which installs into /opt) and the host install (which goes
# under $HOME) can share the same scripts.
ACADOS_DIR="${ACADOS_DIR:-$HOME/github/acados}"
ALIGATOR_CONDA_ENV="${ALIGATOR_CONDA_ENV:-aligator-side}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

# Project venv root. uv puts it at .venv inside the project by default.
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required but not on PATH"
}
