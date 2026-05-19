#!/usr/bin/env bash
# One-shot installer: pip deps + Aligator (conda-forge side env) + acados.
# Idempotent — safe to re-run; each subscript skips already-done work.
#
# Prerequisites the user must have:
#   - uv (https://docs.astral.sh/uv/)
#   - mamba or conda (https://github.com/conda-forge/miniforge)
#   - cmake, git, a C++17 toolchain
# The Dockerfile in this directory installs the same prerequisites then
# runs this script.

. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

cd "$(dirname "${BASH_SOURCE[0]}")"

log "Step 1/3: pip deps + libpinocchio symlink fixup"
bash install_pip_deps.sh

log "Step 2/3: Aligator via conda-forge side env"
bash install_aligator.sh

log "Step 3/3: acados from source"
bash install_acados.sh

log
log "All installers finished."
log "Run the comparison with:"
log "  bash tests/comparison/run_all.sh"
log
log "Make sure these env vars are exported in your shell first:"
log "  export ACADOS_SOURCE_DIR=$ACADOS_DIR"
log "  export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:$ACADOS_DIR/lib"
log "  export ALIGATOR_SITE_PACKAGES=...  # see install_aligator.sh output"
