#!/usr/bin/env bash
# Install Google's Trajax (https://github.com/google/trajax) into the
# project venv. Trajax is a JAX-native iLQR / AL-iLQR trajectory
# optimization library; the package on PyPI named `trajax` is a
# DIFFERENT (unrelated MPPI/MPCC) library, so we explicitly install
# from the Google git repo.
#
# Idempotent: re-running just verifies the import still works.
#
# Override the clone location with TRAJAX_DIR=/some/other/path
# (default $HOME/github/trajax). Only used for the editable-install
# fallback path; the default install path is the git+https one-liner.

. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

require_cmd uv

TRAJAX_DIR="${TRAJAX_DIR:-$HOME/github/trajax}"

log "installing trajax (Google) from git into the project venv..."
UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-0}" \
    uv pip install \
        --python "$VENV_DIR/bin/python3" \
        --no-build-isolation \
        "git+https://github.com/google/trajax" \
    || {
        log "git+https install failed; falling back to editable install from $TRAJAX_DIR"
        if [ ! -d "$TRAJAX_DIR/.git" ]; then
            log "cloning trajax into $TRAJAX_DIR..."
            mkdir -p "$(dirname "$TRAJAX_DIR")"
            git clone --depth 1 https://github.com/google/trajax.git "$TRAJAX_DIR"
        fi
        UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-0}" \
            uv pip install \
                --python "$VENV_DIR/bin/python3" \
                --no-build-isolation \
                -e "$TRAJAX_DIR"
    }

log "verifying import..."
UV_NO_BUILD_ISOLATION=0 uv run --no-sync python -c \
    "from trajax.optimizers import ilqr, constrained_ilqr; print('ok')"

log "trajax installed; no env vars needed (pure-Python JAX package)"
