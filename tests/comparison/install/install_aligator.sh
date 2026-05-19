#!/usr/bin/env bash
# Install Aligator into a side conda env (conda-forge channel). The
# Aligator adapter loads it from the project venv via sys.path
# injection; setting ALIGATOR_SITE_PACKAGES makes the search direct.
#
# Why a side env: Aligator's PyPI sdist is broken (CMake project with
# no Python build backend), and its bundled pinocchio 3.4 conflicts
# with mim_solvers' pinocchio 3.8 if installed into the same venv.
#
# Idempotent: skips if the env already has aligator importable.

. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

# Prefer mamba (faster) but fall back to conda.
SOLVER=$(command -v mamba || command -v conda || true)
[ -n "$SOLVER" ] || die "neither mamba nor conda is on PATH; install miniforge first"
log "using $SOLVER"

if "$SOLVER" run -n "$ALIGATOR_CONDA_ENV" python -c "import aligator" 2>/dev/null; then
    log "aligator is already importable in env '$ALIGATOR_CONDA_ENV'; skipping"
else
    log "creating conda env '$ALIGATOR_CONDA_ENV' with aligator..."
    "$SOLVER" create -n "$ALIGATOR_CONDA_ENV" -c conda-forge \
        -y python="$PYTHON_VERSION" aligator
fi

# Discover the env's site-packages so we can print the value the user
# should export for ALIGATOR_SITE_PACKAGES.
SP=$("$SOLVER" run -n "$ALIGATOR_CONDA_ENV" python -c \
    "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
[ -n "$SP" ] || die "could not determine site-packages for env '$ALIGATOR_CONDA_ENV'"

log "Aligator installed at $SP"
log "Add to your shell:"
log "  export ALIGATOR_SITE_PACKAGES=$SP"
