#!/usr/bin/env bash
# Install everything that's pip-installable (CasADi+IPOPT, Crocoddyl,
# mim_solvers, matplotlib, tabulate) into the project venv via uv,
# then patch the missing libpinocchio_*.so.3 symlinks. Idempotent.

. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

require_cmd uv

log "uv sync (test + mpc-examples + comparisons groups)..."
cd "$REPO_ROOT"
UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-0}" \
    uv sync --extra test --extra mpc-examples --group comparisons

# Patch the missing `libpinocchio_*.so.3` symlinks in the cmeel.prefix
# install. mim_solvers / Crocoddyl link against `.so.3`, but the cmeel
# wheel for pinocchio 3.8 only ships the `.so.3.8.0` files.
log "Patching libpinocchio_*.so.3 symlinks..."
CMEEL_LIB=$(ls -d "$VENV_DIR"/lib/python*/site-packages/cmeel.prefix/lib 2>/dev/null | head -1)
[ -d "$CMEEL_LIB" ] || die "cmeel.prefix/lib not found under $VENV_DIR"
cd "$CMEEL_LIB"
for f in libpinocchio_collision libpinocchio_default \
         libpinocchio_parsers libpinocchio_visualizers; do
    ln -sf "${f}.so.3.8.0" "${f}.so.3"
done

log "pip deps installed"
