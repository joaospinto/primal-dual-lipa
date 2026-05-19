#!/usr/bin/env bash
# "Install" fatrop for the LIPA comparison driver.
#
# fatrop ships a CasADi `nlpsol` plugin and the CasADi pip wheel
# (>=3.7) bundles a prebuilt copy of that plugin. So in the common
# case we don't have to build anything: as long as the project venv
# already has CasADi (it does — the `comparisons` group pulls it in
# via the IPOPT and acados adapters), `nlpsol(...,'fatrop',...)`
# already works out of the box.
#
# This script:
#   1. Verifies the bundled CasADi-fatrop plugin loads in the project venv.
#   2. (Optionally) clones the upstream meco-group/fatrop repo into
#      $FATROP_DIR so the user has access to the official examples and
#      documentation. The clone is purely informational; nothing in this
#      repo (the comparison driver included) reads from it.
#
# Idempotent: skips already-done bits.
#
# The clone location defaults to $HOME/github/fatrop (mirroring the
# acados convention); override with FATROP_DIR=/some/other/path.

. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

require_cmd git
require_cmd uv

FATROP_DIR="${FATROP_DIR:-$HOME/github/fatrop}"

# 1. Verify the bundled CasADi fatrop plugin is available.
log "checking that CasADi has the fatrop nlpsol plugin..."
cd "$REPO_ROOT"
UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-0}" \
    uv run --no-sync python - <<'PY'
import sys
import casadi as ca
try:
    # plugin_load throws if the plugin isn't bundled with the wheel.
    ca.Importer.plugin_load("nlpsol", "fatrop")  # noqa: SLF001
except AttributeError:
    # Fallback: instantiate a 1-var fatrop solver and see if it crashes
    # at the C++ "plugin not found" path.
    nlp = {"x": ca.SX.sym("x"), "f": ca.SX(1.0), "g": ca.SX(0.0)}
    try:
        ca.nlpsol("probe", "fatrop", nlp, {
            "structure_detection": "manual",
            "nx": [1, 0], "nu": [0, 0], "ng": [0, 0], "N": 1,
            "fatrop": {"print_level": 0},
        })
    except Exception as e:  # noqa: BLE001
        if "Plugin" in str(e) and "could not be loaded" in str(e):
            print("CasADi fatrop plugin NOT available:", e, file=sys.stderr)
            sys.exit(1)
print(f"casadi {ca.__version__}: fatrop plugin available")
PY

# 2. Clone the upstream fatrop repo for examples/docs (informational only).
if [ ! -d "$FATROP_DIR/.git" ]; then
    log "cloning fatrop into $FATROP_DIR (for examples/docs only)..."
    mkdir -p "$(dirname "$FATROP_DIR")"
    git clone --recursive --depth 1 https://github.com/meco-group/fatrop.git "$FATROP_DIR"
else
    log "fatrop already cloned at $FATROP_DIR"
fi

log "fatrop ready"
log "Run the smoke test with:"
log "  UV_NO_BUILD_ISOLATION=0 uv run --no-sync \\"
log "      python -m tests.comparison.run_benchmark \\"
log "      --problems cartpole,acrobot --solvers fatrop --max-iter 500 \\"
log "      --out-dir /tmp/fatrop_smoke"
