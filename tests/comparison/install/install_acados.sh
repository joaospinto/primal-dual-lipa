#!/usr/bin/env bash
# Clone, build, and install acados from source, then install the
# `acados_template` Python interface (editable) into the project venv.
# Idempotent: skips already-built bits.
#
# The install location defaults to $HOME/github/acados; override with
# ACADOS_DIR=/some/other/path. Two env vars must then be set in the
# shell that runs the comparison driver:
#   export ACADOS_SOURCE_DIR=$ACADOS_DIR
#   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ACADOS_DIR/lib
# The script prints these at the end.

. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

require_cmd git
require_cmd cmake
require_cmd uv

if [ ! -d "$ACADOS_DIR/.git" ]; then
    log "cloning acados into $ACADOS_DIR..."
    mkdir -p "$(dirname "$ACADOS_DIR")"
    git clone --recursive --depth 1 https://github.com/acados/acados.git "$ACADOS_DIR"
else
    log "acados already cloned at $ACADOS_DIR"
fi

mkdir -p "$ACADOS_DIR/build"
cd "$ACADOS_DIR/build"

# Skip the cmake step if libacados is already built.
if [ ! -f "$ACADOS_DIR/lib/libacados.so" ]; then
    log "configuring + building acados (this takes ~5 min on 8 cores)..."
    if command -v ninja >/dev/null; then
        cmake -GNinja -DACADOS_INSTALL_DIR="$ACADOS_DIR" ..
    else
        cmake -DACADOS_INSTALL_DIR="$ACADOS_DIR" ..
    fi
    cmake --build . --target install
else
    log "libacados.so already built; skipping cmake step"
fi

# acados generates per-OCP C code via Tera (Rust template engine).
# The renderer ships separately as a static binary.
T_RENDERER="$ACADOS_DIR/bin/t_renderer"
if [ ! -x "$T_RENDERER" ]; then
    log "downloading t_renderer..."
    mkdir -p "$ACADOS_DIR/bin"
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)   suffix=linux-amd64 ;;
        Linux-aarch64)  suffix=linux-arm64 ;;
        Darwin-x86_64)  suffix=osx-amd64 ;;
        Darwin-arm64)   suffix=osx-arm64 ;;
        *) die "no t_renderer build for $(uname -s)-$(uname -m)" ;;
    esac
    curl -fL -o "$T_RENDERER" \
        "https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-${suffix}"
    chmod +x "$T_RENDERER"
else
    log "t_renderer already present"
fi

cd "$REPO_ROOT"
# acados_template's setup.py uses setuptools but doesn't declare it as
# a build-system requirement. We pass --no-build-isolation so the
# editable install uses the project venv as its build env, which means
# setuptools must already be present there. The project itself uses
# hatchling so setuptools isn't pulled in by uv sync — install it
# explicitly first. Idempotent (uv pip install is a no-op if present).
log "ensuring setuptools is available for the acados_template build..."
uv pip install setuptools

log "installing acados_template (editable) into the project venv..."
UV_NO_BUILD_ISOLATION="${UV_NO_BUILD_ISOLATION:-0}" \
    uv pip install --no-build-isolation -e "$ACADOS_DIR/interfaces/acados_template"

log "acados installed"
log "Add to your shell:"
log "  export ACADOS_SOURCE_DIR=$ACADOS_DIR"
log "  export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:$ACADOS_DIR/lib"
log "(IMPORTANT: append, do NOT prepend — prepending puts acados's"
log " libblasfeo.so in front of JAX's CPU runtime and segfaults numpy)"
