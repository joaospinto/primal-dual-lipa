# Reproducible CPU-only environment for the LIPA-vs-SOTA solver comparison.
#
# This Dockerfile is a thin wrapper around the same install scripts that
# host users invoke from `tests/comparison/install/install_all.sh`. The
# only Docker-specific bits are (a) the OS / build-toolchain setup, and
# (b) the env-var defaults that point Aligator and acados into /opt
# instead of $HOME.
#
# Use:
#   # From the repo root:
#   docker build -t lipa-comparison -f tests/comparison/Dockerfile .
#   docker run --rm -v "$PWD/comparison_results:/lipa/comparison_results" \
#       lipa-comparison
#
# Caveats:
#   - CPU-only. JAX runs without CUDA; LIPA's GPU `parallel_lqr`
#     codepath is exercised only if you install on a host with
#     nvidia-docker + a CUDA build of JAX.
#   - The image is ~6 GB once acados, conda, and the JAX/CUDA wheels
#     are in. ~25 min initial build on a modest CI box.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps: build toolchain (acados + the cmeel pinocchio wheels need
# C++17), git/curl, BLAS/LAPACK for blasfeo's reference fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        libopenblas-dev \
        liblapack-dev \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (ships conda-forge as default channel — Aligator lives there).
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && $CONDA_DIR/bin/conda config --set always_yes true
ENV PATH=$CONDA_DIR/bin:$PATH

# uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# Park acados under /opt so the layer cache survives source edits.
ENV ACADOS_DIR=/opt/acados

# Copy install scripts + minimal project metadata first so the heavy
# install layers stay cached when only LIPA source changes.
WORKDIR /lipa
COPY pyproject.toml uv.lock /lipa/
COPY primal_dual_lipa/__init__.py /lipa/primal_dual_lipa/
COPY tests/comparison/install /lipa/tests/comparison/install

# Run each install script as its own RUN — gives clean layer separation
# (acados rebuild doesn't bust the conda env layer, etc.).
ENV UV_NO_BUILD_ISOLATION=0

# Sync deps without the project itself (project source not copied yet).
RUN uv sync --extra test --extra mpc-examples --group comparisons --no-install-project \
    && bash tests/comparison/install/install_pip_deps.sh || true
# `|| true` because install_pip_deps re-runs uv sync — its symlink
# fixup is the part that matters here.

RUN bash tests/comparison/install/install_aligator.sh
RUN bash tests/comparison/install/install_acados.sh

# Runtime env vars.
ENV ACADOS_SOURCE_DIR=$ACADOS_DIR
ENV LD_LIBRARY_PATH=$ACADOS_DIR/lib
ENV ALIGATOR_SITE_PACKAGES=/opt/conda/envs/aligator-side/lib/python3.13/site-packages

# Now copy the rest of the project. Edits here don't bust the install
# layers above.
COPY . /lipa

# Re-sync to install the project as editable. Cheap (everything else cached).
RUN uv sync --extra test --extra mpc-examples --group comparisons

CMD ["bash", "tests/comparison/run_all.sh"]
