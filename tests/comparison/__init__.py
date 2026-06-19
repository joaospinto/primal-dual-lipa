"""LIPA-vs-other-solvers comparison driver.

Standalone CLI for benchmarking primal-dual LIPA against other NLP-OCP
solvers (IPOPT, CSQP, Aligator, acados) on the test problems in this
repo. See ``run_benchmark.py`` for the entry point and the project
``PLAN.md`` for the design.
"""

import os
import jax

# Float64 for numerical-comparison consistency.
jax.config.update("jax_enable_x64", True)

# Persistent JAX/XLA compilation cache across subprocess runs. The MJX
# problems take 30-300s to JIT-compile from a cold cache; with the
# cache populated, each subsequent solve of the same problem skips
# that compile entirely. Subprocess-per-pair isolation in the runner
# would otherwise force a full recompile every time.
# Override path via ``JAX_COMPILATION_CACHE_DIR`` env var if needed.
_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax/primal-dual-lipa"),
)
if _cache_dir:
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    except OSError:
        # Read-only filesystem (sandbox, etc.) — skip cache config silently.
        pass
