"""MJX whole-body MPC examples driven by the LIPA solver.

LIPA's main loop has a float64 sentinel (``optimizers.py`` uses
``jnp.array(jnp.inf, dtype=jnp.float64)`` as the initial cost-improvement
value). Without ``jax_enable_x64`` that gets silently truncated to a
finite float32 and the cost-improvement convergence check breaks. We
enable x64 here so that any consumer of this package — the run_offline
script, the smoke tests, or someone importing :mod:`offline_solver`
directly — gets the right setting.

This must be called before any jax computation, hence the package-level
side effect.
"""

import jax

jax.config.update("jax_enable_x64", True)
