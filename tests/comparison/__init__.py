"""LIPA-vs-other-solvers comparison driver.

Standalone CLI for benchmarking primal-dual LIPA against other NLP-OCP
solvers (IPOPT, CSQP, Aligator, acados) on the test problems in this
repo. See ``run_benchmark.py`` for the entry point and the project
``PLAN.md`` for the design.
"""

# Float64 for numerical-comparison consistency.
import jax

jax.config.update("jax_enable_x64", True)
