"""Barrel-roll MJX problem.

Wraps ``tests.mpc_examples.configs.config_barrel_roll`` into a
``ProblemSpec`` via the shared MJX builder in ``_mjx_base``.
"""

from __future__ import annotations

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem(N_override: int | None = None) -> ProblemSpec:
    """Build the barrel_roll ProblemSpec.

    ``N_override`` shortens the horizon for IPOPT-solver feasibility
    tests — the default ``N=100`` produces a flat NLP with ~7400
    variables and ~600K nonzeros even with sparse Jacobians, which
    IPOPT-via-CasADi-callback handles but with a long compile phase.
    Setting ``N_override=10`` or ``20`` makes a smaller smoke test.
    """
    if N_override is not None:
        # Patch the config's N before building. We import the module by
        # value so the override doesn't leak into other consumers.
        import importlib

        config = importlib.import_module(
            "tests.mpc_examples.configs.config_barrel_roll",
        )
        original_N = config.N
        config.N = N_override
        try:
            spec = build_mjx_problem(
                "tests.mpc_examples.configs.config_barrel_roll",
                name=f"barrel_roll_N{N_override}",
            )
        finally:
            config.N = original_N
        return spec

    return build_mjx_problem(
        "tests.mpc_examples.configs.config_barrel_roll",
        name="barrel_roll",
    )
