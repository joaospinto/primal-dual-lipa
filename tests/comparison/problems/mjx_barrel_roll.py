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

    spec = build_mjx_problem(
        "tests.mpc_examples.configs.config_barrel_roll",
        name="barrel_roll",
    )
    spec.metadata["sip_jax_settings"] = {
        "penalty": {
            "initial_penalty_parameter": 1e9,
            "penalty_parameter_increase_factor": 1.1,
        },
        "barrier": {
            "initial_mu": 1e-2,
            "mu_update_factor": 0.9,
        },
        "regularization": {
            "initial": 0.03,
            "maximum": 1e12,
            "max_attempts": 32,
        },
    }
    spec.metadata["sip_two_phase"] = True
    spec.metadata["sip_warmup_settings"] = {
        "max_iterations": 100,
        "termination": {
            "max_dual_residual": 1.0,
            "max_constraint_violation": 1e-3,
            "max_complementarity_gap": 1e-3,
        },
    }
    return spec
