"""Trot (Unitree Aliengo) MJX problem."""

from __future__ import annotations

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem() -> ProblemSpec:
    spec = build_mjx_problem(
        "tests.mpc_examples.configs.config_aliengo_trot",
        name="trot",
    )
    spec.metadata["sip_jax_settings"] = {
        "penalty": {"penalty_parameter_increase_factor": 1.0},
        "barrier": {
            "initial_mu": 1e-3,
            "mu_update_factor": 0.9,
        },
        "regularization": {"initial": 0.01},
    }
    spec.metadata["sip_two_phase"] = True
    spec.metadata["lipa_two_phase"] = False
    return spec
