"""Trot (Unitree Aliengo) MJX problem."""

from __future__ import annotations

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem() -> ProblemSpec:
    spec = build_mjx_problem(
        "tests.mpc_examples.configs.config_aliengo_trot",
        name="trot",
    )
    spec.metadata["sip_jax_settings"] = dict(
        penalty_parameter_increase_factor=1.0,
        mu_update_factor=0.9,
        initial_mu=1e-3,
    )
    return spec
