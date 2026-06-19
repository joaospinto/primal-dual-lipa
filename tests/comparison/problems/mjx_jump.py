"""Jump forward (Unitree H1) MJX problem."""

from __future__ import annotations

import dataclasses

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem() -> ProblemSpec:
    spec = build_mjx_problem(
        "tests.mpc_examples.configs.config_h1_jump_forward",
        name="jump",
        control_warm_start="h1_kinodynamic",
    )
    # ipopt-mjx tuning. A small ``mu_init`` keeps the barrier close to
    # the boundary from the start, avoiding the spurious early
    # restoration entries that a default ``mu_init=0.1`` triggers on
    # jump-style contact transitions. ``monotone`` decays mu
    # deterministically from there; ``alpha_for_y=min`` pairs the dual
    # step size with the primal step.
    spec.metadata["ipopt_mjx_extra_options"] = {
        "mu_init": 1e-3,
        "mu_strategy": "monotone",
        "alpha_for_y": "min",
    }
    spec.metadata["sip_settings"].update(
        {
            "max_iterations": 20,
            "penalty": {
                "initial_penalty_parameter": 1e3,
                "penalty_parameter_increase_factor": 1.1,
            },
            "barrier": {
                "initial_mu": 1e-3,
                "mu_update_factor": 0.95,
            },
            "regularization": {
                "initial": 0.03,
                "maximum": 1e12,
                "max_attempts": 32,
            },
            "line_search": {
                "enable_line_search_failures": True,
                "max_iterations": 200,
            },
            "time_limit_s": 90.0,
        }
    )
    spec.metadata["lipa_two_phase"] = False
    spec.metadata["lipa_settings"] = dataclasses.replace(
        spec.metadata["lipa_settings"],
        max_iterations=450,
        η0=3e5,
    )
    return spec
