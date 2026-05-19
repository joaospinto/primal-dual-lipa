"""Jump forward (Unitree H1) MJX problem."""

from __future__ import annotations

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem() -> ProblemSpec:
    spec = build_mjx_problem(
        "tests.mpc_examples.configs.config_h1_jump_forward",
        name="jump",
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
            "initial_penalty_parameter": 1e5,
            "penalty_parameter_increase_factor": 2.0,
        }
    )
    return spec
