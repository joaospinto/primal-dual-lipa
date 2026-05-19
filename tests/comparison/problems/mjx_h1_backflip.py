"""H1 backflip MJX problem."""

from __future__ import annotations

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem() -> ProblemSpec:
    spec = build_mjx_problem(
        "tests.mpc_examples.configs.config_h1_backflip",
        name="h1_backflip",
    )
    # SIP-MJX adapter defaults destabilise on the h1 problems; smaller
    # eta0 + faster ramp recovers the LIPA basin.
    spec.metadata["sip_mjx_extra_settings"] = {
        "initial_penalty_parameter": 1e5,
        "penalty_parameter_increase_factor": 2.0,
    }
    return spec
