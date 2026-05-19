"""Aliengo trot MJX problem.

Wraps ``tests.mpc_examples.configs.config_aliengo_trot`` into a
``ProblemSpec`` via the same builder used by the barrel-roll problem.
"""

from __future__ import annotations

from tests.comparison.problem_spec import ProblemSpec
from tests.comparison.problems._mjx_base import build_mjx_problem


def make_problem() -> ProblemSpec:
    return build_mjx_problem(
        "tests.mpc_examples.configs.config_aliengo_trot",
        name="aliengo_trot",
    )
