"""Per-problem ProblemSpec factories.

Each problem module exports ``make_problem()`` returning a
``ProblemSpec``. Some additionally attach metadata for solver-specific
adapters (e.g. ``casadi_builder`` for the IPOPT-via-CasADi path,
``crocoddyl_action_factory`` for the CSQP path).
"""

from __future__ import annotations

import importlib
from typing import Callable

from tests.comparison.problem_spec import ProblemSpec, validate_metadata

# Map of problem name -> (module path, extra kwargs to pass to make_problem).
_PROBLEMS: dict[str, tuple[str, dict]] = {
    "cartpole": ("tests.comparison.problems.cartpole", {}),
    "acrobot": ("tests.comparison.problems.acrobot", {}),
    "quadpendulum": ("tests.comparison.problems.quadpendulum", {}),
    # Theta-enabled variant (cross-stage decision variable, only LIPA
    # and IPOPT can ingest natively). Maps to the same module with
    # with_theta=True.
    "quadpendulum_theta": (
        "tests.comparison.problems.quadpendulum",
        {"with_theta": True},
    ),
    "barrel_roll": ("tests.comparison.problems.mjx_barrel_roll", {}),
    # Short-horizon variants used to validate the IPOPT-via-JAX-callback
    # path can run end-to-end in tractable time. Same problem otherwise.
    "barrel_roll_N20": (
        "tests.comparison.problems.mjx_barrel_roll",
        {"N_override": 20},
    ),
    "barrel_roll_N10": (
        "tests.comparison.problems.mjx_barrel_roll",
        {"N_override": 10},
    ),
    "backflip": ("tests.comparison.problems.mjx_backflip", {}),
    "jump": ("tests.comparison.problems.mjx_jump", {}),
    "trot": ("tests.comparison.problems.mjx_trot", {}),
}


def all_problem_names() -> list[str]:
    return list(_PROBLEMS.keys())


def make_problem(name: str, **extra_kwargs) -> ProblemSpec:
    if name not in _PROBLEMS:
        raise KeyError(f"Unknown problem: {name!r}. Known: {list(_PROBLEMS.keys())}")
    module_path, builtin_kwargs = _PROBLEMS[name]
    module = importlib.import_module(module_path)
    kwargs = {**builtin_kwargs, **extra_kwargs}
    spec = module.make_problem(**kwargs)
    validate_metadata(spec.metadata, spec.name)
    return spec
