"""Per-solver adapters that consume a ``ProblemSpec`` and return a ``SolverResult``.

Each adapter degrades gracefully if its solver is not importable: the
adapter module imports its solver lazily inside ``solve()``, and the
registry below catches ``ImportError`` so the benchmark runner can skip
the adapter without crashing.
"""

from __future__ import annotations

from collections.abc import Callable

from tests.comparison.adapters.base import SolverAdapter

# Registry of (name -> factory). Populated below.
_ADAPTERS: dict[str, Callable[..., SolverAdapter]] = {}


def register(
    name: str,
) -> Callable[[Callable[..., SolverAdapter]], Callable[..., SolverAdapter]]:
    def decorator(
        factory: Callable[..., SolverAdapter],
    ) -> Callable[..., SolverAdapter]:
        _ADAPTERS[name] = factory
        return factory

    return decorator


def all_adapter_names() -> list[str]:
    return list(_ADAPTERS.keys())


def get_adapter(name: str, **kwargs) -> SolverAdapter:
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown solver: {name!r}. Known: {list(_ADAPTERS.keys())}")
    return _ADAPTERS[name](**kwargs)


# Eager registration: importing each module registers the factory.
# Import errors inside an adapter module are surfaced when the adapter
# is actually requested (we don't want a broken acados install to take
# the whole comparison runner down).
from tests.comparison.adapters import lipa  # noqa: F401, E402
from tests.comparison.adapters import ipopt_casadi  # noqa: F401, E402
from tests.comparison.adapters import ipopt_mjx_sparse  # noqa: F401, E402
from tests.comparison.adapters import csqp  # noqa: F401, E402
from tests.comparison.adapters import aligator  # noqa: F401, E402
from tests.comparison.adapters import acados  # noqa: F401, E402
from tests.comparison.adapters import fatrop  # noqa: F401, E402
from tests.comparison.adapters import sip  # noqa: F401, E402
from tests.comparison.adapters import fatrop_mjx  # noqa: F401, E402
from tests.comparison.adapters import trajax  # noqa: F401, E402
