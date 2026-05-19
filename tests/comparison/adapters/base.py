"""SolverAdapter base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tests.comparison.problem_spec import ProblemSpec, SolverResult


class SolverAdapter(ABC):
    """Adapter interface: ``solve(problem) -> SolverResult``."""

    name: str

    @abstractmethod
    def solve(self, problem: ProblemSpec) -> SolverResult:
        """Run the solver on the problem and return a unified result."""

    def is_available(self) -> tuple[bool, str]:
        """Return ``(available, reason_if_not)`` so the runner can skip cleanly."""
        return True, ""
