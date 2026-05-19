"""Registry of (solver_label, problem) pairs that always hit the
subprocess hard-kill on this benchmark setup.

When ``run_benchmark.py`` runs WITHOUT ``--ignore-known-timeouts``,
these pairs short-circuit to a ``SolverResult`` with
``notes='hits process timeout (cached)'`` instead of spawning the
subprocess. The pair still appears in the report so the cross-table
has no silent "-" gaps — only the wasted wall-clock of the doomed
solve is saved.

Subagent tuning runs MUST pass ``--ignore-known-timeouts`` so they
can iterate on the offending pair. When a tuning fix lands, the
person merging the fix is expected to delete the corresponding entry
here in the same change.
"""

from __future__ import annotations

KNOWN_HARD_KILLS: set[tuple[str, str]] = {
    ("trajax", "trot"),
    ("trajax", "barrel_roll"),
    ("fatrop-jax", "backflip"),
    ("fatrop-jax", "barrel_roll"),
    ("fatrop-jax", "jump"),
    ("fatrop-jax", "quadpendulum"),
    ("fatrop-jax", "trot"),
    ("ipopt-jax", "backflip"),
    ("ipopt-jax", "jump"),
    ("sip-jax", "backflip"),
    ("sip-jax", "jump"),
    ("csqp-jax", "barrel_roll"),
    ("csqp-jax", "backflip"),
    ("csqp-jax", "jump"),
    ("csqp-jax", "trot"),
    ("aligator-jax", "barrel_roll"),
    ("aligator-jax", "jump"),
    ("aligator-jax", "trot"),
}


def is_known_hard_kill(solver_label: str, problem_name: str) -> bool:
    """Return True if (solver_label, problem_name) is in the registry."""
    return (solver_label, problem_name) in KNOWN_HARD_KILLS
