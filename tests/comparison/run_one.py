"""Subprocess entry point for a single (solver, problem) pair.

Invoked by ``run_benchmark._run_one_via_subprocess`` when
``--hard-timeout-s`` is set. Runs exactly one pair in its own Python
process (so a hung solver can be SIGKILLed without taking down the
whole pass), then pickles the resulting ``SolverResult`` to the
``--out-pickle`` path.

Why subprocess isolation: most solvers (lipa, fatrop, csqp,
trajax, aligator, acados) have no native wall-time cap and can spin
arbitrarily long if they hit a bad iterate. IPOPT and SIP honor
``--timeout-s`` natively; for the rest, the parent process kills this
subprocess at ``--hard-timeout-s`` (which should be set noticeably
larger than ``--timeout-s`` so a solver that DOES honor the soft cap
has time to exit cleanly first, and so JIT compilation has time to
finish — see ``run_all.sh`` for the budget rationale).

This module is not meant to be imported; it's a pure CLI tool.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from tests.comparison.problem_spec import SolverResult, make_failure_result
from tests.comparison.run_benchmark import _run_one_in_process


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--tol", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument(
        "--problem-kwargs-json",
        type=str,
        default=None,
        help="JSON-encoded extra kwargs dict for the problem factory.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="JSON-encoded metadata overlay for the constructed ProblemSpec.",
    )
    parser.add_argument(
        "--initial-solution-npz",
        type=Path,
        default=None,
        help="Saved solution archive whose X/U/Theta arrays replace the "
        "constructed problem's primal warm start.",
    )
    parser.add_argument(
        "--solver-kwargs-json",
        type=str,
        default=None,
        help="JSON-encoded extra-kwargs dict; same semantics as "
        "run_benchmark.py's --solver-kwargs-json (inline JSON only "
        "here — the parent process has already resolved any file path).",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Pass through to the adapter; same as run_benchmark.py.",
    )
    parser.add_argument(
        "--out-pickle",
        type=Path,
        required=True,
        help="Path where the pickled SolverResult is written on clean exit.",
    )
    args = parser.parse_args(argv)

    extra_kwargs: dict = {}
    if args.solver_kwargs_json:
        try:
            extra_kwargs = json.loads(args.solver_kwargs_json)
        except json.JSONDecodeError as e:
            # The parent should have validated this already, but be
            # defensive — emit a failure result the parent can read.
            result = make_failure_result(
                args.solver,
                args.problem,
                f"run_one: bad --solver-kwargs-json: {e}",
            )
            args.out_pickle.parent.mkdir(parents=True, exist_ok=True)
            with args.out_pickle.open("wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return 2

    problem_kwargs: dict = {}
    if args.problem_kwargs_json:
        try:
            problem_kwargs = json.loads(args.problem_kwargs_json)
        except json.JSONDecodeError as e:
            result = make_failure_result(
                args.solver,
                args.problem,
                f"run_one: bad --problem-kwargs-json: {e}",
            )
            args.out_pickle.parent.mkdir(parents=True, exist_ok=True)
            with args.out_pickle.open("wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return 2

    metadata_overlay: dict = {}
    if args.metadata_json:
        try:
            metadata_overlay = json.loads(args.metadata_json)
        except json.JSONDecodeError as e:
            result = make_failure_result(
                args.solver,
                args.problem,
                f"run_one: bad --metadata-json: {e}",
            )
            args.out_pickle.parent.mkdir(parents=True, exist_ok=True)
            with args.out_pickle.open("wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return 2

    try:
        result: SolverResult = _run_one_in_process(
            args.solver,
            args.problem,
            max_iter=args.max_iter,
            tol=args.tol,
            timeout_s=args.timeout_s,
            # Keep subprocess logs self-contained. Hard-timeout MJX runs need
            # phase-level progress to distinguish slow solves from hangs.
            verbose=True,
            solver_verbose=args.solver_verbose,
            backend=args.backend,
            extra_kwargs=extra_kwargs,
            problem_kwargs=problem_kwargs,
            metadata_overlay=metadata_overlay,
            initial_solution_npz=args.initial_solution_npz,
        )
    except Exception as e:  # noqa: BLE001
        # _run_one_in_process already catches per-adapter exceptions; any
        # leak here is a bug in our code, not in the adapter. Still don't
        # crash the subprocess — emit a failure result the parent can read.
        result = make_failure_result(
            args.solver,
            args.problem,
            f"run_one: uncaught {type(e).__name__}: {e}",
        )

    args.out_pickle.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pickle.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
