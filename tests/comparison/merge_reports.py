"""Merge multiple ``results.csv`` files into a single report.

Used because aligator and CSQP can't load in the same Python process
(competing pinocchio/eigenpy ABIs — see
``tests/comparison/aligator_install.md``). The recipe is:

1. Run ``run_benchmark.py`` once with ``LIPA_DISABLE_ALIGATOR=1`` and
   solvers ``lipa,ipopt,csqp,acados`` -> ``out_dir_a/results.csv``.
2. Run it again (no env var) with solvers ``aligator`` ->
   ``out_dir_b/results.csv``. (LIPA/IPOPT/acados are also available
   in this pass; we just don't re-run them.)
3. ``python -m tests.comparison.merge_reports out_dir_a/results.csv
   out_dir_b/results.csv --out-dir comparison_results/`` produces the
   merged markdown + CSV + plot bundle.

The merge step trusts the union of rows; if the same (problem, solver)
appears twice, the later file's row wins (so re-runs of one solver are
straightforward).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tests.comparison.problem_spec import SolverResult
from tests.comparison.report import (
    KKT_CSV_FIELDS,
    render_convergence_plots,
    render_markdown,
    write_csv,
)


def _read_csv(path: Path) -> list[SolverResult]:
    out: list[SolverResult] = []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        result = SolverResult(
            solver_name=r["solver"],
            problem_name=r["problem"],
            iterations=int(r["iterations"]),
            solve_time_ms=float(r["solve_time_ms"]),
            final_cost=float(r["final_cost"]),
            eq_violation_inf=float(r["eq_violation_inf"]),
            ineq_violation_inf=float(r["ineq_violation_inf"]),
            success=bool(int(r["success"])),
            notes=r.get("notes", ""),
        )
        # KKT columns are optional in the input CSV — older files won't
        # have them. An empty cell means "the adapter didn't extract
        # that piece" so we leave the SolverResult slot at None.
        for name in KKT_CSV_FIELDS:
            val = r.get(name, "")
            if val and val.strip():
                setattr(result, name, float(val))
        out.append(result)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "csvs",
        nargs="+",
        type=Path,
        help="One or more results.csv files to merge.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    results: dict[tuple[str, str], SolverResult] = {}
    for path in args.csvs:
        for r in _read_csv(path):
            results[(r.problem_name, r.solver_name)] = r

    merged = list(results.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.md").write_text(render_markdown(merged))
    write_csv(merged, args.out_dir / "results.csv")
    render_convergence_plots(merged, args.out_dir / "plots")
    print(f"Merged {len(merged)} (problem, solver) rows -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
