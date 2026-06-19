"""Report generation: markdown table, CSV dump, convergence plots."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from tests.comparison.problem_spec import SolverResult

# KKT-residual SolverResult attribute names emitted as CSV columns by
# ``write_csv`` and re-parsed by ``merge_reports._read_csv``. The CSV
# column name matches the attribute name. Drive both sides from this
# tuple so adding a new KKT field doesn't require remembering to update
# multiple call sites.
KKT_CSV_FIELDS: tuple[str, ...] = (
    "kkt_init_violation_inf",
    "kkt_dyn_violation_inf",
    "kkt_eq_violation_inf",
    "kkt_ineq_violation_inf",
    "kkt_dual_violation_inf",
    "kkt_complementarity_inf",
    "kkt_stationarity_inf",
    "kkt_residual_inf",
)


def _fmt_optional_float(v, fmt: str = "{:.2e}") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and v != v:  # NaN
        return "-"
    return fmt.format(v)


def render_markdown(results: list[SolverResult]) -> str:
    """One table per problem, plus an across-problem summary.

    The per-problem table has both the legacy primal-violation columns
    (``|eq|_inf`` / ``|max(0,ineq)|_inf``) and the new KKT-residual
    breakdown columns: ``init`` / ``dyn`` / ``eq*`` / ``ineq*`` (the
    last two are the user-eq / user-ineq pieces — ``init`` and ``dyn``
    are the constraint defects coming from the multi-shooting layout)
    plus ``dual`` / ``comp`` / ``stat`` / ``KKT`` (joint max). Solvers
    that did not extract multipliers leave ``dual`` / ``comp`` / ``stat``
    / ``KKT`` blank.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    by_problem: dict[str, list[SolverResult]] = {}
    for r in results:
        by_problem.setdefault(r.problem_name, []).append(r)

    out = ["# LIPA solver comparison\n"]

    for problem, rows in by_problem.items():
        out.append(f"## {problem}\n")
        headers = [
            "solver",
            "iters",
            "time [ms]",
            "final cost",
            "|eq|_inf",
            "|max(0,ineq)|_inf",
            "kkt:init",
            "kkt:dyn",
            "kkt:eq*",
            "kkt:ineq*",
            "kkt:dual",
            "kkt:comp",
            "kkt:stat",
            "KKT",
            "ok",
            "notes",
        ]
        body = []
        for r in rows:
            body.append(
                [
                    r.solver_name,
                    r.iterations,
                    f"{r.solve_time_ms:.1f}",
                    f"{r.final_cost:.4g}",
                    f"{r.eq_violation_inf:.2e}",
                    f"{r.ineq_violation_inf:.2e}",
                    _fmt_optional_float(r.kkt_init_violation_inf),
                    _fmt_optional_float(r.kkt_dyn_violation_inf),
                    _fmt_optional_float(r.kkt_eq_violation_inf),
                    _fmt_optional_float(r.kkt_ineq_violation_inf),
                    _fmt_optional_float(r.kkt_dual_violation_inf),
                    _fmt_optional_float(r.kkt_complementarity_inf),
                    _fmt_optional_float(r.kkt_stationarity_inf),
                    _fmt_optional_float(r.kkt_residual_inf),
                    "ok" if r.success else "x",
                    (r.notes[:140] + "...")
                    if r.notes and len(r.notes) > 140
                    else (r.notes or ""),
                ]
            )
        if tabulate is not None:
            out.append(tabulate(body, headers=headers, tablefmt="github"))
        else:
            out.append(" | ".join(headers))
            for row in body:
                out.append(" | ".join(str(c) for c in row))
        out.append("\n")

    # Cross-problem summary: rows = solver, cols = problem, cell encodes status:
    #   ok       converged at tol
    #   x        ran and failed (didn't converge / iter cap / numerical)
    #   skip     pre-skipped via known_timeouts.py
    #   N/A      adapter structurally doesn't support this problem
    #            (returns success=False with a self-describing notes message)
    #   gated    intentional skip (e.g. lipa-cpu on MJX when
    #            RUN_LIPA_CPU_ON_MJX_PROBLEMS=0; row missing entirely)
    solvers = sorted({r.solver_name for r in results})
    problems = sorted(by_problem.keys())
    summary_headers = ["solver", *problems]
    summary_body = []

    def _cell_status(r: SolverResult | None) -> str:
        if r is None:
            return "gated"
        if r.success:
            return f"{r.iterations} ok"
        n = (r.notes or "").lower()
        if "hits process timeout" in n:
            return "skip"
        if (
            "does not natively support" in n
            or "does not support" in n
            or "mjx-only" in n
        ):
            return "N/A"
        return f"{r.iterations} x"

    for solver in solvers:
        row = [solver]
        for p in problems:
            r = next((rr for rr in by_problem[p] if rr.solver_name == solver), None)
            row.append(_cell_status(r))
        summary_body.append(row)
    out.append("## Summary: iterations + status\n")
    if tabulate is not None:
        out.append(tabulate(summary_body, headers=summary_headers, tablefmt="github"))
    else:
        out.append(" | ".join(summary_headers))
        for row in summary_body:
            out.append(" | ".join(str(c) for c in row))
    out.append("\n")

    # Cross-problem runtime summary — same shape, but each cell is the
    # solver's wall-clock time in milliseconds. Each adapter's
    # `solve_time_ms` excludes JAX JIT compilation, CasADi code-gen, and
    # acados / aligator one-time setup costs (a warm-up call is made
    # before the timed solve in every adapter).
    runtime_body = []
    for solver in solvers:
        row = [solver]
        for p in problems:
            r = next((rr for rr in by_problem[p] if rr.solver_name == solver), None)
            if r is None or r.iterations == 0:
                row.append("-")
            else:
                tag = "ok" if r.success else "x"
                row.append(f"{r.solve_time_ms:.0f} ms {tag}")
        runtime_body.append(row)
    out.append(
        "## Summary: wall-clock time (excludes JIT / codegen / one-time setup)\n"
    )
    if tabulate is not None:
        out.append(tabulate(runtime_body, headers=summary_headers, tablefmt="github"))
    else:
        out.append(" | ".join(summary_headers))
        for row in runtime_body:
            out.append(" | ".join(str(c) for c in row))
    out.append("\n")

    # Cross-problem KKT-residual summary: rows = solver, cols = problem,
    # each cell is the joint KKT residual (max of init / dyn / eq /
    # ineq / dual / comp / stat). Solvers that didn't extract
    # multipliers leave this blank.
    kkt_body = []
    for solver in solvers:
        row = [solver]
        for p in problems:
            r = next((rr for rr in by_problem[p] if rr.solver_name == solver), None)
            if r is None or r.iterations == 0:
                row.append("-")
            else:
                row.append(_fmt_optional_float(r.kkt_residual_inf))
        kkt_body.append(row)
    out.append(
        "## Summary: joint KKT residual (max of init / dyn / eq / ineq / dual / comp / stat)\n"
    )
    if tabulate is not None:
        out.append(tabulate(kkt_body, headers=summary_headers, tablefmt="github"))
    else:
        out.append(" | ".join(summary_headers))
        for row in kkt_body:
            out.append(" | ".join(str(c) for c in row))
    out.append("\n")

    return "\n".join(out)


def write_csv(results: Iterable[SolverResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        # KKT columns: empty cells = the adapter didn't extract
        # multipliers / couldn't compute that piece.
        w.writerow(
            [
                "problem",
                "solver",
                "iterations",
                "solve_time_ms",
                "final_cost",
                "eq_violation_inf",
                "ineq_violation_inf",
                "success",
                *KKT_CSV_FIELDS,
                "notes",
            ]
        )
        for r in results:
            base = [
                r.problem_name,
                r.solver_name,
                r.iterations,
                r.solve_time_ms,
                r.final_cost,
                r.eq_violation_inf,
                r.ineq_violation_inf,
                int(r.success),
            ]
            kkt_cells = [
                "" if getattr(r, name) is None else getattr(r, name)
                for name in KKT_CSV_FIELDS
            ]
            w.writerow([*base, *kkt_cells, r.notes])


def write_solution_archives(results: Iterable[SolverResult], out_dir: Path) -> None:
    """Persist final solver iterates as one compressed NPZ per result.

    This is intentionally separate from CSV/report generation so the default
    benchmark artifacts stay lightweight. Missing iterates (for unsupported
    pairs, hard kills, or adapters that did not return a trajectory) are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _clean(s: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

    for r in results:
        if r.X is None or r.U is None or r.Theta is None:
            continue
        path = out_dir / f"{_clean(r.problem_name)}__{_clean(r.solver_name)}.npz"
        warm_start_arrays = {}
        for key, value in (r.warm_start_out or {}).items():
            if value is None:
                continue
            warm_start_arrays[f"warm_start__{key}"] = np.asarray(value)
        np.savez_compressed(
            path,
            X=np.asarray(r.X),
            U=np.asarray(r.U),
            Theta=np.asarray(r.Theta),
            solver_name=np.asarray(r.solver_name),
            problem_name=np.asarray(r.problem_name),
            iterations=np.asarray(r.iterations),
            solve_time_ms=np.asarray(r.solve_time_ms),
            final_cost=np.asarray(r.final_cost),
            eq_violation_inf=np.asarray(r.eq_violation_inf),
            ineq_violation_inf=np.asarray(r.ineq_violation_inf),
            success=np.asarray(r.success),
            notes=np.asarray(r.notes),
            **warm_start_arrays,
        )


def render_convergence_plots(results: list[SolverResult], out_dir: Path) -> None:
    """One PNG per problem with cost / eq / ineq vs. iteration for each solver.

    Only solvers that recorded ``cost_history`` (or one of the violation
    histories) get a curve. Solvers that recorded none of these three
    contribute a single endpoint marker so you can still see the final
    iterate alongside the curves.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_problem: dict[str, list[SolverResult]] = {}
    for r in results:
        by_problem.setdefault(r.problem_name, []).append(r)

    def _safe_floor(v, floor=1e-16):
        """Return a strictly-positive value plottable on a log scale."""
        if v is None:
            return floor
        v = float(v)
        if not np.isfinite(v):
            return floor
        return max(abs(v), floor)

    def _safe_hist(arr, floor=1e-16):
        """Replace non-finite entries in a history array with the floor."""
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.where(np.isfinite(arr), np.abs(arr), floor)
        return np.maximum(arr, floor)

    for problem, rows in by_problem.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax_cost, ax_eq, ax_ineq = axes
        plotted_any_cost = plotted_any_eq = plotted_any_ineq = False
        for r in rows:
            label = r.solver_name
            # 1-based iter indexing so log-x doesn't choke on iter 0.
            if r.cost_history is not None and len(r.cost_history) > 0:
                hist = _safe_hist(r.cost_history)
                ax_cost.plot(np.arange(1, len(hist) + 1), hist, label=label)
                plotted_any_cost = True
            else:
                ax_cost.scatter(
                    [max(r.iterations, 1)],
                    [_safe_floor(r.final_cost)],
                    label=f"{label} (endpoint)",
                )
                plotted_any_cost = True
            if r.eq_violation_history is not None and len(r.eq_violation_history) > 0:
                hist = _safe_hist(r.eq_violation_history)
                ax_eq.plot(np.arange(1, len(hist) + 1), hist, label=label)
                plotted_any_eq = True
            else:
                ax_eq.scatter(
                    [max(r.iterations, 1)],
                    [_safe_floor(r.eq_violation_inf)],
                    label=f"{label} (endpoint)",
                )
                plotted_any_eq = True
            if (
                r.ineq_violation_history is not None
                and len(r.ineq_violation_history) > 0
            ):
                hist = _safe_hist(r.ineq_violation_history)
                ax_ineq.plot(np.arange(1, len(hist) + 1), hist, label=label)
                plotted_any_ineq = True
            else:
                ax_ineq.scatter(
                    [max(r.iterations, 1)],
                    [_safe_floor(r.ineq_violation_inf)],
                    label=f"{label} (endpoint)",
                )
                plotted_any_ineq = True
        ax_cost.set_title(f"{problem}: cost")
        ax_cost.set_xlabel("iter (log)")
        ax_cost.set_ylabel("|cost|")
        ax_eq.set_title(f"{problem}: eq violation")
        ax_eq.set_xlabel("iter (log)")
        ax_eq.set_ylabel("|eq|_inf")
        ax_ineq.set_title(f"{problem}: ineq violation")
        ax_ineq.set_xlabel("iter (log)")
        ax_ineq.set_ylabel("max(0,ineq)_inf")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            ax.grid(True, which="both", alpha=0.3)
        # When every solver on this problem failed (all values were NaN
        # and got floored, OR no data was plotted), tight_layout chokes
        # on the log-scale axes. Save anyway with a defensive try/except
        # so plot generation never aborts the merge step.
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):
            pass  # log-scale axis with non-positive data; skip layout pass
        try:
            fig.savefig(out_dir / f"{problem}_convergence.png", dpi=120)
        except (ValueError, RuntimeError):
            pass
        plt.close(fig)
