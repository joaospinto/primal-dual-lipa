"""Standalone CLI runner for the LIPA-vs-other-solvers comparison.

Examples:

    # Run cartpole with LIPA + IPOPT
    uv run --extra test --extra mpc-examples --group comparisons \\
        python -m tests.comparison.run_benchmark \\
        --problems cartpole --solvers lipa,ipopt \\
        --out-dir comparison_results/

    # Run all MJX-free problems with all available solvers
    uv run ... --problems cartpole,acrobot,quadpendulum

    # Run barrel_roll through LIPA + IPOPT (the only MJX-capable solvers)
    uv run ... --problems barrel_roll --solvers lipa,ipopt --max-iter 200
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import os
import pickle
import pty
import select
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from tests.comparison.adapters import all_adapter_names, get_adapter
from tests.comparison.known_timeouts import is_known_hard_kill
from tests.comparison.problem_spec import SolverResult, make_failure_result
from tests.comparison.problems import all_problem_names, make_problem
from tests.comparison.report import (
    render_convergence_plots,
    render_markdown,
    write_solution_archives,
    write_csv,
)


def _run_one_in_process(
    solver_name: str,
    problem_name: str,
    *,
    max_iter: int,
    tol: float,
    timeout_s: float | None,
    verbose: bool,
    solver_verbose: bool = False,
    backend: str | None = None,
    extra_kwargs: dict | None = None,
    problem_kwargs: dict | None = None,
    metadata_overlay: dict | None = None,
    initial_solution_npz: str | Path | None = None,
) -> SolverResult:
    """Build the problem, instantiate the adapter, call ``solve``, in-process.

    Used both directly (when ``--hard-timeout-s`` is unset) and as the
    body of ``tests.comparison.run_one`` (the subprocess entry point).
    """
    if verbose:
        print(f"  -> {solver_name} on {problem_name} ...", flush=True)
    try:
        problem = make_problem(problem_name, **(problem_kwargs or {}))
        if metadata_overlay:
            problem = dataclasses.replace(
                problem,
                metadata={**problem.metadata, **metadata_overlay},
            )
        if initial_solution_npz is not None:
            problem = _apply_initial_solution_npz(problem, initial_solution_npz)
    except Exception as e:  # noqa: BLE001
        return make_failure_result(
            solver_name,
            problem_name,
            f"problem build failed: {e}",
        )

    # Per-problem max_iter override: lets a problem cap a solver that
    # has no native time cap and provably won't converge here (e.g.
    # csqp on quadpendulum hits max_iter=1000 in 16 min and fails).
    effective_max_iter = max_iter
    overrides = problem.metadata.get("max_iter_overrides", {})
    if solver_name in overrides:
        effective_max_iter = min(max_iter, int(overrides[solver_name]))
        if verbose and effective_max_iter != max_iter:
            print(
                f"     (max_iter capped at {effective_max_iter} for "
                f"{solver_name} via problem.metadata override; "
                f"CLI was {max_iter})",
                flush=True,
            )

    # Per-solver kwargs for the adapter constructor.
    kwargs: dict = {"max_iter": effective_max_iter}
    if solver_name in {"ipopt-casadi", "ipopt-jax", "csqp", "aligator", "acados"}:
        kwargs["tol"] = tol
    if solver_name in {"ipopt-casadi", "ipopt-jax", "sip"}:
        kwargs["timeout_s"] = timeout_s
    if solver_name in {"ipopt-casadi", "ipopt-jax"}:
        if solver_verbose:
            kwargs["print_level"] = 5
    if solver_name == "csqp" and solver_verbose:
        kwargs["with_callbacks"] = True
    if solver_name == "sip" and solver_verbose:
        kwargs["print_logs"] = True
    # Backend selector for adapters that have a backend knob (csqp,
    # aligator, sip). Other adapters either are CasADi-only (ipopt,
    # acados, fatrop, ipopt-mjx, fatrop-mjx) or JAX-only (lipa, sip-mjx,
    # trajax) and don't accept the kwarg — TypeError fallback below
    # handles that gracefully.
    if backend is not None and solver_name in {"csqp", "aligator", "sip"}:
        kwargs["backend"] = backend

    # Free-form per-solver extras from --solver-kwargs-json. Used for
    # tuning sweeps where we want to drive adapter-specific parameters
    # (e.g. ipopt_extra_options, sip_extra_settings)
    # without burning a round-trip on the adapter's signature.
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        adapter = get_adapter(solver_name, **kwargs)
    except TypeError:
        # adapter doesn't accept tol / timeout / backend — strip them and
        # retry with just max_iter so we don't crash for older adapter
        # signatures or unrecognized kwargs.
        # Preserve extra_kwargs across the retry so tuning sweeps don't
        # silently lose their settings just because the adapter doesn't
        # take ``tol``.
        retry_kwargs = {"max_iter": effective_max_iter}
        if extra_kwargs:
            retry_kwargs.update(extra_kwargs)
        adapter = get_adapter(solver_name, **retry_kwargs)

    avail, reason = adapter.is_available()
    if not avail:
        return make_failure_result(
            solver_name,
            problem_name,
            f"unavailable: {reason}",
        )

    try:
        # Per-(problem, solver) two-phase opt-in. Each problem declares
        # which solvers want two-phase via the flat per-solver flag
        # ``<solver_root>_two_phase: True`` in metadata. Solvers whose
        # algorithm doesn't fit the LIPA-shaped Phase-1 / Phase-2 split
        # (everything other than LIPA today) just don't set the flag.
        two_phase_key = f"{_solver_root(solver_name)}_two_phase"
        # Dev-only override: ``LIPA_DEV_SKIP_WARMUP_PHASE=1`` short-circuits
        # two-phase to single-phase, used together with the warm-cache
        # so the debug solve only runs the final phase from the cached
        # near-converged iterate (Phase 1 would otherwise waste iters in
        # an independent run from the same cache that gets thrown away).
        skip_warmup = bool(os.environ.get("LIPA_DEV_SKIP_WARMUP_PHASE"))
        if problem.metadata.get(two_phase_key, False) and not skip_warmup:
            result = _solve_two_phase(adapter, problem, solver_name, verbose)
        else:
            result = adapter.solve(problem)
    except Exception as e:  # noqa: BLE001
        return make_failure_result(
            solver_name,
            problem_name,
            f"adapter raised: {type(e).__name__}: {e}",
        )

    if verbose:
        _print_pair_outcome(result)
    return result


def _solver_root(solver_name: str) -> str:
    """Strip backend / platform suffixes so per-solver metadata keys are
    addressable from the user-visible label.

    ``lipa-cpu`` / ``lipa-gpu`` -> ``lipa``; ``ipopt-casadi`` /
    ``ipopt-jax`` -> ``ipopt_mjx`` for the jax variant since its
    metadata key is ``ipopt_mjx_*``, otherwise ``ipopt``;
    ``fatrop-jax`` -> ``fatrop_mjx``; ``fatrop-casadi`` -> ``fatrop``;
    ``sip-jax`` / ``sip-casadi`` -> ``sip``; same shape for csqp /
    aligator.
    """
    if solver_name == "ipopt-jax":
        return "ipopt_mjx"
    if solver_name == "fatrop-jax":
        return "fatrop_mjx"
    return solver_name.rsplit("-", 1)[0] if "-" in solver_name else solver_name


def _apply_initial_solution_npz(problem, archive_path: str | Path):
    """Replace a problem's warm start from a saved solution archive."""
    path = Path(archive_path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in ("X", "U", "Theta") if key not in data]
        if missing:
            raise ValueError(f"{path} missing required arrays: {missing}")
        X = np.asarray(data["X"])
        U = np.asarray(data["U"])
        Theta = np.asarray(data["Theta"])
        warm_start = {
            key.removeprefix("warm_start__"): np.asarray(data[key])
            for key in data.files
            if key.startswith("warm_start__")
        }

    expected = {
        "X": problem.X_init.shape,
        "U": problem.U_init.shape,
        "Theta": problem.Theta_init.shape,
    }
    actual = {"X": X.shape, "U": U.shape, "Theta": Theta.shape}
    mismatched = {
        key: (actual[key], expected[key])
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatched:
        raise ValueError(f"{path} shape mismatch for {problem.name}: {mismatched}")

    return dataclasses.replace(
        problem,
        X_init=X,
        U_init=U,
        Theta_init=Theta,
        warm_start=warm_start or problem.warm_start,
    )


def _overlay_settings(base, overlay):
    """Overlay partial dict settings while preserving structured settings values."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        return {**base, **overlay}
    return overlay


def _build_warmup_spec(problem, solver_name):
    """Construct the Phase-1 ProblemSpec from a problem that opted into
    two-phase via ``metadata['<solver_root>_two_phase'] = True``.

    Phase 1 uses ``metadata['warmup_cost']`` as the cost and drops
    inequalities entirely (soft-penalty form). Per-solver Phase-1
    schedules come from ``metadata[<solver>_warmup_settings]`` (or
    ``ipopt_mjx_warmup_extra_options`` for ipopt-jax), shadowing the
    matching main-phase key for the adapter's read.
    """
    import dataclasses

    warmup_cost = problem.metadata.get("warmup_cost")
    if warmup_cost is None:
        return None

    new_metadata = dict(problem.metadata)
    new_metadata.pop("warmup_cost", None)

    # Per-solver schedule swap: <solver>_warmup_settings shadows
    # <solver>_settings for the warmup pass.
    root = _solver_root(solver_name)
    swap_pairs = [
        (f"{root}_warmup_settings", f"{root}_settings"),
        # ipopt-mjx uses ``_extra_options`` instead of ``_settings``.
        (f"{root}_warmup_extra_options", f"{root}_extra_options"),
    ]
    for warmup_key, main_key in swap_pairs:
        if warmup_key in new_metadata:
            new_metadata[main_key] = _overlay_settings(
                new_metadata.get(main_key),
                new_metadata[warmup_key],
            )

    return dataclasses.replace(
        problem,
        cost=warmup_cost,
        inequalities=None,
        ineq_dim=0,
        metadata=new_metadata,
    )


def _solve_two_phase(adapter, problem, solver_name, verbose):
    """Run Phase-1 warmup then Phase-2 main, return the merged
    SolverResult (Phase-2 numbers primary, iters + time merged)."""
    import dataclasses

    warmup_spec = _build_warmup_spec(problem, solver_name)
    if warmup_spec is None:
        return adapter.solve(problem)
    if verbose:
        print("     [phase 1/2] warmup ...", flush=True)
    p1 = adapter.solve(warmup_spec)
    if p1.X is None or p1.U is None:
        if verbose:
            print(
                f"     [phase 1/2] exposed no iterate; running single-phase: "
                f"{p1.notes or 'no iterate exposed'}",
                flush=True,
            )
        return adapter.solve(problem)
    if not p1.success and verbose:
        print(
            f"     [phase 1/2] did not meet final success gate; "
            "continuing with exposed warm-start iterate ...",
            flush=True,
        )
    if verbose:
        print(
            f"     [phase 1/2] done in {p1.iterations} iters / "
            f"{p1.solve_time_ms:.0f} ms; [phase 2/2] main solve ...",
            flush=True,
        )
    # Carry Phase-1's iterate AND any adapter-specific extra state
    # (multipliers, slacks, penalty arrays — anything the adapter dumped via
    # SolverResult.warm_start_out) into Phase 2.
    main_spec = dataclasses.replace(
        problem,
        X_init=p1.X,
        U_init=p1.U,
        Theta_init=p1.Theta,
        warm_start=p1.warm_start_out,
    )
    p2 = adapter.solve(main_spec)
    # Merge iters + time additively; everything else from Phase 2.
    p2.iterations = int(p1.iterations) + int(p2.iterations)
    p2.solve_time_ms = float(p1.solve_time_ms) + float(p2.solve_time_ms)
    if p2.notes:
        p2.notes = f"two-phase warm start; {p2.notes}"
    else:
        p2.notes = "two-phase warm start"
    return p2


def _print_pair_outcome(result: SolverResult) -> None:
    # Early-return rows (problem rejected, build failure, hard-kill,
    # adapter raised) all surface as iters=0, success=False, with the
    # reason in ``notes``. The numeric breakdown would be all nan/0 in
    # that case, which reads as "something went wrong" even when the
    # adapter just structurally doesn't support this problem. Print the
    # notes instead so the reason is the first thing the reader sees.
    if result.iterations == 0 and not result.success:
        print(f"     {result.notes or 'failed (no notes)'}", flush=True)
        return
    print(
        f"     iters={result.iterations} time={result.solve_time_ms:.0f}ms "
        f"cost={result.final_cost:.4g} eq={result.eq_violation_inf:.2e} "
        f"ineq={result.ineq_violation_inf:.2e} ok={result.success}",
        flush=True,
    )


def _subprocess_env(problem_name: str) -> dict[str, str] | None:
    """Mirror the benchmark scripts' stable XLA-CPU flag for MJX runs."""
    if (
        os.environ.get("JAX_PLATFORMS", "").lower() != "cpu"
        or problem_name not in {"trot", "barrel_roll", "backflip", "jump"}
    ):
        return None

    xla_flags = os.environ.get("XLA_FLAGS", "")
    cpu_flag = "--xla_cpu_multi_thread_eigen=false"
    needs_xla_flag = cpu_flag not in xla_flags.split()
    if not needs_xla_flag:
        return None

    child_env = os.environ.copy()
    if needs_xla_flag:
        child_env["XLA_FLAGS"] = f"{xla_flags} {cpu_flag}".strip()
    return child_env


def _run_one_via_subprocess(
    solver_name: str,
    problem_name: str,
    *,
    max_iter: int,
    tol: float,
    timeout_s: float | None,
    hard_timeout_s: float,
    verbose: bool,
    solver_verbose: bool = False,
    backend: str | None = None,
    extra_kwargs: dict | None = None,
    log_dir: Path | None = None,
    problem_kwargs: dict | None = None,
    metadata_overlay: dict | None = None,
    initial_solution_npz: str | Path | None = None,
) -> SolverResult:
    """Run one (solver, problem) pair in its own subprocess with a hard kill.

    Two-tier timeout philosophy:
      * ``timeout_s`` — passed to the adapter as a *soft* cap. Solvers
        with native wall-time support (IPOPT, IPOPT-MJX, SIP) honor this
        and return a clean ``SolverResult`` with a timeout-style status.
        Most solvers ignore it.
      * ``hard_timeout_s`` — wall time we give the subprocess as a
        whole, including JIT compile + problem build + solver run. When
        exceeded the child is SIGKILLed and we synthesise a failure
        result. Should be set noticeably larger than ``timeout_s`` so
        a solver that DOES honor the soft cap has time to exit
        cleanly first, and so JIT compilation completes.

    On clean exit: the child writes a pickled ``SolverResult`` to a
    temp file and exits 0; this function reads it back.

    On hard kill: returns ``make_failure_result(..., notes="hard-killed
    after Ns")`` with ``solve_time_ms`` reflecting actual wall time.
    """
    if verbose:
        print(f"  -> {solver_name} on {problem_name} ...", flush=True)

    # Use a NamedTemporaryFile path-only — the child opens and writes
    # it. delete=False because we re-open it in the parent after the
    # child exits; we unlink it ourselves in the finally block.
    out_pkl_handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".pkl",
        prefix=f"runone_{solver_name}_{problem_name}_",
        delete=False,
    )
    out_pkl_handle.close()
    out_pkl = Path(out_pkl_handle.name)

    cmd = [
        sys.executable,
        "-m",
        "tests.comparison.run_one",
        "--solver",
        solver_name,
        "--problem",
        problem_name,
        "--max-iter",
        str(max_iter),
        "--tol",
        str(tol),
        "--out-pickle",
        str(out_pkl),
    ]
    if timeout_s is not None:
        cmd += ["--timeout-s", str(timeout_s)]
    if backend is not None:
        cmd += ["--backend", backend]
    if solver_verbose:
        cmd += ["--solver-verbose"]
    if problem_kwargs:
        cmd += ["--problem-kwargs-json", json.dumps(problem_kwargs)]
    if metadata_overlay:
        cmd += ["--metadata-json", json.dumps(metadata_overlay)]
    if initial_solution_npz is not None:
        cmd += ["--initial-solution-npz", str(initial_solution_npz)]
    if extra_kwargs:
        cmd += ["--solver-kwargs-json", json.dumps(extra_kwargs)]

    if log_dir is None:
        log_dir = Path(tempfile.gettempdir()) / "lipa_comparison_subprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{out_pkl.stem}.log"
    child_env = _subprocess_env(problem_name)

    start = timer()
    master_fd, slave_fd = pty.openpty()
    timed_out = False
    completed_returncode: int | None = None
    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            env=child_env,
        )
        os.close(slave_fd)
        try:
            while True:
                remaining = hard_timeout_s - (timer() - start)
                if remaining <= 0.0:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    break
                rlist, _, _ = select.select([master_fd], [], [], min(0.25, remaining))
                if master_fd in rlist:
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError as e:
                        if e.errno != errno.EIO:
                            raise
                        chunk = b""
                    if chunk:
                        log_f.write(chunk)
                        log_f.flush()
                        if verbose:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                    elif proc.poll() is not None:
                        break
                if proc.poll() is not None:
                    while True:
                        rlist, _, _ = select.select([master_fd], [], [], 0.0)
                        if master_fd not in rlist:
                            break
                        try:
                            chunk = os.read(master_fd, 65536)
                        except OSError as e:
                            if e.errno != errno.EIO:
                                raise
                            break
                        if not chunk:
                            break
                        log_f.write(chunk)
                        log_f.flush()
                        if verbose:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                    break
            completed_returncode = proc.wait()
        finally:
            os.close(master_fd)
    elapsed_ms = 1e3 * (timer() - start)
    if timed_out:
        result = make_failure_result(
            solver_name,
            problem_name,
            "hard-killed after "
            f"{hard_timeout_s:.0f}s (subprocess timeout); log={log_path}",
            solve_time_ms=elapsed_ms,
        )
        if verbose:
            print(
                f"     HARD-KILLED after {elapsed_ms:.0f}ms "
                f"(budget {hard_timeout_s:.0f}s, log={log_path})",
                flush=True,
            )
        out_pkl.unlink(missing_ok=True)
        return result
    completed = subprocess.CompletedProcess(cmd, completed_returncode)

    try:
        if completed.returncode != 0 or out_pkl.stat().st_size == 0:
            return make_failure_result(
                solver_name,
                problem_name,
                f"subprocess exit={completed.returncode}; no result written "
                f"(after {elapsed_ms:.0f}ms); log={log_path}",
                solve_time_ms=elapsed_ms,
            )
        with out_pkl.open("rb") as f:
            result: SolverResult = pickle.load(f)  # noqa: S301
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        return make_failure_result(
            solver_name,
            problem_name,
            f"subprocess result unreadable: {type(e).__name__}: {e}; log={log_path}",
            solve_time_ms=elapsed_ms,
        )
    finally:
        out_pkl.unlink(missing_ok=True)

    if verbose:
        _print_pair_outcome(result)
    return result


def _run_one(
    solver_name: str,
    problem_name: str,
    *,
    max_iter: int,
    tol: float,
    timeout_s: float | None,
    hard_timeout_s: float | None,
    verbose: bool,
    solver_verbose: bool = False,
    backend: str | None = None,
    extra_kwargs: dict | None = None,
    log_dir: Path | None = None,
    problem_kwargs: dict | None = None,
    metadata_overlay: dict | None = None,
    initial_solution_npz: str | Path | None = None,
) -> SolverResult:
    """Dispatch to subprocess (with hard kill) or in-process."""
    if hard_timeout_s is not None:
        return _run_one_via_subprocess(
            solver_name,
            problem_name,
            max_iter=max_iter,
            tol=tol,
            timeout_s=timeout_s,
            hard_timeout_s=hard_timeout_s,
            verbose=verbose,
            solver_verbose=solver_verbose,
            backend=backend,
            extra_kwargs=extra_kwargs,
            log_dir=log_dir,
            problem_kwargs=problem_kwargs,
            metadata_overlay=metadata_overlay,
            initial_solution_npz=initial_solution_npz,
        )
    return _run_one_in_process(
        solver_name,
        problem_name,
        max_iter=max_iter,
        tol=tol,
        timeout_s=timeout_s,
        verbose=verbose,
        solver_verbose=solver_verbose,
        backend=backend,
        extra_kwargs=extra_kwargs,
        problem_kwargs=problem_kwargs,
        metadata_overlay=metadata_overlay,
        initial_solution_npz=initial_solution_npz,
    )


def _parse_csv(s: str | None, default: list[str]) -> list[str]:
    if s is None:
        return default
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_json_arg(value: str | None, flag_name: str) -> dict:
    if not value:
        return {}
    candidate = Path(value)
    try:
        if not value.lstrip().startswith(("{", "[")) and candidate.is_file():
            decoded = json.loads(candidate.read_text())
        else:
            decoded = json.loads(value)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"{flag_name} failed to parse: {e}") from e
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{flag_name} must decode to a dict, got {type(decoded).__name__}",
        )
    return decoded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help=f"Comma-separated problem names. Default: all known. Choices: {','.join(all_problem_names())}",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help=f"Comma-separated solver names. Default: all registered. Choices: {','.join(all_adapter_names())}",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Soft per-solve wall-clock cap, passed to the adapter. "
        "IPOPT / IPOPT-MJX honor it natively via max_wall_time; SIP "
        "honors it via sip_python's internal time_limit_s; "
        "other adapters ignore it. Use --hard-timeout-s as a backstop.",
    )
    parser.add_argument(
        "--hard-timeout-s",
        type=float,
        default=None,
        help="Hard wall-clock cap enforced by running each (solver, problem) "
        "pair in its own subprocess and SIGKILLing it when this budget "
        "is exceeded. Covers JIT compile + problem build + solver run, "
        "so set it noticeably larger than --timeout-s. If unset, all "
        "pairs run in-process and are bounded only by --max-iter and "
        "any solver-native cap. Killed pairs land in the report as "
        "success=False, notes='hard-killed after Ns'.",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Pass print_level=5 to IPOPT and with_callbacks=True to CSQP "
        "so each solver emits its own per-iter log line.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["jax", "casadi"],
        help="Per-solver backend selector (only csqp / aligator / sip "
        "accept this knob; ignored for other solvers). 'jax' "
        "preserves each adapter's historical default; 'casadi' "
        "uses casadi.Function-backed callbacks (faster on "
        "analytical problems, unavailable for MJX problems "
        "which have no casadi_builder). If omitted, each "
        "adapter's own default is used.",
    )
    parser.add_argument(
        "--solver-kwargs-json",
        type=str,
        default=None,
        help="JSON dict (or path to a JSON file) of extra kwargs to "
        "forward to EVERY solver adapter constructor. Used by "
        "tuning sweeps to inject adapter-specific knobs like "
        "ipopt_extra_options or sip_extra_settings. "
        "Adapters that don't accept the kwarg will silently drop "
        "it via the existing TypeError fallback. Example: "
        "--solver-kwargs-json "
        '\'{"sip_extra_settings": {"line_search": {"max_iterations": 30}}}\'.',
    )
    parser.add_argument(
        "--problem-kwargs-json",
        type=str,
        default=None,
        help="JSON dict (or path to a JSON file) of extra kwargs forwarded "
        "to every selected problem factory.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="JSON dict (or path to a JSON file) overlaid onto every selected "
        "ProblemSpec.metadata after construction.",
    )
    parser.add_argument(
        "--initial-solution-npz",
        type=Path,
        default=None,
        help="Saved solution archive whose X/U/Theta arrays replace every "
        "selected problem's primal warm start. Archives from --save-solutions "
        "may also carry adapter warm-start state.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("comparison_results"))
    parser.add_argument(
        "--save-solutions",
        action="store_true",
        help="Persist final X/U/Theta iterates as compressed NPZ files under "
        "<out-dir>/solutions/. This is opt-in because the CSV/report artifacts "
        "are the stable lightweight benchmark output.",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default="",
        help="Append this string to every solver name in the emitted CSV / "
        "report. Lets two passes of the same solver (e.g. lipa on GPU "
        "vs. lipa on CPU under JAX_PLATFORMS=cpu) coexist in a merged "
        "report under distinct labels (e.g. --label-suffix '-cpu').",
    )
    parser.add_argument(
        "--ignore-known-timeouts",
        action="store_true",
        help="Run (solver, problem) pairs listed in "
        "tests/comparison/known_timeouts.py:KNOWN_HARD_KILLS even "
        "when this script would otherwise short-circuit them with "
        "'hits process timeout (cached)'. Tuning subagents pass "
        "this so they can actually iterate on the offending pair.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    problems = _parse_csv(args.problems, all_problem_names())
    solvers = _parse_csv(args.solvers, all_adapter_names())

    try:
        extra_kwargs = _parse_json_arg(
            args.solver_kwargs_json,
            "--solver-kwargs-json",
        )
        problem_kwargs = _parse_json_arg(
            args.problem_kwargs_json,
            "--problem-kwargs-json",
        )
        metadata_overlay = _parse_json_arg(args.metadata_json, "--metadata-json")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if verbose and extra_kwargs:
        print(f"Extra solver kwargs: {extra_kwargs}")
    if verbose and problem_kwargs:
        print(f"Extra problem kwargs: {problem_kwargs}")
    if verbose and metadata_overlay:
        print(f"Metadata overlay: {metadata_overlay}")

    if verbose:
        print(f"Problems: {problems}")
        print(f"Solvers: {solvers}")
        print(
            f"max_iter={args.max_iter} tol={args.tol} "
            f"timeout_s={args.timeout_s} hard_timeout_s={args.hard_timeout_s}",
        )

    results: list[SolverResult] = []
    for problem in problems:
        if verbose:
            print(f"\n=== {problem} ===")
        for solver in solvers:
            # Short-circuit pairs we already know hard-kill, unless the
            # caller asked us to retry. We compose the post-label name
            # here because the registry is keyed by the user-visible
            # label (sip-jax / lipa-gpu / aligator-jax / etc.), not
            # the raw adapter name. Pair still appears in the report;
            # we just skip the doomed subprocess spawn.
            solver_label = solver + args.label_suffix
            if not args.ignore_known_timeouts and is_known_hard_kill(
                solver_label,
                problem,
            ):
                if verbose:
                    print(f"  -> {solver} on {problem} ...", flush=True)
                r = make_failure_result(
                    solver_label,
                    problem,
                    "hits process timeout (cached)",
                )
                if verbose:
                    _print_pair_outcome(r)
                results.append(r)
                continue

            r = _run_one(
                solver,
                problem,
                max_iter=args.max_iter,
                tol=args.tol,
                timeout_s=args.timeout_s,
                hard_timeout_s=args.hard_timeout_s,
                verbose=verbose,
                solver_verbose=args.solver_verbose,
                backend=args.backend,
                extra_kwargs=extra_kwargs,
                log_dir=args.out_dir / "subprocess_logs",
                problem_kwargs=problem_kwargs,
                metadata_overlay=metadata_overlay,
                initial_solution_npz=args.initial_solution_npz,
            )
            if args.label_suffix:
                r = dataclasses.replace(
                    r,
                    solver_name=r.solver_name + args.label_suffix,
                )
            results.append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(results)
    (args.out_dir / "report.md").write_text(md)
    write_csv(results, args.out_dir / "results.csv")
    if args.save_solutions:
        write_solution_archives(results, args.out_dir / "solutions")
    render_convergence_plots(results, args.out_dir / "plots")

    if verbose:
        print(f"\nWrote markdown -> {args.out_dir / 'report.md'}")
        print(f"Wrote CSV -> {args.out_dir / 'results.csv'}")
        if args.save_solutions:
            print(f"Wrote solution archives -> {args.out_dir / 'solutions'}/")
        print(f"Wrote plots -> {args.out_dir / 'plots'}/")

    # Exit nonzero if any solver reported failure (so CI-style consumers can detect).
    if any(not r.success for r in results):
        if verbose:
            print("\nSome solver/problem pairs did NOT succeed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
