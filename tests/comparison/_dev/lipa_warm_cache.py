"""Dev-only LIPA warm-cache: pickle final (vars, params) per problem
to disk so subsequent solves skip the slow "get-near-the-solution"
phase. Gated by ``LIPA_DEV_WARM_CACHE_DIR``; NOT for production use.

This module is intentionally scratch — DELETE before final merge.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional


def cache_dir() -> Optional[Path]:
    """Return the warm-cache directory if enabled, else None."""
    p = os.environ.get("LIPA_DEV_WARM_CACHE_DIR")
    if not p:
        return None
    pp = Path(p).expanduser()
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def cache_file(problem_name: str) -> Optional[Path]:
    d = cache_dir()
    if d is None:
        return None
    return d / f"{problem_name}.pkl"


def load(problem_name: str) -> Optional[dict]:
    """Load cached final state for a problem, or None if not present."""
    f = cache_file(problem_name)
    if f is None or not f.exists():
        return None
    try:
        with open(f, "rb") as fh:
            return pickle.load(fh)
    except Exception:  # noqa: BLE001
        return None


def save(problem_name: str, state: dict) -> None:
    """Persist final state for next run's warm start. No-op if disabled
    or if ``LIPA_DEV_WARM_CACHE_READONLY`` is set (used when comparing
    designs against a fixed warm-start snapshot — otherwise each run
    overwrites the cache and successive design tests start from each
    other's degraded final states instead of the same baseline).
    """
    if os.environ.get("LIPA_DEV_WARM_CACHE_READONLY"):
        return
    f = cache_file(problem_name)
    if f is None:
        return
    try:
        with open(f, "wb") as fh:
            pickle.dump(state, fh)
    except Exception:  # noqa: BLE001
        pass
