"""Full-solve smoke test for the barrel-roll Aliengo task.

Opt-in: set ``RUN_MPC_TESTS=1`` to enable. Skipped by default because the
full LIPA solve takes ~tens of seconds (mostly JIT compile on first run)
and requires the ``mpc-examples`` extra plus fetched MJX assets.

Run with:

    RUN_MPC_TESTS=1 uv run --extra mpc-examples python -m unittest tests.test_mpc_barrel_roll
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from tests.mpc_examples import fetch_assets

_RUN = os.environ.get("RUN_MPC_TESTS")


@unittest.skipUnless(_RUN, "set RUN_MPC_TESTS=1 to run mpc-example smoke tests")
class TestBarrelRoll(unittest.TestCase):
    """Solve the barrel-roll task and check the solver converged sensibly."""

    @classmethod
    def setUpClass(cls) -> None:
        fetch_assets.fetch(["aliengo"])

    def test(self) -> None:
        from tests.mpc_examples.run_offline import solve_task

        result = solve_task("barrel_roll", verbose=True)
        stats = result["stats"]

        self.assertTrue(stats["converged"], f"LIPA reported errors; stats={stats}")
        # Match the LIPA `primal_violation_threshold` configured in the
        # config (1e-3, inf-norm of raw primal residuals). The looser
        # MJX bar reflects LIPA's standard-schedule plateau on these
        # contact-rich short-horizon OCPs (see _mjx_base.py's
        # ``success_tol`` metadata for the per-class rationale).
        # Loosening further would mask convergence regressions.
        self.assertLess(stats["final_dynamics_violation"], 1e-3)
        # Cost is problem-specific (and on this task can rise from the
        # warm start once hard constraints are enforced — phase 1 ignores
        # them). Use a loose upper bound to catch only catastrophic blow-up.
        self.assertLess(stats["final_objective"], 1e7)


if __name__ == "__main__":
    unittest.main()
