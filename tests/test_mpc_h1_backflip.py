"""Full-solve smoke test for the H1 humanoid backflip task.

Opt-in: set ``RUN_MPC_TESTS=1`` to enable. See ``test_mpc_barrel_roll.py``
for the rationale.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from tests.mpc_examples import fetch_assets

_RUN = os.environ.get("RUN_MPC_TESTS")


@unittest.skipUnless(_RUN, "set RUN_MPC_TESTS=1 to run mpc-example smoke tests")
class TestH1Backflip(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fetch_assets.fetch(["unitree_h1"])

    def test(self) -> None:
        from tests.mpc_examples.run_offline import solve_task

        result = solve_task("h1_backflip", verbose=True)
        stats = result["stats"]

        self.assertTrue(stats["converged"], f"LIPA reported errors; stats={stats}")
        # Match the LIPA `primal_violation_threshold` configured in the
        # config (1e-3, inf-norm of raw primal residuals — MJX-class
        # tolerance, see _mjx_base.py).
        self.assertLess(stats["final_dynamics_violation"], 1e-3)
        self.assertLess(stats["final_objective"], 1e7)


if __name__ == "__main__":
    unittest.main()
