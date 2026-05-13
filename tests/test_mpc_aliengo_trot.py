"""Full-solve smoke test for the Aliengo trot task.

Opt-in: set ``RUN_MPC_TESTS=1`` to enable. See ``test_mpc_barrel_roll.py``
for the rationale.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

_RUN = os.environ.get("RUN_MPC_TESTS")


def _assets_present() -> bool:
    data_dir = Path(__file__).resolve().parent / "mpc_examples" / "data"
    return (data_dir / "aliengo" / "scene_flat.xml").exists()


@unittest.skipUnless(_RUN, "set RUN_MPC_TESTS=1 to run mpc-example smoke tests")
@unittest.skipUnless(
    _assets_present(),
    "fetch assets first: `python -m tests.mpc_examples.fetch_assets --robots aliengo`",
)
class TestAliengoTrotTwoStep(unittest.TestCase):
    def test(self) -> None:
        from tests.mpc_examples.run_offline import solve_task

        result = solve_task("aliengo_trot", verbose=True)
        stats = result["stats"]

        self.assertTrue(stats["converged"], f"LIPA reported errors; stats={stats}")
        # Match the LIPA `primal_violation_threshold` configured in the
        # config (1e-5, sum-of-squares).
        self.assertLess(stats["final_dynamics_violation"], 1e-5)
        self.assertLess(stats["final_objective"], 1e7)


if __name__ == "__main__":
    unittest.main()
