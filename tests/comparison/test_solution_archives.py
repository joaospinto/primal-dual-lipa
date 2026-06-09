from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.comparison.problem_spec import SolverResult
from tests.comparison.render_solution_frames import _frame_indices, _tile_frames
from tests.comparison.report import write_solution_archives


class SolutionArchiveTest(unittest.TestCase):
    def test_write_solution_archives_skips_missing_iterates(self):
        result = SolverResult(
            solver_name="lipa",
            problem_name="toy problem",
            iterations=7,
            solve_time_ms=12.5,
            final_cost=3.25,
            eq_violation_inf=1e-8,
            ineq_violation_inf=2e-8,
            success=True,
            X=np.arange(6, dtype=float).reshape(3, 2),
            U=np.arange(4, dtype=float).reshape(2, 2),
            Theta=np.array([0.5]),
            notes="ok",
        )
        missing = SolverResult(
            solver_name="ipopt",
            problem_name="toy problem",
            iterations=0,
            solve_time_ms=0.0,
            final_cost=np.nan,
            eq_violation_inf=np.nan,
            ineq_violation_inf=np.nan,
            success=False,
            notes="unavailable",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_solution_archives([result, missing], out_dir)
            paths = sorted(out_dir.glob("*.npz"))

            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "toy_problem__lipa.npz")
            with np.load(paths[0], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["X"], result.X)
                np.testing.assert_array_equal(archive["U"], result.U)
                np.testing.assert_array_equal(archive["Theta"], result.Theta)
                self.assertEqual(str(archive["solver_name"]), "lipa")
                self.assertEqual(str(archive["problem_name"]), "toy problem")
                self.assertEqual(int(archive["iterations"]), 7)
                self.assertTrue(bool(archive["success"]))


class FrameStripHelpersTest(unittest.TestCase):
    def test_frame_indices_are_uniform_and_include_endpoints(self):
        np.testing.assert_array_equal(_frame_indices(11, 6), [0, 2, 4, 6, 8, 10])

    def test_tile_frames_inserts_white_gaps(self):
        a = np.zeros((2, 3, 3), dtype=np.uint8)
        b = np.full((2, 3, 3), 10, dtype=np.uint8)

        tiled = _tile_frames([a, b], gap_px=2)

        self.assertEqual(tiled.shape, (2, 8, 3))
        np.testing.assert_array_equal(tiled[:, :3], a)
        self.assertTrue(np.all(tiled[:, 3:5] == 255))
        np.testing.assert_array_equal(tiled[:, 5:], b)


if __name__ == "__main__":
    unittest.main()
