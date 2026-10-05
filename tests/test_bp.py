import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from platonic.bp.experiments import run_ii, run_mknn


class BPTests(unittest.TestCase):
    def test_settled_smoke_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ii_path = run_ii(output, shared_depth=2, seed=42, n=32)
            mknn_path = run_mknn(output, shared_depth=2, seed=42, n=32, k=3)
            with ii_path.open() as handle:
                ii_rows = list(csv.DictReader(handle))
            with mknn_path.open() as handle:
                mknn_rows = list(csv.DictReader(handle))
        self.assertEqual(len(ii_rows), 132)
        self.assertEqual([row["stage_name"] for row in mknn_rows], [
            "u0", "u1", "u1+u2", "u1+u2+u3", "u1+u2+u3+d3",
            "u1+u2+d2", "u1+d1", "d0",
        ])
        expected = [
            0.07291666666666666, 0.07291666666666666, 0.22916666666666666,
            0.4270833333333333, 0.5104166666666666, 0.38541666666666663,
            0.07291666666666666, 0.16666666666666666,
        ]
        np.testing.assert_allclose(
            [float(row["mutual_knn"]) for row in mknn_rows], expected, atol=1e-14
        )
