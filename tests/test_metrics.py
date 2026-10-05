import unittest

import numpy as np

from platonic.metrics import information_imbalance, mutual_knn


class MetricTests(unittest.TestCase):
    def test_identical_tie_free_geometry(self):
        rng = np.random.default_rng(30)
        values = rng.normal(size=(20, 5))
        ii = information_imbalance(values, values)
        knn = mutual_knn(values, values, k=3)
        self.assertTrue(np.isclose(ii["symmetric"], 2.0 / len(values)))
        self.assertTrue(np.isclose(knn["mknn"], 1.0))
