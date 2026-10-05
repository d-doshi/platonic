import json
import tempfile
import unittest
from pathlib import Path

from platonic.experiments import load_config, run_one


class SmokeTests(unittest.TestCase):
    def test_end_to_end_without_bp(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "smoke.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = run_one(config, output, include_bp=False)
            self.assertTrue(checkpoint.exists())
            self.assertTrue((output / "transformer" / "metrics.csv").exists())
            manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["derived_seeds"]["random_seed"], 42)
