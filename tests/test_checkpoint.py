import tempfile
import unittest
from pathlib import Path

import torch

from platonic.data import DataConfig, build_rules
from platonic.model import CausalTransformer, TransformerConfig
from platonic.training import load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        data = DataConfig()
        rules = build_rules(data, 41)
        model = CausalTransformer(TransformerConfig(
            data.surface_vocab_size, data.sequence_length,
            width=16, heads=2, layers=1,
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, data, rules, {"rules": 41}, 2, {"epochs": 2})
            loaded, loaded_data, loaded_rules, payload = load_checkpoint(path)
        self.assertEqual(loaded_data, data)
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(len(loaded_rules.shared), data.shared_depth)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
