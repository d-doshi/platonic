import unittest

import numpy as np

from platonic.data import DataConfig, build_rules, sample_corpus, sample_translation_pairs


class DataTests(unittest.TestCase):
    def test_generation_is_deterministic_and_translation_latents_match(self):
        config = DataConfig()
        rules = build_rules(config, 11)
        first = sample_corpus(12, config, rules, 12)
        second = sample_corpus(12, config, rules, 12)
        np.testing.assert_array_equal(first["tokens"], second["tokens"])

        pairs = sample_translation_pairs(10, config, rules, 13)
        self.assertEqual(pairs["tokens_a"].shape, (10, config.sequence_length))
        self.assertEqual(pairs["tokens_b"].shape, (10, config.sequence_length))
        for depth, shared in enumerate(pairs["shared_levels"]):
            np.testing.assert_array_equal(pairs["levels_a"][depth], shared)
            np.testing.assert_array_equal(pairs["levels_b"][depth], shared)
        self.assertTrue(np.all(pairs["tokens_a"] < config.vocab_size))
        self.assertTrue(np.all(pairs["tokens_b"] >= config.vocab_size))
