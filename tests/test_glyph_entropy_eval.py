"""Deterministic checks for the glyph-entropy evaluation formulas."""

import unittest

import numpy as np
import torch

from scripts.evaluate_glyph_entropy import (
    CONDITIONS,
    LABEL,
    NEURAL,
    TARGET_SNRS,
    analyze,
    crossed_source,
    dice,
    gaussian_nearest,
    glyph_features,
    loco,
    pack_masks,
    ridge,
)


class GlyphEntropyMetricsTest(unittest.TestCase):
    def test_mixed_windows_and_exposed_perimeter(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        np.testing.assert_allclose(glyph_features(mask), [1 / 9, 4 / np.pi, 2])
        # The single black corner contributes one mixed window, with exposed outside edges.
        mask[1, 1] = False
        mask[0, 0] = True
        np.testing.assert_allclose(glyph_features(mask), [1 / 9, 4 / np.pi, 0])

    def test_dice_empty_partial_and_crossed(self):
        a = np.array([[True, True], [False, False]])
        b = np.array([[False, True], [True, False]])
        empty = np.zeros_like(a)
        self.assertEqual(float(dice(a, b)), 0.5)
        self.assertEqual(float(dice(empty, empty)), 1.0)
        self.assertEqual(float(dice(a, empty)), 0.0)
        clean = np.stack([a, b])[:, None]
        matched = clean[:, :, None].repeat(3, axis=2)
        identical = matched[[0, 0]]
        np.testing.assert_array_equal(crossed_source(matched, clean), np.full((1, 3), 0.5))
        np.testing.assert_array_equal(crossed_source(identical, clean), np.zeros((1, 3)))
        np.testing.assert_array_equal(crossed_source(matched[::-1], clean), np.full((1, 3), -0.5))

    def test_nearest_uses_alpha_bar_and_first_tie(self):
        noisy = torch.tensor([0.5, 0.0]).reshape(2, 1, 1, 1)
        templates = torch.tensor([-1.0, 1.0]).reshape(1, 2, 1, 1, 1).expand(2, -1, -1, -1, -1)
        self.assertEqual(gaussian_nearest(noisy, templates, torch.tensor(0.25)).tolist(), [1, 0])

    def test_ridge_standardizes_inside_the_fold(self):
        # The held-out value is several training standard deviations from its group.
        # After removing the group intercepts the one standardized varying feature
        # has sum of squares 4 and cross product 4, so its ridge slope is 4/5.
        train_x = np.array([[-1.0, 0.1], [1.0, 0.1], [-1.0, 0.1], [1.0, 0.1]])
        groups = np.array([4, 4, 6, 6])
        y = np.array([9.0, 11.0, 19.0, 21.0])
        prediction, coefficient = ridge(train_x, groups, y, np.array([[3.0, 5.0]]), np.array([4]))
        np.testing.assert_allclose(coefficient, [10.0, 20.0, 0.0, 0.0, 0.8, 0.0], atol=1e-12)
        np.testing.assert_allclose(prediction, [12.4])

    def test_loco_holdout_outcome_cannot_change_its_prediction(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(24, 3))
        groups = np.repeat([4, 6, 8, 10], 6)
        y = rng.uniform(size=24)
        original, _ = loco(x, groups, y)
        y[11] = 100
        updated, _ = loco(x, groups, y)
        np.testing.assert_array_equal(original[11], updated[11])
        self.assertFalse(np.allclose(original[6], updated[6]))

    def test_packed_masks_round_trip_in_row_major_order(self):
        mask = np.zeros((2, 48, 48), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, -1, -1] = True
        packed = pack_masks(mask)
        self.assertEqual(int(packed[0, 0]), 1)
        self.assertEqual(int(packed[1, -1]), 128)
        np.testing.assert_array_equal(
            np.unpackbits(packed, axis=-1, bitorder="little").reshape(mask.shape), mask
        )

    def test_analysis_averages_the_declared_axes(self):
        shape = (6, 2, 24, 16)
        snr, font, char, draw = np.indices(shape)
        baseline = 0.25 + 0.02 * snr + 0.04 * font + 0.002 * char + 0.001 * draw
        width_delta = 0.01 * snr + 0.004 * char
        scores = {
            key: baseline + 0.05 * (key[2] == 202)
            + width_delta * (key[0] == NEURAL and key[1] == 32)
            for key in CONDITIONS
        }
        crossed = {
            key: np.broadcast_to((0.01 * np.arange(6))[:, None, None], (6, 24, 16))
            for key in CONDITIONS
        }
        crossed[LABEL, None, None] = np.zeros((6, 24, 16))
        chars = [{"class_id": i, "glyph": str(i), "stroke_count": 4 + 2 * (i // 6)} for i in range(24)]
        rng = np.random.default_rng(18)
        x = rng.uniform(size=(3, 24, 3))
        steps = [{"target_log2_snr": target} for target in TARGET_SNRS]
        results = analyze(scores, crossed, x, chars, ["a", "b"], ["c", "d", "e"], steps)
        expected = (baseline + 0.025).mean(axis=(0, 3))
        self.assertEqual(len(results["primary"]["conditions"]), 4)
        self.assertEqual(len(results["primary"]["delete_one"]), 24)
        self.assertAlmostEqual(results["primary"]["conditions"][1]["characters"][9]["mean_dice"], expected[1, 9])
        self.assertAlmostEqual(results["paired_width"]["characters"][9]["mean_delta_dice"], 0.025 + 0.004 * 9)
        endpoints = results["crossed_source"]["endpoints"]
        self.assertAlmostEqual(endpoints[0]["contrast_delta_0_minus_minus_5"], 0.05)
        self.assertEqual(next(row for row in endpoints if row["method"] == LABEL)["contrast_delta_0_minus_minus_5"], 0)


if __name__ == "__main__":
    unittest.main()
