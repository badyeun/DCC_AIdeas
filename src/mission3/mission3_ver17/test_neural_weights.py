"""CPU checks for training-only per-label positive CE costs."""
import math
import unittest

import torch
from torch.nn import functional as F

if __package__:
    from .neural import classification_loss, predictions
else:
    from neural import classification_loss, predictions


class PositiveClassWeightsTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(72)
        self.logits = torch.randn(4, 9, 2, generator=generator)
        # Each label has both positive and negative examples in this batch.
        self.targets = torch.tensor([
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
        ])
        self.weights = [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]

    def test_unit_vector_matches_original_ce_and_gradients_exactly(self):
        expected_logits = self.logits.clone().requires_grad_()
        actual_logits = self.logits.clone().requires_grad_()
        expected = F.cross_entropy(expected_logits.reshape(-1, 2), self.targets.flatten())
        actual = classification_loss(actual_logits, self.targets, positive_class_weights=[1] * 9)
        self.assertTrue(torch.equal(actual, expected))
        expected.backward()
        actual.backward()
        self.assertTrue(torch.equal(actual_logits.grad, expected_logits.grad))

    def test_positive_items_only_and_original_mean_denominator(self):
        actual = classification_loss(self.logits, self.targets, positive_class_weights=self.weights)
        per_item = F.cross_entropy(self.logits.reshape(-1, 2), self.targets.flatten(), reduction="none").reshape(4, 9)
        expected = sum(
            per_item[row, label] * (self.weights[label] if self.targets[row, label] else 1)
            for row in range(4) for label in range(9)
        ) / 36
        torch.testing.assert_close(actual, expected)
        # Weighting is not a normalized weighted average: uniform logits make
        # the denominator difference directly observable.
        uniform = classification_loss(torch.zeros_like(self.logits), self.targets,
                                      positive_class_weights=self.weights)
        multiplier_mean = (sum(self.weights) + 9) / 18
        self.assertAlmostEqual(float(uniform), math.log(2) * multiplier_mean, places=6)

    def test_gradients_match_analytical_positive_costs(self):
        logits = self.logits.clone().requires_grad_()
        loss = classification_loss(logits, self.targets, positive_class_weights=self.weights)
        loss.backward()
        unscaled = logits.detach().softmax(-1) - F.one_hot(self.targets, 2).float()
        costs = torch.tensor([
            [self.weights[label] if target else 1.0 for label, target in enumerate(row)]
            for row in self.targets.tolist()
        ])
        expected = unscaled * costs.unsqueeze(-1) / 36
        torch.testing.assert_close(logits.grad, expected, atol=1e-8, rtol=1e-6)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_nausea_only_vector_matches_legacy_scalar_and_gradients(self):
        for weight in (0.5, 2.0, 3.0):
            with self.subTest(weight=weight):
                scalar_logits = self.logits.clone().requires_grad_()
                vector_logits = self.logits.clone().requires_grad_()
                vector = [1.0] * 9
                vector[6] = weight
                scalar_loss = classification_loss(scalar_logits, self.targets, nausea_positive_weight=weight)
                vector_loss = classification_loss(vector_logits, self.targets, positive_class_weights=vector)
                self.assertTrue(torch.equal(scalar_loss, vector_loss))
                scalar_loss.backward()
                vector_loss.backward()
                self.assertTrue(torch.equal(scalar_logits.grad, vector_logits.grad))

    def test_negative_items_are_not_reweighted(self):
        targets = torch.zeros_like(self.targets)
        original = classification_loss(self.logits, targets)
        actual = classification_loss(self.logits, targets, positive_class_weights=self.weights)
        torch.testing.assert_close(actual, original)

    def test_both_predeclared_candidates_accept_tuple_or_list(self):
        broader = (1.75, 2.25, 2.5, 1.25, 1.5, 1.0, 3.0, 1.75, 1.5)
        for weights in (self.weights, broader):
            loss = classification_loss(self.logits, self.targets, positive_class_weights=weights)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(loss), 0)

    def test_invalid_vectors_are_rejected_even_without_positive_targets(self):
        malformed = [
            [], [1] * 8, [1] * 10, [[1] * 9], [[1]] * 9,
            3.0, "111111111", {index: 1 for index in range(9)},
            torch.ones(9),
        ]
        for invalid in (True, False, 0, -1, float("nan"), float("inf"), -float("inf"),
                        "2", None, 10 ** 1000):
            malformed.append([1] * 8 + [invalid])
        for weights in malformed:
            with self.subTest(weights_type=type(weights).__name__):
                with self.assertRaises(ValueError):
                    classification_loss(self.logits, torch.zeros_like(self.targets),
                                        positive_class_weights=weights)

    def test_conflicting_scalar_and_vector_are_rejected(self):
        for weight in (0.5, 2, 3):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "not both"):
                    classification_loss(self.logits, self.targets, nausea_positive_weight=weight,
                                        positive_class_weights=self.weights)
        for invalid in (True, float("nan"), 0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    classification_loss(self.logits, self.targets, nausea_positive_weight=invalid,
                                        positive_class_weights=self.weights)

    def test_loss_does_not_mutate_logits_targets_or_argmax_predictions(self):
        original_logits, original_targets = self.logits.clone(), self.targets.clone()
        expected = predictions(self.logits).clone()
        weights = list(self.weights)
        classification_loss(self.logits, self.targets, positive_class_weights=weights)
        self.assertTrue(torch.equal(self.logits, original_logits))
        self.assertTrue(torch.equal(self.targets, original_targets))
        self.assertTrue(torch.equal(predictions(self.logits), expected))
        self.assertEqual(weights, self.weights)


if __name__ == "__main__":
    unittest.main()
