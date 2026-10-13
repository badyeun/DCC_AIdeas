"""Check the new objective against its mathematical definition and CE baseline."""
import math
import unittest

import torch
from torch.nn import functional as F

from .neural import classification_loss, training_loss, validate_loss_config


class NauseaLossTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(903)
        self.logits = torch.randn(12, 9, 2, generator=generator)
        self.targets = torch.randint(0, 2, (12, 9), generator=generator)
        self.weights = [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]

    def value_gradient(self, fn, logits=None):
        value = (self.logits if logits is None else logits).clone().requires_grad_()
        loss = fn(value)
        return loss.detach(), torch.autograd.grad(loss, value)[0]

    def ce(self, logits):
        return classification_loss(logits, self.targets, positive_class_weights=self.weights)

    def test_baseline_warmup_and_zero_mix_are_bitwise_original_ce(self):
        expected = self.value_gradient(self.ce)
        for settings in [{}, {"loss_mode": "nausea_ce_gce", "epoch": 1},
                         {"loss_mode": "nausea_ce_gce", "epoch": 3, "gce_mix": 0},
                         {"loss_mode": "nausea_ce_gce", "epoch": 3, "ce_warmup_epochs": 3}]:
            actual = self.value_gradient(lambda x: training_loss(x, self.targets, positive_class_weights=self.weights, **settings))
            for before, after in zip(expected, actual):
                self.assertTrue(torch.equal(before, after))

    def test_matches_explicit_per_item_definition_and_original_denominator(self):
        value = self.logits.clone().requires_grad_()
        loss = training_loss(value, self.targets, positive_class_weights=self.weights,
                             loss_mode="nausea_ce_gce", epoch=2)
        probability = value.float().softmax(-1).gather(2, self.targets.unsqueeze(-1)).squeeze(-1)
        terms = -probability.log()
        terms = torch.cat((terms[:, :6],
                           (0.5 * terms[:, 6] + 0.5 * (1 - probability[:, 6].pow(.7)) / .7)[:, None],
                           terms[:, 7:]), dim=1)
        weights = torch.where(self.targets == 1, torch.tensor(self.weights)[None], 1.0)
        expected = (terms * weights).mean()
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(torch.autograd.grad(loss, value, retain_graph=True)[0],
                                   torch.autograd.grad(expected, value)[0], rtol=2e-6, atol=1e-8)

    def test_only_nausea_gradients_change_by_the_analytic_factor(self):
        _, baseline = self.value_gradient(self.ce)
        _, mixed = self.value_gradient(lambda x: training_loss(x, self.targets, positive_class_weights=self.weights,
                                                               loss_mode="nausea_ce_gce", epoch=2))
        other = [i for i in range(9) if i != 6]
        self.assertTrue(torch.equal(baseline[:, other], mixed[:, other]))
        p = self.logits[:, 6].softmax(-1).gather(1, self.targets[:, 6, None]).squeeze(1)
        scale = .5 + .5 * p.pow(.7)
        torch.testing.assert_close(mixed[:, 6], baseline[:, 6] * scale[:, None], rtol=2e-6, atol=1e-8)
        self.assertTrue(bool(((scale >= .5) & (scale <= 1)).all()))

    def test_extreme_wrong_targets_keep_half_ce_gradient_for_both_classes(self):
        logits = self.logits.clone()
        logits[:, 6] = torch.where(self.targets[:, 6, None] == torch.arange(2)[None], -10000., 10000.)
        base, base_gradient = self.value_gradient(self.ce, logits)
        loss, gradient = self.value_gradient(lambda x: training_loss(x, self.targets, positive_class_weights=self.weights,
                                                                    loss_mode="nausea_ce_gce", epoch=2), logits)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        torch.testing.assert_close(gradient[:, 6], .5 * base_gradient[:, 6], rtol=1e-6, atol=1e-8)
        self.assertGreater(float(loss), 0)
        self.assertLess(float(loss), float(base))

    def test_extreme_correct_targets_and_small_q_are_finite(self):
        logits = self.logits.clone()
        logits[:, 6] = torch.where(self.targets[:, 6, None] == torch.arange(2)[None], 10000., -10000.)
        for q in (1e-6, .7, 1.):
            loss, gradient = self.value_gradient(lambda x: training_loss(x, self.targets, loss_mode="nausea_ce_gce", gce_q=q, epoch=2), logits)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_half_precision_inputs_use_stable_float32_math(self):
        for dtype in (torch.float16, torch.bfloat16):
            value = self.logits.to(dtype).requires_grad_()
            loss = training_loss(value, self.targets, loss_mode="nausea_ce_gce", epoch=2)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(torch.autograd.grad(loss, value)[0]).all()))

    def test_invalid_settings_and_targets_fail_explicitly(self):
        for kwargs in [{"loss_mode": "focal"}, {"loss_mode": []}, {"gce_q": 0}, {"gce_q": 1.1},
                       {"gce_q": True}, {"gce_q": math.nan}, {"gce_mix": -.1}, {"gce_mix": .6},
                       {"gce_mix": math.inf}, {"ce_warmup_epochs": 0}, {"ce_warmup_epochs": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_loss_config(**kwargs)
        for epoch in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                training_loss(self.logits, self.targets, epoch=epoch)
        malformed = self.targets.clone(); malformed[0, 6] = 2
        with self.assertRaises(ValueError):
            training_loss(self.logits, malformed, loss_mode="nausea_ce_gce", epoch=2)
        with self.assertRaises(ValueError):
            training_loss(self.logits, self.targets, positive_class_weights=[1] * 8, loss_mode="nausea_ce_gce", epoch=2)


if __name__ == "__main__":
    unittest.main()
