"""Behavioral CPU checks for continued encoder training and its safeguards."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from . import adaptation
from .budget import BudgetExceeded
from .neural import DialogueClassifier, training_loss


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.dropout = nn.Dropout(.1)

    def forward(self, inputs):
        return inputs + self.dropout(self.linear(inputs).tanh()) * .1


class TinyEncoder(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="roberta", num_hidden_layers=12,
                                      hidden_size=4, use_cache=True)
        self.embeddings = nn.Sequential(nn.Embedding(120, 4), nn.Dropout(.2))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([TinyLayer() for _ in range(12)])
        self.pooler = nn.Linear(4, 4)
        self.checkpoint_options = None

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.checkpoint_options = gradient_checkpointing_kwargs

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class GenerousBudget:
    def __init__(self, admit=True):
        self.admit = admit
        self.checks = 0
        self.estimates = []

    def check(self, stage, training=False):
        self.checks += 1
        return 100000.

    def remaining(self, training=False):
        return 100000.

    def can_start(self, seconds, training=True):
        self.estimates.append(seconds)
        return self.admit


class OverflowScaler:
    """Model the documented enabled GradScaler skip/backoff contract on CPU."""
    def __init__(self, overflows):
        self.overflows = set(overflows)
        self.attempt = 0
        self.current_scale = 64.

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.current_scale

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        if self.attempt in self.overflows:
            next(parameter for group in optimizer.param_groups for parameter in group["params"]
                 if parameter.grad is not None).grad.fill_(float("inf"))

    def step(self, optimizer):
        if self.attempt not in self.overflows:
            optimizer.step()

    def update(self):
        if self.attempt in self.overflows:
            self.current_scale *= .5
        self.attempt += 1


def new_model():
    torch.manual_seed(123)
    model = DialogueClassifier(TinyEncoder(), feature_dim=8, utterance_evidence=True,
                               evidence_hidden=5, evidence_dropout=.1)
    adaptation.configure_adaptation(model)
    return model


def batch(rows=4, offset=0):
    ids = (torch.arange(rows * 6).reshape(rows, 1, 6) + offset) % 100
    turns = torch.tensor([[[-1, 0, 0, 1, 1, -1]]]).expand(rows, -1, -1).clone()
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "window_mask": torch.ones(rows, 1, dtype=torch.bool), "turn_ids": turns,
            "token_weights": (turns >= 0).float(),
            "targets": ((torch.arange(rows).unsqueeze(1) + torch.arange(9) + offset) % 3 == 0).long()}


def settings(**kwargs):
    return {"learning_rate": .005, "head_learning_rate": .01, "weight_decay": .01,
            "gradient_accumulation": 4, "scheduler_horizon_epochs": 2,
            "log_every": 1000, "pilot_optimizer_steps": 10, **kwargs}


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_only_upper_four_and_classification_paths_train(self):
        model = new_model()
        for index, layer in enumerate(model.encoder.encoder.layer):
            self.assertTrue(all(p.requires_grad == (index >= 8) for p in layer.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.encoder.embeddings.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.encoder.pooler.parameters()))
        self.assertFalse(model.feature_weights.requires_grad)
        self.assertFalse(model.feature_bias.requires_grad)
        self.assertTrue(model.class_weights.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.projection.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.evidence.parameters()))
        self.assertEqual(model.encoder.checkpoint_options, {"use_reentrant": False})

    def test_frozen_prefix_eval_after_model_train(self):
        model = new_model()
        model.train()
        adaptation.enforce_frozen_eval(model)
        self.assertFalse(model.encoder.embeddings.training)
        self.assertFalse(model.encoder.pooler.training)
        self.assertTrue(all(not layer.training for layer in model.encoder.encoder.layer[:8]))
        self.assertTrue(all(layer.training for layer in model.encoder.encoder.layer[8:]))
        self.assertTrue(model.evidence.training)

    def test_configuration_rejects_invalid_architecture_before_changing_flags(self):
        model = new_model()
        before = [p.requires_grad for p in model.parameters()]
        model.encoder.config.num_hidden_layers = 24
        with self.assertRaises(ValueError):
            adaptation.configure_adaptation(model)
        self.assertEqual(before, [p.requires_grad for p in model.parameters()])

    def test_gce_active_immediately_and_matches_original_objective(self):
        torch.manual_seed(4)
        logits, targets = torch.randn(4, 9, 2), batch()["targets"]
        actual, details = adaptation.adaptation_loss(logits, targets)
        expected = training_loss(logits, targets, positive_class_weights=adaptation.POSITIVE_WEIGHTS,
                                 loss_mode="nausea_ce_gce", gce_q=.7, gce_mix=.5,
                                 ce_warmup_epochs=1, epoch=2)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(details["consistency_loss"], 0.)

    def test_rdrop_symmetric_and_both_distributions_receive_full_gradients(self):
        torch.manual_seed(4)
        first = torch.randn(4, 9, 2, requires_grad=True)
        second = torch.randn(4, 9, 2, requires_grad=True)
        targets = batch()["targets"]
        total, details = adaptation.adaptation_loss(first, targets, second, recipe="continue_rdrop", alpha=.7)
        swapped, _ = adaptation.adaptation_loss(second, targets, first, recipe="continue_rdrop", alpha=.7)
        self.assertTrue(torch.allclose(total, swapped))
        self.assertGreater(details["consistency_loss"], 0.)
        gradients = torch.autograd.grad(total, (first, second), retain_graph=True)
        first_loss = adaptation.adaptation_loss(first, targets)[0]
        second_loss = adaptation.adaptation_loss(second, targets)[0]
        log_first, log_second = F.log_softmax(first, -1), F.log_softmax(second, -1)
        kl = .5 * (F.kl_div(log_second, log_first.exp(), reduction="none").sum(-1).mean()
                   + F.kl_div(log_first, log_second.exp(), reduction="none").sum(-1).mean())
        expected = (first_loss + second_loss) / 2 + .7 * kl
        expected_gradients = torch.autograd.grad(expected, (first, second))
        for actual, reference in zip(gradients, expected_gradients):
            self.assertTrue(torch.allclose(actual, reference, atol=1e-7))
            self.assertGreater(float(actual.abs().sum()), 0.)

    def test_rdrop_identical_views_add_zero(self):
        logits, targets = torch.randn(4, 9, 2), batch()["targets"]
        actual, details = adaptation.adaptation_loss(logits, targets, logits, recipe="continue_rdrop")
        expected, _ = adaptation.adaptation_loss(logits, targets)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(details["consistency_loss"], 0.)

    def test_invalid_and_nonfinite_objectives_fail_explicitly(self):
        logits, targets = torch.randn(4, 9, 2), batch()["targets"]
        for options in ({"recipe": "unknown"}, {"recipe": "continue_rdrop"}, {"alpha": float("nan")}):
            with self.assertRaises(ValueError):
                adaptation.adaptation_loss(logits, targets, **options)
        bad = logits.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises((ValueError, FloatingPointError)):
            adaptation.adaptation_loss(bad, targets)

    def test_training_changes_upper_encoder_and_all_nine_heads_only(self):
        for recipe in adaptation.RECIPES:
            model = new_model()
            before = adaptation.trainable_state(model)
            frozen_before = adaptation.frozen_digest(model)
            batches = [batch(4, offset) for offset in (0, 7, 13, 21, 2)]
            state = adaptation.make_optimizer(model, settings(), len(batches), "cpu")
            row = adaptation.train_epoch(model, batches, settings(), "cpu", state, GenerousBudget(), 1, recipe)
            self.assertEqual(row["optimizer_steps"], 2)
            self.assertEqual(row["samples"], 20)
            self.assertEqual(row["positive_target_counts"], torch.cat([b["targets"] for b in batches]).sum(0).tolist())
            self.assertEqual(adaptation.frozen_digest(model), frozen_before)
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                                if name.startswith("encoder.encoder.layer.8")))
            for label in range(9):
                self.assertFalse(torch.equal(model.class_weights[label], before["class_weights"][label]))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_accumulation_matches_sample_weighted_groups_including_short_tail(self):
        model = new_model()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.
        expected = deepcopy(model)
        batches = [batch(4, 0), batch(4, 2), batch(1, 6), batch(2, 9), batch(1, 3)]
        options = settings(gradient_accumulation=3, weight_decay=0.)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        state = optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), torch.float16
        adaptation.train_epoch(model, batches, options, "cpu", state, GenerousBudget(), 1, "continue_gce")
        reference_optimizer = torch.optim.SGD([p for p in expected.parameters() if p.requires_grad], lr=.01)
        expected.train()
        adaptation.enforce_frozen_eval(expected)
        for start in (0, 3):
            group = batches[start:start + 3]
            joined = {key: torch.cat([row[key] for row in group]) for key in group[0]}
            loss, _ = adaptation.adaptation_loss(expected(joined)[0], joined["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in expected.parameters() if p.requires_grad], 1.)
            reference_optimizer.step()
            reference_optimizer.zero_grad(set_to_none=True)
        for actual, reference in zip(model.parameters(), expected.parameters()):
            self.assertTrue(torch.allclose(actual, reference, atol=1e-7, rtol=1e-6))

    def test_restore_validates_before_write_and_preserves_frozen_state(self):
        model = new_model()
        state = adaptation.trainable_state(model)
        original = state["class_bias"].clone()
        state["class_bias"] += .1
        self.assertTrue(torch.equal(model.class_bias, original))
        audit = adaptation.restore_trainable_state(model, state)
        self.assertEqual(audit["changed_elements"], 18)
        self.assertTrue(audit["frozen_state_unchanged"])
        valid = adaptation.trainable_state(model)
        invalid = dict(valid, unrelated=torch.tensor(1.))
        with self.assertRaises(ValueError):
            adaptation.restore_trainable_state(model, invalid)
        invalid = deepcopy(valid)
        invalid["class_bias"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            adaptation.restore_trainable_state(model, invalid)
        for name, value in adaptation.trainable_state(model).items():
            self.assertTrue(torch.equal(value, valid[name]))

    def test_pilot_updates_are_real_and_not_repeated(self):
        model = new_model()
        batches = [batch(1, index) for index in range(45)]
        options = settings()
        state = adaptation.make_optimizer(model, options, len(batches), "cpu")
        row = adaptation.train_epoch(model, batches, options, "cpu", state, GenerousBudget(), 1,
                                     "continue_gce", {"holdout_batches": 10, "full_ratio": 1.25})
        self.assertEqual(row["optimizer_steps"], 12)
        self.assertEqual(row["samples"], 45)
        self.assertEqual(row["pilot_report"]["observed_optimizer_steps"], 11)
        self.assertEqual(row["pilot_report"]["measured_microbatches"], 40)
        self.assertTrue(row["pilot_report"]["includes_training_updates"])

    def test_budget_rejection_raises_instead_of_returning_partial_epoch(self):
        model = new_model()
        options = settings(pilot_optimizer_steps=1)
        batches = [batch(1, index) for index in range(10)]
        state = adaptation.make_optimizer(model, options, len(batches), "cpu")
        budget = GenerousBudget(admit=False)
        with self.assertRaises(BudgetExceeded):
            adaptation.train_epoch(model, batches, options, "cpu", state, budget, 1, "continue_gce",
                                   {"holdout_batches": 2, "full_ratio": 1.25})
        self.assertTrue(budget.estimates)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_full_refit_pilot_reserves_all_planned_epochs(self):
        budget = GenerousBudget()
        report = adaptation._pilot_report(settings(), microbatches=44, measured_batches=40,
            measured_seconds=40., total_batches=100, budget=budget, device="cpu",
            context={"refit": True, "refit_epochs": 2, "remaining_epochs": 2, "holdout_batches": 0})
        self.assertEqual(report["estimated_refit_seconds"], 0.)
        self.assertEqual(report["estimated_required_seconds"], 56. + 100. + 120.)

    def test_candidate_pilot_keeps_the_already_selected_refit_reserve(self):
        report = adaptation._pilot_report(settings(), microbatches=44, measured_batches=40,
            measured_seconds=40., total_batches=100, budget=GenerousBudget(), device="cpu",
            context={"full_ratio": 1.25, "refit_epochs": 1, "minimum_refit_seconds": 1200.})
        self.assertEqual(report["estimated_refit_seconds"], 1200.)
        self.assertEqual(report["estimated_required_seconds"], 56. + 1200. + 120.)

    def test_fp16_overflow_backs_off_then_trains_without_advancing_skipped_schedule(self):
        model = new_model()
        batches = [batch(2, index) for index in range(3)]
        options = settings(gradient_accumulation=1, pilot_optimizer_steps=1)
        optimizer, scheduler, _, dtype = adaptation.make_optimizer(model, options, len(batches), "cpu")
        scaler = OverflowScaler({0})
        before = model.class_weights.detach().clone()
        row = adaptation.train_epoch(model, batches, options, "cpu", (optimizer, scheduler, scaler, dtype),
            GenerousBudget(), 1, "continue_gce", {"holdout_batches": 1})
        self.assertEqual(row["optimizer_attempts"], 3)
        self.assertEqual(row["optimizer_steps"], 2)
        self.assertEqual(row["skipped_optimizer_steps"], 1)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual(scaler.get_scale(), 32.)
        self.assertFalse(torch.equal(before, model.class_weights))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        self.assertEqual(row["pilot_report"]["observed_optimizer_steps"], 2)
        self.assertEqual(row["pilot_report"]["observed_optimizer_attempts"], 3)
        self.assertEqual(row["pilot_report"]["measured_microbatches"], 1)

    def test_pilot_does_not_reset_warmup_when_an_intermediate_update_overflows(self):
        model = new_model()
        batches = [batch(2, index) for index in range(4)]
        options = settings(gradient_accumulation=1, pilot_optimizer_steps=2)
        optimizer, scheduler, _, dtype = adaptation.make_optimizer(model, options, len(batches), "cpu")
        row = adaptation.train_epoch(model, batches, options, "cpu",
            (optimizer, scheduler, OverflowScaler({1}), dtype), GenerousBudget(), 1, "continue_gce",
            {"holdout_batches": 1})
        self.assertEqual(row["optimizer_steps"], 3)
        self.assertEqual(row["pilot_report"]["measured_microbatches"], 3)
        self.assertTrue(row["pilot_report"]["warmup_update_excluded"])

    def test_all_overflow_epoch_fails_without_publishing_success(self):
        model = new_model()
        batches = [batch(2, index) for index in range(2)]
        options = settings(gradient_accumulation=1)
        optimizer, scheduler, _, dtype = adaptation.make_optimizer(model, options, len(batches), "cpu")
        before = adaptation.trainable_state(model)
        with self.assertRaisesRegex(FloatingPointError, "No successful optimizer update"):
            adaptation.train_epoch(model, batches, options, "cpu",
                (optimizer, scheduler, OverflowScaler({0, 1}), dtype), GenerousBudget(), 1, "continue_gce")
        self.assertEqual(scheduler.last_epoch, 0)
        for name, value in adaptation.trainable_state(model).items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_disabled_scaler_rejects_nonfinite_gradients(self):
        model = new_model()
        optimizer, scheduler, scaler, _ = adaptation.make_optimizer(model, settings(), 1, "cpu")
        model.class_weights.grad = torch.full_like(model.class_weights, float("inf"))
        with self.assertRaises(RuntimeError):
            adaptation._optimizer_step(model, optimizer, scheduler, scaler)
        self.assertEqual(scheduler.last_epoch, 0)


if __name__ == "__main__":
    unittest.main()
