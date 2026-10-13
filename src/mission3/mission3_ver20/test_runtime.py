"""Actual CPU training checks and synthetic protocol checks; no real F1 claims."""
from __future__ import annotations

from collections import Counter
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parent))
    from mission3_ver20 import data, neural, reference, runtime
else:
    from . import data, neural, reference, runtime
from mission3_ver14.test_neural import FakeTokenizer
from mission3_ver20.test_partial_freeze import ToyRoberta


class PartialEncoder(ToyRoberta):
    """Production-shaped 24-layer fixture; freezing remains real in tests."""

    def __init__(self):
        super().__init__(layers=24)
        self.config.save_pretrained = lambda directory: Path(directory, "config.json").write_text(
            json.dumps({"model_type": "roberta", "hidden_size": 8, "num_hidden_layers": 24}),
            encoding="utf-8")

    @property
    def embedding(self):
        return self.embeddings[0]


def settings(root, **overrides):
    return {
        "version": "ver20", "backend": "neural", "model_id": "klue/roberta-large",
        "revision": "28d911204e9022eda172571ca8cc61eaffd942f7",
        "expected_encoder_layers": 24, "trainable_encoder_layers": 8,
        "scheduler_horizon_epochs": 3, "pilot_optimizer_steps": 10,
        "input_zip": "", "train_dir": str(root / "train"), "validation_dir": str(root / "validation"),
        "output_dir": str(root / "results"), "work_dir": str(root / "work"), "resume_run": "",
        "run_validation": True, "seed": 21, "holdout_size": .2, "batch_size": 2,
        "gradient_accumulation": 2, "max_epochs": 3, "patience": 2, "max_length": 8,
        "overlap": 2, "encoder_window_batch": 2, "learning_rate": .01, "head_learning_rate": .02,
        "refit_full_training": True, "allow_cpu": True, "log_every": 100,
        "candidate_id": "fixture", "positive_class_weights": list(runtime.CANDIDATES[0][1]),
        "time_budget_minutes": 175., "finalization_reserve_seconds": 1200., "budget_safety_factor": 1.25,
        **overrides,
    }


def examples():
    texts = [f"token_{index + 1} token_{index + 15}" for index in range(6)]
    targets = np.array([[(index + label) % 2 for label in range(9)] for index in range(6)], dtype=np.int8)
    return texts, targets


class ControlledBudget:
    def __init__(self, *, stop_microbatch=None, admit=True):
        self.stop_microbatch = stop_microbatch
        self.admit = admit
        self.microbatches = 0

    def check(self, stage, training=False):
        if stage == "train microbatch":
            self.microbatches += 1
            if self.stop_microbatch is not None and self.microbatches >= self.stop_microbatch:
                raise runtime.BudgetExceeded(stage)
        return 10000.

    def can_start(self, estimate, training=True):
        return self.admit

    def remaining(self, training=False):
        return 10000.


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        neural.seed_everything(21)
        self.temporary = tempfile.TemporaryDirectory(prefix=".runtime_test_", dir=HERE)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def model_loader(self, config=None):
        config = config or settings(self.root)
        texts, targets = examples()
        encoder = PartialEncoder()
        neural.configure_partial_finetuning(encoder)
        model = neural.DialogueClassifier(encoder, label_attention=False, contrastive=False, feature_dim=16)
        model.encoder_window_batch = config["encoder_window_batch"]
        loader = runtime._loader(texts, targets, FakeTokenizer(), config, False)
        return model, loader

    def test_fixed_large_settings_and_training_plan_are_validated(self):
        config = settings(self.root)
        checked = runtime.validate_settings(config)
        self.assertEqual(checked["ce_views"], 1)
        self.assertEqual(len(runtime.CANDIDATES), 1)
        self.assertEqual(runtime.CANDIDATES[0][0], "large_partial_ce")
        for key, value in (("model_id", "klue/roberta-base"), ("revision", "main"),
                           ("expected_encoder_layers", 12), ("trainable_encoder_layers", 24),
                           ("trainable_encoder_layers", True), ("max_epochs", 4),
                           ("scheduler_horizon_epochs", 4), ("pilot_optimizer_steps", 0),
                           ("loss_mode", "nausea_ce_gce"), ("utterance_evidence", True),
                           ("label_attention", True), ("time_budget_minutes", 181)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runtime.validate_settings({**config, key: value})

    def test_optimizer_excludes_all_frozen_parameters(self):
        config = settings(self.root)
        model, loader = self.model_loader(config)
        optimizer, _, _, _ = runtime._optimizer(model, config, len(loader), torch.device("cpu"))
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimized, {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        self.assertFalse(optimized & {id(parameter) for parameter in model.parameters() if not parameter.requires_grad})
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_real_epoch_uses_one_forward_and_vector_loss_per_microbatch(self):
        config = settings(self.root)
        model, loader = self.model_loader(config)
        original = neural.classification_loss
        calls, forwards, gradients = [], [], []
        before = model.encoder.embedding.weight.detach().clone()
        top_before = model.encoder.encoder.layer[-1][0].weight.detach().clone()

        def observe(logits, targets, **kwargs):
            calls.append(kwargs["positive_class_weights"])
            logits.register_hook(lambda gradient: gradients.append(gradient.detach().clone()))
            return original(logits, targets, **kwargs)

        handle = model.register_forward_hook(lambda module, args, output: forwards.append(output[0].detach()))
        try:
            with patch.object(neural, "classification_loss", side_effect=observe), \
                    patch.object(neural, "binary_supervised_contrastive_loss", side_effect=AssertionError("Unexpected second objective")):
                row = runtime._train_epoch(model, loader, config, torch.device("cpu"),
                                           runtime._optimizer(model, config, len(loader), torch.device("cpu")),
                                           ControlledBudget(), 1)
        finally:
            handle.remove()
        self.assertEqual(len(forwards), len(loader))
        self.assertEqual(calls, [config["positive_class_weights"]] * len(loader))
        self.assertEqual(len(gradients), len(loader))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))
        self.assertTrue(torch.equal(before, model.encoder.embedding.weight))
        self.assertFalse(torch.equal(top_before, model.encoder.encoder.layer[-1][0].weight))
        self.assertEqual(row["ce_views"], 1)
        self.assertEqual(row["scheduler_horizon_epochs"], 3)
        self.assertEqual(row["positive_target_counts"], examples()[1].sum(0).tolist())

    def prepare(self, model_id, revision, assets):
        encoder, tokenizer = PartialEncoder(), FakeTokenizer()
        for name, value in (("encoder", encoder.config), ("tokenizer", tokenizer)):
            directory = assets / name
            directory.mkdir(parents=True)
            value.save_pretrained(directory)
        return encoder, tokenizer, revision

    def test_fresh_three_epoch_refit_trains_all_targets_and_only_top_layers(self):
        texts, targets = examples()
        config = settings(self.root)
        initial_encoder = PartialEncoder()
        with torch.no_grad():
            initial_encoder.embedding.weight.fill_(99)
        torch.save(initial_encoder.state_dict(), self.root / "selection_weights.pt")
        prepared, observed_targets, learning_rates, tokens = [], [], [], []
        original_train, optimizer_type = runtime._train_epoch, torch.optim.AdamW

        def prepare(*args):
            encoder, tokenizer, revision = self.prepare(*args)
            prepared.append((encoder, encoder.embedding.weight.detach().clone(),
                             encoder.encoder.layer[-1][0].weight.detach().clone()))
            return encoder, tokenizer, revision

        def observe_epoch(model, loader, *args, **kwargs):
            observed_targets.append(loader.dataset.targets.numpy().copy())
            hook = model.register_forward_pre_hook(lambda module, arguments:
                                                  tokens.extend(arguments[0]["input_ids"][:, 0, 1].tolist()))
            try:
                return original_train(model, loader, *args, **kwargs)
            finally:
                hook.remove()

        class ObservedAdamW(optimizer_type):
            def step(self, *args, **kwargs):
                learning_rates.append([group["lr"] for group in self.param_groups])
                return super().step(*args, **kwargs)

        with patch.object(neural, "prepare_encoder", side_effect=prepare), \
                patch.object(runtime, "_train_epoch", side_effect=observe_epoch), \
                patch.object(torch.optim, "AdamW", ObservedAdamW):
            manifest, expected = runtime.refit_selected(texts, targets, config, 3, self.root, ControlledBudget())
        self.assertEqual(len(prepared), 1)
        self.assertIsNot(prepared[0][0], initial_encoder)
        self.assertFalse(torch.equal(prepared[0][1], initial_encoder.embedding.weight))
        self.assertTrue(torch.equal(prepared[0][1], prepared[0][0].embedding.weight))
        self.assertFalse(torch.equal(prepared[0][2], prepared[0][0].encoder.layer[-1][0].weight))
        for observed in observed_targets:
            np.testing.assert_array_equal(observed, targets)
        self.assertEqual(len(observed_targets), 3)
        self.assertEqual(Counter(tokens), Counter({11 + i: 3 for i in range(6)}))
        self.assertEqual(manifest["fitted_samples"], 6)
        self.assertEqual(manifest["training_scope"], "all_training")
        self.assertEqual(manifest["selected_epochs"], 3)
        # Three batches / accumulation two => two updates per epoch, horizon six.
        expected_rates = np.array([[.01, .02]]) * np.array([1., 1., 4/5, 3/5, 2/5, 1/5])[:, None]
        np.testing.assert_allclose(learning_rates, expected_rates, atol=1e-12, rtol=1e-12)
        history = runtime._read(self.root / "final_history.json")
        self.assertEqual([r["scheduler_horizon_epochs"] for r in history], [3, 3, 3])
        restored = neural.DialogueClassifier(PartialEncoder(), label_attention=False, contrastive=False)
        restored.encoder_window_batch = 2
        runtime._load_weights(restored, self.root / manifest["weights"])
        actual = runtime._predict(restored, runtime._loader(texts, None, FakeTokenizer(), config, False), torch.device("cpu"))
        np.testing.assert_array_equal(actual, expected)

    def test_portable_inference_rechecks_partial_architecture_audit(self):
        config = settings(self.root)
        with patch.object(neural, "prepare_encoder", side_effect=self.prepare):
            manifest, expected = runtime.refit_selected(*examples(), config, 1, self.root, ControlledBudget())
        adapters = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=PartialEncoder().config)),
            AutoModel=SimpleNamespace(from_config=Mock(side_effect=lambda *args, **kwargs: PartialEncoder())),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=FakeTokenizer())))
        checkpoint = self.root / "final_model.ckpt"
        with patch.dict(sys.modules, {"transformers": adapters}):
            actual = runtime.predict_checkpoint(checkpoint, examples()[0])
            np.testing.assert_array_equal(actual, expected)
            manifest["partial_finetuning"]["trainable_parameters"] += 1
            data.write_json(checkpoint, manifest)
            with self.assertRaisesRegex(ValueError, "audit disagrees"):
                runtime.predict_checkpoint(checkpoint, examples()[0])

    def test_candidate_records_actual_freeze_and_fit_only_scope(self):
        config = settings(self.root, max_epochs=1)
        texts, targets = examples()
        fit, holdout = np.arange(4), np.arange(4, 6)
        report_path = self.root / "shared_pilot.json"
        with patch.object(neural, "prepare_encoder", side_effect=self.prepare):
            manifest, result = runtime.train_candidate(
                texts, targets, fit, holdout, config, self.root, ControlledBudget(),
                pilot_report_path=report_path)
        self.assertEqual(manifest["training_scope"], "fit_only")
        self.assertEqual(manifest["fitted_samples"], len(fit))
        self.assertEqual(manifest["partial_finetuning"]["trainable_layer_indices"], list(range(16, 24)))
        self.assertEqual(result["history"][0]["positive_target_counts"], targets[fit].sum(0).tolist())
        self.assertEqual(runtime._read(report_path), runtime._read(self.root / "pilot_report.json"))
        with np.load(self.root / "holdout_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["indices"], holdout)
            np.testing.assert_array_equal(saved["targets"], targets[holdout])

    def test_joint_improvement_selects_lower_macro_and_preserves_first_tie(self):
        config = settings(self.root, patience=3)
        model, loader = self.model_loader(config)
        truth = examples()[1]
        reference_predictions = truth.copy()
        qualified = truth.copy()
        unqualified_higher_macro = truth.copy()
        for label in range(9):
            positive_rows = np.flatnonzero(truth[:, label])
            reference_predictions[positive_rows[:2], label] = 0
            if label != 6:
                qualified[positive_rows[0], label] = 0
        unqualified_higher_macro[np.flatnonzero(truth[:, 6])[:2], 6] = 0
        reference_metrics = data.score_predictions(truth, reference_predictions)
        qualified_metrics = data.score_predictions(truth, qualified)
        other_metrics = data.score_predictions(truth, unqualified_higher_macro)
        self.assertLess(qualified_metrics["macro_f1"], other_metrics["macro_f1"])
        self.assertEqual(runtime._selection_key(qualified_metrics, reference_metrics)[0], True)
        self.assertEqual(runtime._selection_key(other_metrics, reference_metrics)[0], False)
        with patch.object(runtime, "_predict", side_effect=[qualified, unqualified_higher_macro, qualified]), \
                patch.object(runtime, "_save_weights", wraps=runtime._save_weights) as save:
            result = runtime._fit_selection(
                model, loader, loader, truth, np.arange(6), config, torch.device("cpu"), self.root,
                ControlledBudget(), 1.25, reference_metrics=reference_metrics)
        self.assertEqual(result["selected_epochs"], 1)
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(save.call_count, 1)
        with np.load(self.root / "holdout_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["predictions"], qualified)

    def test_pilot_observes_existing_updates_without_changing_model(self):
        config = settings(self.root, pilot_optimizer_steps=1)
        first, loader = self.model_loader(config)
        second, _ = self.model_loader(config)
        second.load_state_dict(first.state_dict(), strict=True)
        report_path = self.root / "pilot.json"
        for model, context in ((first, None), (second, {"report_paths": [report_path],
                                 "holdout_batches": len(loader), "full_ratio": 1.25})):
            state = runtime._optimizer(model, config, len(loader), torch.device("cpu"))
            torch.manual_seed(901)
            runtime._train_epoch(model, loader, config, torch.device("cpu"), state,
                                 ControlledBudget(), 1, pilot_context=context)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        report = runtime._read(report_path)
        self.assertEqual(report["observed_microbatches"], len(loader))
        self.assertEqual(report["observed_optimizer_steps"], 2)
        self.assertEqual(report["measured_microbatches"], 1)
        self.assertTrue(report["includes_training_updates"])
        self.assertFalse(report["resets_or_repeats_updates"])

    def test_pilot_abort_leaves_no_completed_checkpoint(self):
        config = settings(self.root, pilot_optimizer_steps=1)
        model, loader = self.model_loader(config)
        with self.assertRaises(runtime.BudgetExceeded):
            runtime._fit_selection(model, loader, loader, examples()[1], np.arange(6), config,
                                   torch.device("cpu"), self.root, ControlledBudget(admit=False), 1.25)
        self.assertEqual(runtime._read(self.root / "pilot_report.json")["decision"], "stop_no_complete_epoch")
        self.assertFalse((self.root / "selection_weights.pt").exists())
        self.assertFalse((self.root / "holdout_predictions.npz").exists())

    def test_incomplete_second_epoch_keeps_first_completed_checkpoint(self):
        config = settings(self.root)
        model, loader = self.model_loader(config)
        originals, save = [], runtime._save_weights

        def observe_save(model, path):
            originals.append({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
            return save(model, path)

        with patch.object(runtime, "_save_weights", side_effect=observe_save):
            result = runtime._fit_selection(model, loader, loader, examples()[1], np.arange(6), config,
                                            torch.device("cpu"), self.root,
                                            ControlledBudget(stop_microbatch=5), 1.25)
        self.assertEqual(result["selected_epochs"], 1)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["stop_reason"], "budget_during_epoch")
        self.assertEqual(len(originals), 1)
        saved = torch.load(self.root / "selection_weights.pt", map_location="cpu", weights_only=True)
        for name, expected in originals[0].items():
            torch.testing.assert_close(saved[name], expected, rtol=0, atol=0)
        with np.load(self.root / "holdout_predictions.npz", allow_pickle=False) as values:
            self.assertEqual(data.score_predictions(examples()[1], values["predictions"]), result["metrics"])

    def test_incomplete_first_epoch_and_refit_are_never_promoted(self):
        config = settings(self.root)
        model, loader = self.model_loader(config)
        with self.assertRaises(runtime.BudgetExceeded):
            runtime._fit_selection(model, loader, loader, examples()[1], np.arange(6), config,
                                   torch.device("cpu"), self.root, ControlledBudget(stop_microbatch=2), 1.25)
        self.assertFalse((self.root / "selection_weights.pt").exists())
        with patch.object(neural, "prepare_encoder", side_effect=self.prepare):
            with self.assertRaises(runtime.BudgetExceeded):
                runtime.refit_selected(*examples(), config, 2, self.root, ControlledBudget(stop_microbatch=5))
        self.assertFalse((self.root / "final_weights.pt").exists())
        self.assertFalse((self.root / "final_model.ckpt").exists())
        self.assertEqual(len(runtime._read(self.root / "final_history.json")), 1)

    def test_slow_budget_preserves_first_fit_epoch_and_reserves_full_refit(self):
        class FitOnlyBudget(ControlledBudget):
            def __init__(self):
                super().__init__()
                self.admissions = []
                self.saved = False

            def check(self, stage, training=False):
                if stage == "save selected epoch":
                    self.saved = True
                return super().check(stage, training)

            def can_start(self, estimate, training=True):
                self.admissions.append((estimate, training))
                # A first epoch can finish, but after its checkpoint there is
                # no room for both another candidate epoch and a fresh refit.
                return not self.saved and estimate < 180.

        config = settings(self.root, max_epochs=3, patience=3)
        model, loader = self.model_loader(config)
        budget = FitOnlyBudget()
        result = runtime._fit_selection(model, loader, loader, examples()[1], np.arange(6), config,
                                        torch.device("cpu"), self.root, budget, 1.25)
        self.assertEqual([row["epoch"] for row in result["history"]], [1])
        self.assertEqual(result["stop_reason"], "budget_reserving_refit")
        self.assertEqual(budget.microbatches, len(loader))
        self.assertTrue(any(estimate >= 120 for estimate, _ in budget.admissions))
        self.assertTrue(all(training is True for _, training in budget.admissions))
        self.assertTrue((self.root / "selection_weights.pt").is_file())


def synthetic_data(root):
    for split, indices in (("train", range(90)), ("validation", range(100, 112))):
        directory = root / split
        directory.mkdir()
        for index in indices:
            target = [int((index + 3 * label) % (label + 3) < 2) for label in range(9)]
            text = " ".join(f"feature_{j}_{value}" for j, value in enumerate(target)) + f" sample_{index}"
            payload = {"utterances": [{"text": text}],
                       "symptom": [label for label, value in zip(data.TARGET_SYMPTOMS, target) if value]}
            (directory / f"fixture_{index:04d}.json").write_text(json.dumps(payload), encoding="utf-8")


class PipelineProtocolTests(unittest.TestCase):
    def test_one_candidate_one_refit_final_only_validation_resume_and_corruption(self):
        self.exercise_pipeline()

    def test_insufficient_budget_skips_refit_with_explicit_fit_only_output(self):
        self.exercise_pipeline(limited=True)

    def test_interrupted_refit_keeps_complete_fit_only_candidate(self):
        self.exercise_pipeline(interrupt_refit=True)

    def test_reference_not_improved_skips_refit_and_preserves_experimental_output(self):
        self.exercise_pipeline(not_improved=True)

    def exercise_pipeline(self, limited=False, interrupt_refit=False, not_improved=False):
        with tempfile.TemporaryDirectory(prefix=".protocol_test_", dir=HERE) as temporary:
            root = Path(temporary)
            synthetic_data(root)
            config = settings(root)
            config.pop("candidate_id")
            config.pop("positive_class_weights")
            calls, events, refits = [], [], []
            true_load = data.load_dataset
            current_run = [None]
            _, reference_texts, reference_targets = true_load(root / "train", include_targets=True)
            reference_fit, reference_holdout = data.make_split(reference_texts, reference_targets, .2, 21)
            reference_predictions = reference_targets[reference_holdout].copy()
            if not not_improved:
                reference_predictions[:, [0, 6]] = 1 - reference_predictions[:, [0, 6]]
            reference_payload = {
                "format": "mission3_training_reference_v1", "version": "fixture_previous",
                "candidate_id": "fixture_previous", "labels": list(data.TARGET_SYMPTOMS),
                "training_scope": "fit_only", "decision_rule": "categorical predict/argmax",
                "training_digest": runtime.dataset_digest(reference_texts, reference_targets),
                "training_samples": 90, "fit_samples": 72, "holdout_samples": 18,
                "split_seed": 21, "holdout_size": .2,
                "array_hashes": {
                    "fit_indices": reference.array_digest(reference_fit, "<i8"),
                    "holdout_indices": reference.array_digest(reference_holdout, "<i8"),
                    "training_targets": reference.array_digest(reference_targets, "i1"),
                    "holdout_targets": reference.array_digest(reference_targets[reference_holdout], "i1"),
                    "predictions": reference.array_digest(reference_predictions, "i1"),
                },
                "predictions_payload": {
                    "encoding": "zlib+base64/int8-c-order", "shape": list(reference_predictions.shape),
                    "sha256": reference.array_digest(reference_predictions, "i1"),
                    "data": base64.b64encode(zlib.compress(reference_predictions.tobytes())).decode("ascii"),
                },
                "metrics": data.score_predictions(reference_targets[reference_holdout], reference_predictions),
                "limitations": ["Synthetic protocol fixture; not real model performance"],
            }
            reference_path = root / "fixture_reference.json"
            reference_path.write_text(json.dumps(reference_payload, ensure_ascii=False), encoding="utf-8")
            reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            partial_audit = neural.configure_partial_finetuning(PartialEncoder())

            def load(directory, include_targets=False):
                if Path(directory) == root / "validation":
                    events.append(("validation", include_targets))
                    self.assertTrue((current_run[0] / "final_model_ready.json").is_file())
                    if include_targets:
                        self.assertTrue((current_run[0] / "mission3.csv").is_file())
                return true_load(directory, include_targets=include_targets)

            def predict_values(texts, name):
                result = np.array([[int(token.rsplit("_", 1)[1]) for token in text.split()[:9]]
                                   for text in texts], dtype=np.int8)
                return result

            def artifacts(local, weights, name):
                for child, file in (("encoder", "config.json"), ("tokenizer", "tokenizer.json")):
                    directory = local / "assets" / child
                    directory.mkdir(parents=True)
                    (directory / file).write_text("{}")
                (local / weights).write_text(name, encoding="utf-8")

            def train(texts, targets, fit, holdout, cfg, local, budget, pilot_report_path=None,
                      reference_metrics=None):
                calls.append(cfg["candidate_id"])
                events.append(("candidate", cfg["candidate_id"]))
                self.assertEqual(len(targets), 90)
                self.assertEqual(len(fit), 72)
                self.assertEqual(len(holdout), 18)
                predicted = predict_values([texts[i] for i in holdout], cfg["candidate_id"])
                metrics = data.score_predictions(targets[holdout], predicted)
                artifacts(local, "selection_weights.pt", cfg["candidate_id"])
                np.savez_compressed(local / "holdout_predictions.npz", indices=holdout,
                                    targets=targets[holdout], predictions=predicted)
                history = [{"epoch": 2, "training_seconds": 1., "epoch_seconds": 2.,
                            "holdout_macro_f1": metrics["macro_f1"], "holdout_metrics": metrics}]
                manifest = runtime._manifest(cfg, cfg["revision"], 2, len(fit), "fit_only", partial_audit)
                data.write_json(local / "candidate_model.ckpt", manifest)
                data.write_json(local / "partial_finetuning.json", partial_audit)
                pilot = {"decision": "continue_with_refit_projection", "device": "synthetic fixture",
                         "projected_feasible_epochs": 3, "includes_training_updates": True,
                         "resets_or_repeats_updates": False}
                data.write_json(local / "pilot_report.json", pilot)
                if pilot_report_path:
                    data.write_json(pilot_report_path, pilot)
                data.write_json(local / "training_metrics.json", metrics)
                data.write_json(local / "selection_history.json", history)
                return manifest, {"metrics": metrics, "history": history, "stop_reason": "max_epochs"}

            def refit(texts, targets, cfg, epochs, local, budget):
                refits.append(cfg["candidate_id"])
                events.append(("refit", cfg["candidate_id"]))
                self.assertEqual(len(texts), 90)
                self.assertEqual(len(targets), 90)
                self.assertEqual(epochs, 2)
                if interrupt_refit:
                    (local / "partial_training.txt").write_text("incomplete fixture refit")
                    raise runtime.BudgetExceeded("fixture refit microbatch")
                artifacts(local, "final_weights.pt", cfg["candidate_id"])
                data.write_json(local / "final_history.json", [{"epoch": 1}, {"epoch": 2}])
                manifest = runtime._manifest(cfg, cfg["revision"], epochs, len(targets), "all_training", partial_audit)
                data.write_json(local / "final_model.ckpt", manifest)
                return manifest, predict_values(texts[:12], cfg["candidate_id"])

            def predict(checkpoint, texts, budget=None):
                checkpoint = Path(checkpoint)
                current_run[0] = checkpoint.parent
                manifest = runtime._read(checkpoint)
                mode = (checkpoint.parent / manifest["weights"]).read_text(encoding="utf-8")
                return predict_values(texts, mode)

            controlled = ControlledBudget(admit=not limited)
            with patch.object(runtime, "Budget", return_value=controlled), \
                    patch.object(runtime, "train_candidate", side_effect=train), \
                    patch.object(runtime, "refit_selected", side_effect=refit), \
                    patch.object(runtime, "predict_checkpoint", side_effect=predict), \
                    patch.object(runtime, "_environment", return_value={"device": "synthetic adapter"}), \
                    patch.object(data, "load_dataset", side_effect=load), \
                    patch.object(reference, "REFERENCE_PATH", reference_path), \
                    patch.object(reference, "REFERENCE_FILE_SHA256", reference_sha):
                sources = {"entry.py": "# immutable fixture source"}
                result = runtime.run(config, sources)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(refits), 0 if limited or not_improved else 1)
                self.assertEqual([event for event in events if event[0] == "validation"],
                                 [("validation", False), ("validation", True)])
                status = runtime._read(result / "runtime.json")
                manifest = runtime._read(result / "mission3_ver20_model.ckpt")
                self.assertEqual(manifest["training_scope"], "fit_only" if limited or interrupt_refit or not_improved else "all_training")
                self.assertEqual(status["status"], "complete_not_improved" if not_improved else
                                 "complete_budget_limited" if limited or interrupt_refit else "complete")
                if limited:
                    self.assertEqual(status["refit_skipped_reason"], "budget_before_refit")
                if interrupt_refit:
                    self.assertEqual(status["refit_skipped_reason"], "budget_during_refit")
                    self.assertFalse((result / "final_model").exists())
                if not_improved:
                    self.assertEqual(status["refit_skipped_reason"], "training_reference_not_jointly_improved")
                    self.assertFalse((result / "final_model").exists())
                self.assertEqual(status["selected_candidate"], "large_partial_ce")
                comparison = runtime._read(result / "training_reference_comparison.json")
                self.assertEqual(comparison["comparison_scope"], "Training internal holdout only")
                self.assertEqual(comparison["both_macro_and_nausea_improved"], not not_improved)
                fit, holdout = data.make_split(*true_load(root / "train", include_targets=True)[1:], .2, 21)
                with np.load(result / "holdout_predictions.npz", allow_pickle=False) as values:
                    np.testing.assert_array_equal(values["indices"], holdout)
                    self.assertEqual(data.score_predictions(values["targets"], values["predictions"]),
                                     runtime._read(result / "training_metrics.json"))
                config["resume_run"] = str(result)
                old_calls, old_events = list(calls), list(events)
                self.assertEqual(runtime.run(config, sources), result)
                self.assertEqual(old_calls, calls)
                self.assertEqual(old_events, events)
                old_status = (result / "runtime.json").read_bytes()
                old_completion = (result / "run_complete.json").read_bytes()
                with patch.object(runtime, "_verify_hashes", side_effect=runtime.BudgetExceeded("fixture hash verification")):
                    self.assertEqual(runtime.run(config, sources), result)
                self.assertEqual((result / "runtime.json").read_bytes(), old_status)
                self.assertEqual((result / "run_complete.json").read_bytes(), old_completion)
                with self.assertRaisesRegex(ValueError, "changed|mismatch"):
                    runtime.run({**config, "seed": 22}, sources)
                self.assertEqual((result / "runtime.json").read_bytes(), old_status)
                with self.assertRaisesRegex(ValueError, "changed"):
                    runtime.run(config, {"entry.py": "# changed fixture source"})
                # Candidate cache is checked directly because completed runs
                # intentionally verify only their final published artifacts.
                name, weights = runtime.CANDIDATES[0]
                cached_path = result / "experiments" / name / "completed.json"
                original_cache = cached_path.read_bytes()
                tampered = runtime._read(cached_path)
                tampered["settings"]["positive_class_weights"][0] = 99.
                data.write_json(cached_path, tampered)
                _, texts, targets = true_load(root / "train", include_targets=True)
                cfg = {**runtime.validate_settings(config), "candidate_id": name, "positive_class_weights": weights}
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    runtime._candidate_record(cfg, texts, targets, fit, holdout, result, controlled)
                cached_path.write_bytes(original_cache)
                runtime._candidate_record(cfg, texts, targets, fit, holdout, result, controlled)
                self.assertEqual(old_calls, calls)
                (result / manifest["weights"]).write_text("corrupted fixture weights")
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    runtime.run(config, sources)
                self.assertEqual((result / "runtime.json").read_bytes(), old_status)
                self.assertEqual((result / "run_complete.json").read_bytes(), old_completion)


if __name__ == "__main__":
    unittest.main()
