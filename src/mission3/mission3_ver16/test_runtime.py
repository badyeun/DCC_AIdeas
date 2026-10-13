"""Actual CPU training checks and synthetic protocol checks; no real F1 claims."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parent))
    from mission3_ver16 import data, neural, runtime
else:
    from . import data, neural, runtime
from mission3_ver14.test_neural import FakeEncoder, FakeTokenizer


def settings(root, **overrides):
    return {
        "version": "ver16", "backend": "neural", "model_id": "fixture/encoder", "revision": "a" * 40,
        "input_zip": "", "train_dir": str(root / "train"), "validation_dir": str(root / "validation"),
        "output_dir": str(root / "results"), "work_dir": str(root / "work"), "resume_run": "",
        "run_validation": True, "seed": 21, "holdout_size": .2, "batch_size": 2,
        "gradient_accumulation": 2, "max_epochs": 3, "scheduler_horizon_epochs": 4, "patience": 2, "max_length": 8,
        "overlap": 2, "encoder_window_batch": 2, "learning_rate": .01, "head_learning_rate": .02,
        "refit_full_training": True, "allow_cpu": True, "log_every": 100,
        "candidate_id": "fixture", "positive_class_weights": list(runtime.POSITIVE_CLASS_WEIGHTS), "label_correlation": False,
        "correlation_hidden": 32, "correlation_depth": 1, "correlation_dropout": .1,
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
        model = neural.DialogueClassifier(FakeEncoder(), label_attention=False, contrastive=False, feature_dim=16)
        model.encoder_window_batch = config["encoder_window_batch"]
        loader = runtime._loader(texts, targets, FakeTokenizer(), config, False)
        return model, loader

    def test_real_epoch_uses_one_forward_and_vector_loss_per_microbatch(self):
        config = settings(self.root)
        model, loader = self.model_loader(config)
        original = neural.classification_loss
        calls, forwards, gradients = [], [], []
        before = model.encoder.embedding.weight.detach().clone()

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
        self.assertFalse(torch.equal(before, model.encoder.embedding.weight))
        self.assertEqual(row["ce_views"], 1)
        self.assertEqual(row["scheduler_horizon_epochs"], 4)
        self.assertEqual(row["positive_target_counts"], examples()[1].sum(0).tolist())

    def prepare(self, model_id, revision, assets):
        encoder, tokenizer = FakeEncoder(), FakeTokenizer()
        for name, value in (("encoder", encoder.config), ("tokenizer", tokenizer)):
            directory = assets / name
            directory.mkdir(parents=True)
            value.save_pretrained(directory)
        return encoder, tokenizer, revision

    def test_fresh_three_epoch_refit_trains_all_targets_on_four_epoch_schedule(self):
        texts, targets = examples()
        config = settings(self.root)
        initial_encoder = FakeEncoder()
        with torch.no_grad():
            initial_encoder.embedding.weight.fill_(99)
        torch.save(initial_encoder.state_dict(), self.root / "selection_weights.pt")
        prepared, observed_targets, learning_rates, tokens = [], [], [], []
        original_train, optimizer_type = runtime._train_epoch, torch.optim.AdamW

        def prepare(*args):
            encoder, tokenizer, revision = self.prepare(*args)
            prepared.append((encoder, encoder.embedding.weight.detach().clone()))
            return encoder, tokenizer, revision

        def observe_epoch(model, loader, *args):
            observed_targets.append(loader.dataset.targets.numpy().copy())
            hook = model.register_forward_pre_hook(lambda module, arguments:
                                                  tokens.extend(arguments[0]["input_ids"][:, 0, 1].tolist()))
            try:
                return original_train(model, loader, *args)
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
        self.assertFalse(torch.equal(prepared[0][1], prepared[0][0].embedding.weight))
        for observed in observed_targets:
            np.testing.assert_array_equal(observed, targets)
        self.assertEqual(len(observed_targets), 3)
        self.assertEqual(Counter(tokens), Counter({11 + i: 3 for i in range(6)}))
        self.assertEqual(manifest["fitted_samples"], 6)
        self.assertEqual(manifest["training_scope"], "all_training")
        self.assertEqual(manifest["selected_epochs"], 3)
        # Three batches / accumulation two => two updates per epoch, horizon eight.
        expected_rates = np.array([[.01, .02]]) * np.array([1., 1., 6/7, 5/7, 4/7, 3/7])[:, None]
        np.testing.assert_allclose(learning_rates, expected_rates, atol=1e-12, rtol=1e-12)
        history = runtime._read(self.root / "final_history.json")
        self.assertEqual([r["scheduler_horizon_epochs"] for r in history], [4, 4, 4])
        restored = neural.DialogueClassifier(FakeEncoder(), label_attention=False, contrastive=False)
        restored.encoder_window_batch = 2
        runtime._load_weights(restored, self.root / manifest["weights"])
        actual = runtime._predict(restored, runtime._loader(texts, None, FakeTokenizer(), config, False), torch.device("cpu"))
        np.testing.assert_array_equal(actual, expected)

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

    def test_slow_budget_prioritizes_three_fit_epochs_before_reserving_refit(self):
        class FitOnlyBudget(ControlledBudget):
            def __init__(self):
                super().__init__()
                self.admissions = []

            def can_start(self, estimate, training=True):
                self.admissions.append((estimate, training))
                # Synthetic admission policy: there is room for a fit epoch,
                # but not for the additional 120-second refit setup reserve.
                return estimate < 120.

        config = settings(self.root, max_epochs=3, patience=3)
        model, loader = self.model_loader(config)
        budget = FitOnlyBudget()
        result = runtime._fit_selection(model, loader, loader, examples()[1], np.arange(6), config,
                                        torch.device("cpu"), self.root, budget, 1.25)
        self.assertEqual([row["epoch"] for row in result["history"]], [1, 2, 3])
        self.assertEqual(result["stop_reason"], "max_epochs")
        self.assertEqual(budget.microbatches, 3 * len(loader))
        self.assertEqual(len(budget.admissions), 4)
        self.assertEqual([estimate < 120 for estimate, _ in budget.admissions],
                         [False, True, False, True])
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
    def test_two_candidates_one_refit_final_only_validation_resume_and_corruption(self):
        self.exercise_pipeline()

    def test_insufficient_budget_skips_candidate_and_refit_with_explicit_fit_only_output(self):
        self.exercise_pipeline(limited=True)

    def test_interrupted_refit_keeps_complete_fit_only_candidate(self):
        self.exercise_pipeline(interrupt_refit=True)

    def exercise_pipeline(self, limited=False, interrupt_refit=False):
        with tempfile.TemporaryDirectory(prefix=".protocol_test_", dir=HERE) as temporary:
            root = Path(temporary)
            synthetic_data(root)
            config = settings(root)
            config.pop("candidate_id")
            config.pop("positive_class_weights")
            calls, events, refits = [], [], []
            true_load = data.load_dataset
            current_run = [None]

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
                if name == "baseline_ce":
                    result[:, 0] = 1 - result[:, 0]
                return result

            def artifacts(local, weights, name):
                for child, file in (("encoder", "config.json"), ("tokenizer", "tokenizer.json")):
                    directory = local / "assets" / child
                    directory.mkdir(parents=True)
                    (directory / file).write_text("{}")
                (local / weights).write_text(name, encoding="utf-8")

            def train(texts, targets, fit, holdout, cfg, local, budget):
                calls.append(cfg["candidate_id"])
                events.append(("candidate", cfg["candidate_id"]))
                self.assertEqual(len(targets), 90)
                self.assertEqual(len(fit), 72)
                self.assertEqual(len(holdout), 18)
                self.assertEqual(cfg["positive_class_weights"], runtime.POSITIVE_CLASS_WEIGHTS)
                self.assertEqual(cfg["label_correlation"], cfg["candidate_id"] == "cornet")
                self.assertEqual(cfg["max_epochs"], 3)
                self.assertEqual(cfg["scheduler_horizon_epochs"], 4)
                predicted = predict_values([texts[i] for i in holdout], cfg["candidate_id"])
                metrics = data.score_predictions(targets[holdout], predicted)
                artifacts(local, "selection_weights.pt", cfg["candidate_id"])
                np.savez_compressed(local / "holdout_predictions.npz", indices=holdout,
                                    targets=targets[holdout], predictions=predicted)
                history = [{"epoch": 2, "training_seconds": 1., "epoch_seconds": 2.,
                            "holdout_macro_f1": metrics["macro_f1"], "holdout_metrics": metrics}]
                manifest = runtime._manifest(cfg, cfg["revision"], 2, len(fit), "fit_only")
                data.write_json(local / "candidate_model.ckpt", manifest)
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
                manifest = runtime._manifest(cfg, cfg["revision"], epochs, len(targets), "all_training")
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
                    patch.object(data, "load_dataset", side_effect=load):
                sources = {"entry.py": "# immutable fixture source"}
                result = runtime.run(config, sources)
                self.assertEqual(len(calls), 1 if limited else 2)
                self.assertEqual(len(refits), 0 if limited else 1)
                self.assertEqual([event for event in events if event[0] == "validation"],
                                 [("validation", False), ("validation", True)])
                status = runtime._read(result / "runtime.json")
                manifest = runtime._read(result / "mission3_ver16_model.ckpt")
                self.assertEqual(manifest["training_scope"], "fit_only" if limited or interrupt_refit else "all_training")
                self.assertEqual(status["status"], "complete_budget_limited" if limited or interrupt_refit else "complete")
                if limited:
                    self.assertEqual(status["refit_skipped_reason"], "budget_before_refit")
                if interrupt_refit:
                    self.assertEqual(status["refit_skipped_reason"], "budget_during_refit")
                    self.assertFalse((result / "final_model").exists())
                self.assertEqual(status["selected_candidate"], "baseline_ce" if limited else "cornet")
                self.assertEqual(manifest["label_correlation"], not limited)
                self.assertEqual(manifest["correlation_hidden"], 32)
                self.assertEqual(manifest["correlation_depth"], 1)
                self.assertEqual(manifest["correlation_dropout"], .1)
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
                with self.assertRaisesRegex(ValueError, "changed"):
                    runtime.run({**config, "seed": 22}, sources)
                self.assertEqual((result / "runtime.json").read_bytes(), old_status)
                with self.assertRaisesRegex(ValueError, "changed"):
                    runtime.run(config, {"entry.py": "# changed fixture source"})
                # Candidate cache is checked directly because completed runs
                # intentionally verify only their final published artifacts.
                candidate = runtime.CANDIDATES[0]
                name, weights = candidate["candidate_id"], list(runtime.POSITIVE_CLASS_WEIGHTS)
                cached_path = result / "experiments" / name / "completed.json"
                original_cache = cached_path.read_bytes()
                tampered = runtime._read(cached_path)
                tampered["settings"]["positive_class_weights"][0] = 99.
                data.write_json(cached_path, tampered)
                _, texts, targets = true_load(root / "train", include_targets=True)
                cfg = {**runtime.validate_settings(config), **candidate, "positive_class_weights": weights}
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
