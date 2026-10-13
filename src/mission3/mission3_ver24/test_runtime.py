"""Experiment selection, reserve, privacy, and complete pipeline checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, RobertaConfig, RobertaModel

from . import data, runtime, trainer
from .budget import Budget, BudgetExceeded
from .build_notebook import default_settings


class RuntimeTests(unittest.TestCase):
    def test_medical_selection_requires_both_improvement_and_nausea_guard(self):
        control = dict(macro_f1=.65, nausea_f1=.41, selected_epochs=3)
        for macro, nausea, chosen in ((.66, .41, "medical"), (.65, .45, "control"),
                                       (.66, .40, "control"), (.64, .42, "control")):
            result = runtime.choose_candidate({"control": control,
                "medical": dict(macro_f1=macro, nausea_f1=nausea, selected_epochs=2)})
            self.assertEqual(result["selected_candidate"], chosen)
            self.assertEqual(result["selected_epochs"], 2 if chosen == "medical" else 3)

    def test_budget_plan_reserves_both_candidates_and_worst_full_refit(self):
        probes = {name: {"seconds_per_update": 1., "seconds_per_prediction": .01} for name in runtime.ORDER}
        settings = default_settings()
        generous = Budget(10000, 1200, 1.35, clock=lambda: 0)
        plan = runtime.plan_epochs(probes, 23360, 29200, 5840, settings, generous)
        self.assertEqual(plan["common_epochs"], 1)
        self.assertTrue(generous.can_start(plan["estimated_work_seconds"]))
        impossible = Budget(2000, 1200, 1.35, clock=lambda: 0)
        with self.assertRaises(BudgetExceeded):
            runtime.plan_epochs(probes, 23360, 29200, 5840, settings, impossible)

    def test_settings_enforce_recipe_and_budget(self):
        runtime.validate_settings(default_settings())
        for patch in ({"threshold": .3}, {"max_epochs": 4}, {"time_budget_minutes": 181},
                      {"pilot_optimizer_steps": 1}, {"learning_rate": .01}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                runtime.validate_settings({**default_settings(), **patch})

    def test_internal_inference_reserves_finalization(self):
        budget = Budget(100, 20, clock=lambda: 85)
        self.assertGreater(budget.check("export"), 0)
        with self.assertRaises(BudgetExceeded):
            runtime._TrainingBudget(budget).check("internal prediction", training=False)

    def test_refit_is_rejected_before_using_a_fit_only_model_as_final(self):
        selection = {"selected_candidate": "control", "selected_epochs": 2}
        record = {"actual_epoch_seconds": 500, "fitted_samples": 8}
        plan = {"estimates": {"control": {"full_epoch_seconds": 600}}}
        budget = Budget(1300, 1200, clock=lambda: 0)
        with mock.patch.object(runtime, "_create") as create, self.assertRaises(BudgetExceeded):
            runtime._refit({}, selection, record, ["text"] * 10, np.zeros((10, 9)),
                           default_settings(), plan, "cpu", budget, Path("unused"))
        create.assert_not_called()

    def test_duplicate_texts_cannot_cross_fit_and_holdout(self):
        texts = [f"dialogue {i}" for i in range(60)] * 2
        targets = np.asarray([[(i + j) % 3 == 0 for j in range(9)] for i in range(120)], dtype=np.int8)
        fit, holdout = data.make_split(texts, targets)
        self.assertFalse({data.text_group(texts[i]) for i in fit} & {data.text_group(texts[i]) for i in holdout})
        self.assertEqual(len(fit) + len(holdout), len(texts))

    def test_inference_input_ignores_label_and_other_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(json.dumps({"utterances": [{"text": "synthetic dialogue", "speaker": "hidden"}],
                "symptom": {"malformed": "must not inspect"}, "chief_complaint": "hidden"}), encoding="utf-8")
            _, texts, labels = data.load_dataset(directory, include_targets=False)
            self.assertEqual(texts, ["synthetic dialogue"])
            self.assertIsNone(labels)


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_tiny_encoders_complete_and_saved_inference_matches(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "환자", "오심", "구토", "열", "없음", "있음"]
            vocab_path = root / "vocab.txt"
            vocab_path.write_text("\n".join(vocab), encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
            snapshots = {}
            for name, config_type, model_type in (("medical", BertConfig, BertModel),
                                                   ("control", RobertaConfig, RobertaModel)):
                folder = root / name
                config = config_type(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=24, max_position_embeddings=514, pad_token_id=0)
                model_type(config, add_pooling_layer=False).save_pretrained(folder)
                tokenizer.save_pretrained(folder)
                snapshots[name] = folder
            for kind, count in (("train", 320), ("validation", 18)):
                directory = root / kind
                directory.mkdir()
                for i in range(count):
                    document = {"utterances": [{"text": f"환자 오심 없음 구토 열 있음 {kind} {i}"}],
                        "symptom": [symptom for j, symptom in enumerate(data.TARGET_SYMPTOMS) if (i + j * 7) % 5 < 2]}
                    (directory / f"synthetic_{i:04d}.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            settings = {**default_settings(), "input_zip": "", "output_dir": str(root / "results"),
                "work_dir": str(root / "work"), "train_dir": str(root / "train"),
                "validation_dir": str(root / "validation"), "allow_cpu": True, "max_epochs": 1}
            calls = []
            original_load = data.load_dataset

            def create(spec, current, device):
                name = "medical" if "bi-KM" in spec["model_id"] else "control"
                calls.append(name)
                model, tok = trainer.create_model(str(snapshots[name]), seed=current["seed"], local_files_only=True)
                return model.to(device), tok

            def load_dataset(directory, include_targets=False):
                if Path(directory).name == "validation":
                    self.assertEqual(len(calls), 5)
                    run_dir = next((root / "results").iterdir())
                    self.assertTrue((run_dir / "portable_verification.json").exists())
                    self.assertTrue((run_dir / "final_selection.json").exists())
                return original_load(directory, include_targets)

            with mock.patch.object(runtime, "_create", side_effect=create), \
                    mock.patch.object(data, "load_dataset", side_effect=load_dataset), contextlib.redirect_stdout(io.StringIO()):
                result = runtime.run(settings, runtime_sources={"__init__.py": ""})
            complete = runtime.read_json(result / "run_complete.json")
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["fitted_samples"], 320)
            self.assertEqual(calls[:4], ["medical", "control", "medical", "control"])
            self.assertTrue(runtime.read_json(result / "portable_verification.json")["decisions_equal"])
            metadata = runtime.read_json(result / "mission3_ver24_model.ckpt")
            self.assertEqual(metadata["training_scope"], "all_training")
            self.assertEqual(metadata["selected_epochs"], 1)
            runtime.inference(result / "mission3_ver24_model.ckpt", root / "validation", root / "again.csv")
            self.assertEqual((root / "again.csv").read_bytes(), (result / "mission3.csv").read_bytes())
            for relative, digest in complete["hashes"].items():
                self.assertEqual(runtime._hash(result / relative), digest)
            review = (result / "training_review_candidates.csv").read_text(encoding="utf-8-sig")
            self.assertNotIn("환자", review)
            self.assertNotIn("synthetic_", review)

    def test_failed_stage_does_not_publish_complete_marker(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            settings = {**default_settings(), "output_dir": temp, "allow_cpu": True}
            with mock.patch.object(runtime, "prepare_data", side_effect=ValueError("synthetic failure")), \
                    self.assertRaises(ValueError):
                runtime.run(settings)
            root = next(Path(temp).iterdir())
            self.assertFalse((root / "run_complete.json").exists())
            self.assertEqual(runtime.read_json(root / "runtime.json")["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
