"""Small offline fixtures for data integrity, grouped splitting, and ZIP safety."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

try:
    from . import bootstrap, data
except ImportError:
    import bootstrap
    import data


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def annotation(self, name, symptom, text=" 첫 발화\t정리 "):
        source = {"utterances": [{"text": text, "speaker": "metadata"}, {"text": "다음 발화"}],
                  "symptom": symptom, "audioPath": "unused.wav"}
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        return path

    def test_text_only_loading_preserves_turns_and_all_zero_rows(self):
        self.annotation("first.json", ["오심", "구토"])
        self.annotation("second.json", ["other symptom"])
        self.annotation("third.json", [])
        paths, texts, labels = data.load_dataset(self.root, include_targets=True)
        self.assertEqual([path.name for path in paths], ["first.json", "second.json", "third.json"])
        self.assertEqual(texts, ["첫 발화 정리\n다음 발화"] * 3)
        self.assertEqual(labels.sum(axis=1).tolist(), [2, 0, 0])

    def test_inference_ignores_poisoned_symptom(self):
        self.annotation("input.json", {"must": "never be interpreted"})
        paths, texts, labels = data.load_dataset(self.root)
        self.assertEqual(len(paths), 1)
        self.assertEqual(texts, ["첫 발화 정리\n다음 발화"])
        self.assertIsNone(labels)
        with self.assertRaisesRegex(ValueError, "symptom"):
            data.load_dataset(self.root, include_targets=True)

    def test_strict_utterance_schema_and_missing_targets(self):
        with self.assertRaisesRegex(ValueError, "utterances"):
            data.dialogue_text({})
        with self.assertRaisesRegex(ValueError, "string text"):
            data.dialogue_text({"utterances": [{"text": 42}]})
        with self.assertRaisesRegex(ValueError, "symptom target"):
            data.target_vector({"utterances": []})
        self.assertEqual(data.dialogue_text({"utterences": [{"text": "지원"}]}), "지원")

    def test_duplicate_file_identifiers_fail(self):
        self.annotation("a/same.json", [])
        self.annotation("b/same.json", [])
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            data.load_dataset(self.root)

    def test_split_is_deterministic_keeps_duplicate_conflicts_together(self):
        rng = np.random.default_rng(93)
        labels = (rng.random((240, 9)) < .27).astype(np.int8)
        texts = [f"sample {index}" for index in range(len(labels))]
        texts.extend(["sample 0", "sample   0"])
        labels = np.concatenate([labels, np.ones((2, 9), dtype=np.int8)])
        # Force the dependency-free path even if iterstrat is installed later.
        with patch.dict(sys.modules, {"iterstrat": None, "iterstrat.ml_stratifiers": None}):
            fit, holdout = data.make_split(texts, labels)
            repeat_fit, repeat_holdout = data.make_split(texts, labels)
        np.testing.assert_array_equal(fit, repeat_fit)
        np.testing.assert_array_equal(holdout, repeat_holdout)
        self.assertEqual(set(fit) | set(holdout), set(range(len(texts))))
        self.assertFalse(set(fit) & set(holdout))
        self.assertFalse({data.text_group(texts[index]) for index in fit}
                         & {data.text_group(texts[index]) for index in holdout})
        self.assertTrue({0, 240, 241} <= set(fit) or {0, 240, 241} <= set(holdout))
        for subset in (fit, holdout):
            self.assertTrue(np.all(labels[subset].sum(axis=0) > 0))
            self.assertTrue(np.all(labels[subset].sum(axis=0) < len(subset)))
        self.assertLess(abs(len(holdout) / len(texts) - .2), .06)

    def test_unsplittable_class_fails_explicitly(self):
        labels = np.zeros((30, 9), dtype=np.int8)
        labels[0] = 1
        with self.assertRaisesRegex(ValueError, "at least two text groups"):
            data.make_split([str(index) for index in range(30)], labels)

    def test_sparse_positive_groups_are_reserved_for_both_sides(self):
        labels = np.concatenate([np.ones((2, 9), dtype=np.int8), np.zeros((18, 9), dtype=np.int8)])
        with patch.dict(sys.modules, {"iterstrat": None, "iterstrat.ml_stratifiers": None}):
            fit, holdout = data.make_split([f"row {index}" for index in range(20)], labels)
        self.assertTrue(np.all(labels[fit].sum(axis=0) == 1))
        self.assertTrue(np.all(labels[holdout].sum(axis=0) == 1))

    def test_discrete_submission_has_exact_header_and_symptom_order(self):
        predicted = np.zeros((2, 9), dtype=np.int8)
        predicted[0, [1, 6]] = 1
        output = self.root / "inference.csv"
        result = data.write_predictions(["a.json", "b.json"], predicted, output)
        np.testing.assert_array_equal(result, predicted)
        with output.open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(list(csv.reader(handle)), [
                ["label file name", "symptom"], ["a.json", "['구토', '오심']"], ["b.json", "[]"]])
        with self.assertRaisesRegex(ValueError, "discrete"):
            data.write_predictions(["a.json", "b.json"], predicted * .7, output)

    def test_discrete_scores_include_symptom_false_positive_and_false_negative(self):
        labels = np.concatenate([np.ones((1, 9), dtype=np.int8), np.zeros((1, 9), dtype=np.int8)])
        self.assertEqual(data.score_predictions(labels, labels)["macro_f1"], 1.0)
        predicted = labels.copy()
        predicted[:, 6] = 1 - predicted[:, 6]
        metrics = data.score_predictions(labels, predicted)
        self.assertAlmostEqual(metrics["macro_f1"], 8 / 9)
        self.assertEqual(metrics["per_symptom"]["오심"]["false_positive"], 1)
        self.assertEqual(metrics["per_symptom"]["오심"]["false_negative"], 1)
        self.assertEqual(metrics["per_symptom"]["오심"]["support"], 1)
        with self.assertRaisesRegex(ValueError, "discrete"):
            data.score_predictions(labels, predicted * .8)

    def test_aggregate_diagnostics_and_local_error_export(self):
        texts = ["울렁거리나요? 아니요 없습니다", "울렁거리나요? 아니요 없습니다", "구토를 해요"]
        labels = np.zeros((3, 9), dtype=np.int8)
        labels[1, 6] = 1
        labels[2, 1] = 1
        diagnostics = data.training_diagnostics(texts, labels)
        self.assertEqual(diagnostics["conflicting_label_groups"], 1)
        self.assertEqual(diagnostics["all_zero_samples"], 1)
        self.assertEqual(diagnostics["lexical_cooccurrence_counts"]["nausea_and_negation"], 2)
        self.assertNotIn(texts[0], json.dumps(diagnostics, ensure_ascii=False))
        paths = data.export_errors(["a.json", "b.json", "c.json"], texts, labels,
                                   np.zeros_like(labels), self.root / "errors")
        self.assertEqual(len(paths), 11)
        nausea_csv = self.root / "errors" / "errors_오심.csv"
        with nausea_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["text"], texts[1])
        self.assertEqual(rows[0]["error_type"], "FN")
        self.assertEqual(rows[0]["review_category"], "")
        self.assertEqual(rows[0]["review_note"], "")
        self.assertEqual(rows[0]["has_nausea_expression"], "1")
        self.assertEqual(rows[0]["has_question_expression"], "1")
        self.assertEqual(rows[0]["has_negation_expression"], "1")
        grouped = json.loads((self.root / "errors" / "nausea_vomiting_error_groups.json").read_text(encoding="utf-8"))
        self.assertEqual({name: row["n"] for name, row in grouped.items()},
                         {"neither": 1, "nausea_only": 1, "vomiting_only": 1, "both": 0})
        self.assertEqual(grouped["nausea_only"]["nausea_fn"], 1)
        self.assertEqual(grouped["vomiting_only"]["vomiting_fn"], 1)
        self.assertEqual(grouped["neither"]["nausea_tn"], 1)
        data.write_json(self.root / "diagnostics.json", diagnostics)
        self.assertEqual(json.loads((self.root / "diagnostics.json").read_text(encoding="utf-8")), diagnostics)

    def test_nausea_review_includes_controls_with_correct_source_identifiers(self):
        labels = np.zeros((4, 9), dtype=np.int8)
        predicted = np.zeros_like(labels)
        labels[:2, 6] = 1
        predicted[[0, 2], 6] = 1
        labels[[1, 3], 1] = 1
        predicted[[0, 3], 1] = 1
        texts = ["메스꺼워요? 아니요", "느글거려요", "구역질", "그냥 대화"]
        sources = [f"source_{index}.json" for index in range(4)]
        first, second = self.root / "first", self.root / "second"
        for directory in (first, second):
            data.export_errors(sources, texts, labels, predicted, directory)
        self.assertEqual((first / "review_오심.csv").read_bytes(), (second / "review_오심.csv").read_bytes())
        with (first / "review_오심.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        by_category = {row["error_type"]: row for row in rows}
        expected = {"TP": 0, "FN": 1, "FP": 2, "TN": 3}
        for category, index in expected.items():
            self.assertEqual(by_category[category]["label file name"], sources[index])
            self.assertEqual(by_category[category]["text"], texts[index])
            self.assertEqual(by_category[category]["review_category"], "")
        summary = json.loads((first / "nausea_vomiting_error_groups.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["both"]["nausea_fn"], 1)
        self.assertEqual(summary["nausea_only"]["nausea_tp"], 1)
        self.assertEqual(summary["neither"]["nausea_fp"], 1)
        self.assertEqual(summary["vomiting_only"]["vomiting_tp"], 1)
        for values in summary.values():
            for symptom in ("nausea", "vomiting"):
                self.assertEqual(sum(values[f"{symptom}_{kind}"] for kind in ("tp", "fp", "fn", "tn")), values["n"])

    def test_nausea_review_is_seeded_and_capped_per_category(self):
        labels = np.zeros((80, 9), dtype=np.int8)
        predicted = np.zeros_like(labels)
        labels[:40, 6] = 1
        predicted[:20, 6] = 1
        predicted[40:60, 6] = 1
        paths = [f"source_{index}.json" for index in range(80)]
        texts = [f"dialogue {index}" for index in range(80)]
        output = self.root / "review"
        data.export_errors(paths, texts, labels, predicted, output)
        with (output / "review_오심.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 40)
        for category in ("TP", "FP", "FN", "TN"):
            self.assertEqual(sum(row["error_type"] == category for row in rows), 10)
        self.assertEqual(len({row["label file name"] for row in rows}), 40)
        # Controls are sampled across each category, not the first sorted records.
        tp_names = {row["label file name"] for row in rows if row["error_type"] == "TP"}
        self.assertNotEqual(tp_names, set(paths[:10]))

    def test_broad_nausea_regex_is_descriptive_only(self):
        expressions = ["메스꺼", "메스껍", "메슥", "매스꺼", "울렁", "구역", "미식", "느글", "토할"]
        labels = np.zeros((len(expressions), 9), dtype=np.int8)
        original = labels.copy()
        diagnostics = data.training_diagnostics(expressions, labels)
        self.assertEqual(diagnostics["lexical_cooccurrence_counts"]["nausea"], len(expressions))
        self.assertEqual(diagnostics["lexical_cooccurrence_counts"]["nausea_mention_label_negative"], len(expressions))
        self.assertIn("do not", diagnostics["lexical_counts_note"])
        np.testing.assert_array_equal(labels, original)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self, *, bad_hash=False, bad_size=False, extra=None, manifest=True):
        annotation = json.dumps({"utterances": [{"text": "fixture"}], "symptom": []}).encode()
        files = {bootstrap.TRAIN_PREFIX + "train.json": annotation,
                 bootstrap.VALID_PREFIX + "valid.json": annotation,
                 "DCC_AIdeas/results/mission3/ver4/fold_1.ckpt": b"old model must stay in ZIP"}
        rows = [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in files.items()]
        if bad_hash:
            rows[0]["sha256"] = "0" * 64
        if bad_size:
            rows[0]["size"] += 1
        archive_path = self.root / "mission3_ver5_inputs.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, content in files.items():
                    archive.writestr(name, content)
                if manifest:
                    archive.writestr("bundle_manifest.json", json.dumps({
                        "format": "mission3_ver5_inputs_v1", "files": rows,
                        "training_count": 1, "validation_count": 1}))
                if extra is not None:
                    if isinstance(extra, str) and "\\" in extra:
                        # ZipInfo normalizes Windows separators when constructed.
                        # Preserve the malicious stored member name for this test.
                        member = zipfile.ZipInfo("placeholder")
                        member.filename = extra
                        extra = member
                    archive.writestr(extra, b"ignored")
        return archive_path, files

    def test_extracts_only_jsons_and_verifies_reuse_even_if_timestamp_unchanged(self):
        archive_path, files = self.bundle()
        destination = self.root / "extracted"
        train, valid = bootstrap.prepare_data(archive_path, destination)
        self.assertEqual(train, destination / bootstrap.TRAIN_PREFIX)
        self.assertEqual(valid, destination / bootstrap.VALID_PREFIX)
        self.assertFalse(list(destination.rglob("*.ckpt")))
        self.assertTrue((destination / bootstrap.MARKER_NAME).is_file())
        damaged = train / "train.json"
        original = damaged.stat()
        damaged.write_bytes(b"x" * original.st_size)
        os.utime(damaged, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(bootstrap.prepare_data(archive_path, destination), (train, valid))
        self.assertEqual(damaged.read_bytes(), files[bootstrap.TRAIN_PREFIX + "train.json"])
        self.assertFalse(list(destination.rglob("*.tmp")))

    def test_manifest_hash_failure_leaves_no_marker_or_partial_file(self):
        archive_path, _ = self.bundle(bad_hash=True)
        destination = self.root / "extracted"
        with self.assertRaisesRegex(ValueError, "verification failed"):
            bootstrap.prepare_data(archive_path, destination)
        self.assertFalse((destination / bootstrap.MARKER_NAME).exists())
        self.assertFalse(list(destination.rglob("*.tmp")))
        self.assertFalse(list(destination.rglob("*.json")))

    def test_manifest_size_failure_occurs_before_extraction(self):
        archive_path, _ = self.bundle(bad_size=True)
        destination = self.root / "extracted"
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            bootstrap.prepare_data(archive_path, destination)
        self.assertFalse(list(destination.iterdir()))

    def test_unsafe_paths_and_duplicate_members_fail_before_extraction(self):
        for index, name in enumerate(("../outside.json", "/outside.json", "DCC_AIdeas/../../outside.json",
                                      "DCC_AIdeas\\outside.json", "C:/outside.json",
                                      bootstrap.TRAIN_PREFIX + "train.json")):
            with self.subTest(name=name):
                archive_path, _ = self.bundle(extra=name)
                destination = self.root / f"extracted_{index}"
                with self.assertRaisesRegex(ValueError, "Unsafe|Duplicate"):
                    bootstrap.prepare_data(archive_path, destination)
                self.assertFalse(list(destination.iterdir()))

    def test_symlink_zip_member_fails(self):
        link = zipfile.ZipInfo("DCC_AIdeas/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive_path, _ = self.bundle(extra=link)
        with self.assertRaisesRegex(ValueError, "Links"):
            bootstrap.prepare_data(archive_path, self.root / "extracted")

    def test_duplicate_directories_and_file_parent_collisions_fail(self):
        for index, member in enumerate(("DCC_AIdeas/duplicate/", "DCC_AIdeas/DATA")):
            archive_path, _ = self.bundle(extra=member)
            if member.endswith("/"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(archive_path, "a") as archive:
                        archive.writestr(member, b"")
            destination = self.root / f"collision_{index}"
            with self.assertRaisesRegex(ValueError, "Duplicate|collision"):
                bootstrap.prepare_data(archive_path, destination)
            self.assertFalse(list(destination.iterdir()))

    def test_no_manifest_supported_without_extracting_old_models(self):
        archive_path, _ = self.bundle(manifest=False)
        destination = self.root / "extracted"
        train, valid = bootstrap.prepare_data(archive_path, destination)
        self.assertEqual(len(list(train.glob("*.json"))), 1)
        self.assertEqual(len(list(valid.glob("*.json"))), 1)
        self.assertFalse(list(destination.rglob("*.ckpt")))

    def test_unexpected_old_json_is_rejected_without_deleting_it(self):
        archive_path, _ = self.bundle()
        destination = self.root / "extracted"
        train, _ = bootstrap.prepare_data(archive_path, destination)
        extra = train / "stale.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "another input bundle"):
            bootstrap.prepare_data(archive_path, destination)
        self.assertTrue(extra.exists())

    def test_explicit_directories_must_be_disjoint_and_complete(self):
        train, valid = self.root / "train", self.root / "valid"
        for directory in (train, valid):
            directory.mkdir()
            (directory / "file.json").write_text("{}", encoding="utf-8")
        self.assertEqual(bootstrap.prepare_data("missing.zip", "unused", train, valid), (train, valid))
        with self.assertRaisesRegex(ValueError, "both"):
            bootstrap.prepare_data("missing.zip", "unused", train, "")
        with self.assertRaisesRegex(ValueError, "distinct"):
            bootstrap.prepare_data("missing.zip", "unused", train, train)


if __name__ == "__main__":
    unittest.main()
