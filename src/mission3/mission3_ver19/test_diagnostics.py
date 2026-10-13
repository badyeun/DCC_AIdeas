"""Synthetic checks for diagnostic accounting without training data mutations."""
import json
import unittest
import numpy as np

from .diagnostics import compare_nausea_groups


class DiagnosticTests(unittest.TestCase):
    def test_paired_transitions_group_membership_and_no_mutations(self):
        texts = ["속이 울렁거리세요?\n아니요.", "구토했어요.", "어지러워요.", "특별한 증상이 없어요."]
        gold = np.zeros((4, 9), dtype=np.int8)
        gold[[0, 1], 6] = 1
        gold[1, 1] = 1; gold[2, 4] = 1
        baseline, challenger = np.zeros_like(gold), np.zeros_like(gold)
        baseline[[1, 2], 6] = 1
        challenger[[0, 3], 6] = 1
        originals = [a.copy() for a in (gold, baseline, challenger)]
        result = compare_nausea_groups(texts, gold, baseline, challenger)
        all_rows = result["groups"]["all"]
        for key in ("fn_to_tp", "tp_to_fn", "fp_to_tn", "tn_to_fp"):
            self.assertEqual(all_rows[key], 1)
        self.assertEqual(all_rows["samples"], 4)
        self.assertEqual(result["groups"]["direct_near_question_expression"]["samples"], 1)
        self.assertEqual(result["groups"]["direct_near_negation_expression"]["samples"], 1)
        self.assertEqual(result["groups"]["gold_dizziness_without_nausea"]["fp_to_tn"], 1)
        self.assertEqual(result["groups"]["gold_vomiting_without_nausea"]["samples"], 0)
        for old, current in zip(originals, (gold, baseline, challenger)):
            np.testing.assert_array_equal(old, current)
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertNotIn(texts[0], serialized)
        self.assertNotIn("아니요", serialized)
        self.assertFalse(result["affects_training_or_selection"])

    def test_distant_negation_does_not_count_as_local_negation(self):
        gold = np.zeros((1, 9), dtype=np.int8)
        report = compare_nausea_groups(["아니요.\n중간 발화\n다른 말\n속이 메스꺼워요."], gold, gold, gold)
        self.assertEqual(report["groups"]["direct_near_negation_expression"]["samples"], 0)
        self.assertEqual(report["groups"]["direct_without_near_question_or_negation_expression"]["samples"], 1)

    def test_empty_text_supported_but_invalid_alignment_rejected(self):
        gold = np.zeros((1, 9), dtype=np.int8)
        report = compare_nausea_groups([""], gold, gold, gold)
        self.assertEqual(report["groups"]["no_direct_expression_match"]["samples"], 1)
        for texts, targets in [([], gold), ([None], gold), ([""], np.zeros((1, 8))), ([""], np.ones((1, 9)) * 2)]:
            with self.assertRaises(ValueError):
                compare_nausea_groups(texts, targets, gold, gold)


if __name__ == "__main__":
    unittest.main()
