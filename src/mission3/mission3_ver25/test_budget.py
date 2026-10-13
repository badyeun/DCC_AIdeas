"""Verify finite and unlimited admission, supervision, and error cleanup."""
import json
import subprocess
import sys
import unittest
from unittest import mock

from . import budget


class BudgetTests(unittest.TestCase):
    def test_unlimited_admits_long_work_and_serializes_without_infinity(self):
        clock = mock.Mock(return_value=10**9)
        unlimited = budget.Budget(None, 1500, 1.35, clock=clock)
        for training in (True, False):
            self.assertIsNone(unlimited.remaining(training=training))
            self.assertIsNone(unlimited.check("long training", training=training))
            self.assertTrue(unlimited.can_start(10**12, training=training))
        self.assertEqual(json.dumps({"remaining_seconds": unlimited.remaining()}, allow_nan=False),
                         '{"remaining_seconds": null}')
        clock.assert_not_called()
        for estimate in (-1, True, float("nan"), float("inf")):
            with self.subTest(estimate=estimate), self.assertRaises(ValueError):
                unlimited.can_start(estimate)

    def test_finite_budget_preserves_reserve_admission_and_expiration(self):
        finite = budget.Budget(200, 50, 1.5, clock=lambda: 100)
        self.assertEqual(finite.remaining(), 100)
        self.assertEqual(finite.check("training", training=True), 50)
        self.assertTrue(finite.can_start(30))
        self.assertFalse(finite.can_start(40))
        with self.assertRaises(budget.BudgetExceeded):
            budget.Budget(100, clock=lambda: 100).check("expired")

    def test_unlimited_supervisor_waits_without_timeout(self):
        process = mock.Mock()
        process.wait.return_value = 0
        with mock.patch.object(budget.subprocess, "Popen", return_value=process), \
             mock.patch.object(budget, "_stop_process") as stop:
            self.assertEqual(budget.run_supervised(["fixture"], None), 0)
        process.wait.assert_called_once_with(timeout=None)
        stop.assert_not_called()
        self.assertEqual(budget.run_supervised([sys.executable, "-c", "pass"], None), 0)

    def test_unlimited_error_and_interrupt_still_cleanup_and_propagate(self):
        for failure in (None, KeyboardInterrupt()):
            process = mock.Mock()
            process.wait.return_value = 7
            process.wait.side_effect = failure
            expected = subprocess.CalledProcessError if failure is None else KeyboardInterrupt
            with self.subTest(failure=failure), \
                 mock.patch.object(budget.subprocess, "Popen", return_value=process), \
                 mock.patch.object(budget.time, "monotonic", return_value=100), \
                 mock.patch.object(budget, "_stop_process") as stop:
                with self.assertRaises(expected):
                    budget.run_supervised(["fixture"], None)
                stop.assert_called_once_with(process, 110)
            process.wait.assert_called_once_with(timeout=None)

    def test_finite_supervisor_still_rejects_expired_work_and_stops_timeout(self):
        process = mock.Mock()
        process.wait.side_effect = subprocess.TimeoutExpired(["fixture"], 90)
        with mock.patch.object(budget.subprocess, "Popen", return_value=process) as popen, \
             mock.patch.object(budget.time, "monotonic", return_value=100), \
             mock.patch.object(budget, "_stop_process") as stop:
            with self.assertRaises(budget.BudgetExceeded):
                budget.run_supervised(["fixture"], 105)
            popen.assert_not_called()
            with self.assertRaises(budget.BudgetExceeded):
                budget.run_supervised(["fixture"], 200)
            process.wait.assert_called_once_with(timeout=90)
            stop.assert_called_once_with(process, 200)


if __name__ == "__main__":
    unittest.main()
