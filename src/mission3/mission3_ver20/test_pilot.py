"""Measured-throughput projections preserve reserve and deadline semantics."""
from pathlib import Path
import sys
import unittest

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mission3_ver20 import runtime
    from mission3_ver20.budget import Budget
else:
    from . import runtime
    from .budget import Budget


class PilotProjectionTests(unittest.TestCase):
    def project(self, training_seconds, safety_factor=1):
        budget = Budget(100 + training_seconds + 1200, finalization_reserve_s=1200,
                        safety_factor=safety_factor, clock=lambda: 100.)
        initial_deadline = budget.deadline_monotonic
        report = runtime._pilot_projection(
            {"max_epochs": 3, "gradient_accumulation": 2},
            microbatches=22, measured_batches=20, measured_seconds=40,
            total_batches=100, holdout_batches=25, full_ratio=1.25,
            budget=budget, device="cpu")
        self.assertEqual(initial_deadline, budget.deadline_monotonic)
        self.assertEqual(report["estimated_epoch_seconds"], 200.)
        self.assertEqual(report["estimated_holdout_seconds"], 50.)
        self.assertEqual(report["observed_optimizer_steps"], 11)
        return report

    def test_full_refit_projection_uses_measured_rate(self):
        report = self.project(1800)
        self.assertEqual(report["projected_feasible_epochs"], 3)
        self.assertEqual(report["decision"], "continue_with_refit_projection")

    def test_reserve_and_safety_factor_reduce_admitted_work(self):
        self.assertEqual(self.project(1200)["projected_feasible_epochs"], 2)
        self.assertEqual(self.project(1200, safety_factor=1.25)["projected_feasible_epochs"], 1)

    def test_fit_only_risk_is_reported_without_inventing_refit_time(self):
        report = self.project(500)
        self.assertEqual(report["projected_feasible_epochs"], 0)
        self.assertEqual(report["decision"], "fit_only_risk")

    def test_unfinishable_first_epoch_requests_stop(self):
        report = self.project(300)
        self.assertEqual(report["projected_feasible_epochs"], 0)
        self.assertEqual(report["decision"], "stop_no_complete_epoch")


if __name__ == "__main__":
    unittest.main()
