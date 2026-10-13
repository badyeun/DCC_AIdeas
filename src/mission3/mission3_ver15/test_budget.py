"""Exercise budget admission and real subprocess termination without ML work."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, call, patch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from budget import Budget, BudgetExceeded, run_supervised
else:
    from .budget import Budget, BudgetExceeded, run_supervised


class BudgetTests(unittest.TestCase):
    def test_shared_clock_reserves_time_only_for_training(self):
        now = [100.0]
        budget = Budget(200, finalization_reserve_s=20, clock=lambda: now[0])
        self.assertEqual(budget.remaining(), 100)
        self.assertEqual(budget.remaining(training=True), 80)
        self.assertEqual(budget.check("epoch", training=True), 80)
        now[0] = 180
        with self.assertRaises(BudgetExceeded) as caught:
            budget.check("epoch", training=True)
        self.assertEqual(caught.exception.stage, "epoch")
        self.assertEqual(budget.check("export"), 20)
        now[0] = 201
        self.assertEqual(budget.remaining(), 0)
        with self.assertRaises(BudgetExceeded):
            budget.check("export")

    def test_admission_includes_safety_factor_and_rejects_invalid_estimates(self):
        budget = Budget(200, finalization_reserve_s=20, safety_factor=1.25, clock=lambda: 100)
        self.assertTrue(budget.can_start(64))
        self.assertFalse(budget.can_start(64.01))
        self.assertTrue(budget.can_start(80, training=False))
        self.assertFalse(budget.can_start(80.01, training=False))
        for value in (True, -1, float("nan"), float("inf"), "10", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.can_start(value)
        exhausted = Budget(100, finalization_reserve_s=0, clock=lambda: 100)
        self.assertFalse(exhausted.can_start(0))

    def test_constructor_and_clock_validation(self):
        for key, value in (("deadline_monotonic", float("nan")),
                           ("finalization_reserve_s", -1), ("safety_factor", .99),
                           ("safety_factor", True), ("clock", None)):
            args = {"deadline_monotonic": 100, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                Budget(**args)
        with self.assertRaises(ValueError):
            Budget(100, clock=lambda: float("inf")).remaining()
        with self.assertRaises(ValueError):
            Budget(100, clock=lambda: 0).remaining(training=1)


class SupervisorTests(unittest.TestCase):
    def test_real_success_and_nonzero_exit(self):
        self.assertEqual(run_supervised([sys.executable, "-c", "pass"], time.monotonic() + 15), 0)
        command = [sys.executable, "-c", "raise SystemExit(7)"]
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            run_supervised(command, time.monotonic() + 15)
        self.assertEqual(caught.exception.returncode, 7)
        self.assertEqual(caught.exception.cmd, command)

    def test_real_timeout_terminates_child_inside_supplied_budget(self):
        started = time.monotonic()
        processes = []
        real_popen = subprocess.Popen

        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(subprocess, "Popen", side_effect=capture):
            with self.assertRaises(BudgetExceeded):
                run_supervised([sys.executable, "-c", "import time; time.sleep(30)"],
                               started + 2, shutdown_margin_s=.4)
        self.assertLess(time.monotonic() - started, 2.75)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_expired_deadline_and_invalid_arguments_never_launch(self):
        with patch.object(subprocess, "Popen") as popen:
            with self.assertRaises(BudgetExceeded):
                run_supervised([sys.executable], time.monotonic() + .2, shutdown_margin_s=.3)
            for command, margin in (("python", 1), ([], 1), ([sys.executable], 0)):
                with self.subTest(command=command), self.assertRaises(ValueError):
                    run_supervised(command, time.monotonic() + 20, shutdown_margin_s=margin)
            popen.assert_not_called()

    def test_unresponsive_process_gets_forced_stop_with_bounded_waits(self):
        module = sys.modules[Budget.__module__]
        process = MagicMock()
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 1.6),
                                    subprocess.TimeoutExpired("fixture", .2), 0]
        with patch.object(subprocess, "Popen", return_value=process), \
                patch.object(module.time, "monotonic", side_effect=[100, 100, 101.6, 101.6]), \
                patch.object(module, "_signal_process") as signal_process:
            with self.assertRaises(BudgetExceeded):
                run_supervised(["fixture"], 102, shutdown_margin_s=.4)
        self.assertEqual(signal_process.call_args_list, [call(process), call(process, force=True)])
        waits = [entry.kwargs["timeout"] for entry in process.wait.call_args_list]
        self.assertAlmostEqual(waits[0], 1.6)
        self.assertLessEqual(waits[1], .200001)
        self.assertLessEqual(waits[2], .400001)

    def test_keyboard_interrupt_stops_owned_process_before_propagating(self):
        module = sys.modules[Budget.__module__]
        process = MagicMock()
        process.wait.side_effect = [KeyboardInterrupt, 0, 0]
        with patch.object(subprocess, "Popen", return_value=process), \
                patch.object(module, "_signal_process") as signal_process:
            with self.assertRaises(KeyboardInterrupt):
                run_supervised(["fixture"], time.monotonic() + 30)
        self.assertEqual(signal_process.call_args_list, [call(process), call(process, force=True)])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process-group watchdog")
    def test_timeout_kills_descendant_even_if_leader_exits_on_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "descendant.json"
            child_source = (
                "import json,os,signal,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"Path({str(marker)!r}).write_text(json.dumps(os.getpid())); "
                "time.sleep(30)"
            )
            parent_source = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{child_source!r}]); "
                "time.sleep(30)"
            )
            with self.assertRaises(BudgetExceeded):
                run_supervised([sys.executable, "-c", parent_source],
                               time.monotonic() + 2.5, shutdown_margin_s=.5)
            self.assertTrue(marker.is_file(), "Descendant must start before the watchdog fires")
            descendant = json.loads(marker.read_text())
            proc_status = Path(f"/proc/{descendant}/status")
            limit = time.monotonic() + .5
            while proc_status.exists() and time.monotonic() < limit:
                state = next(line for line in proc_status.read_text().splitlines() if line.startswith("State:"))
                if "Z" in state:
                    break
                time.sleep(.01)
            if proc_status.exists():
                state = next(line for line in proc_status.read_text().splitlines() if line.startswith("State:"))
                self.assertIn("Z", state, "A killed orphan may await init's reaping, but cannot keep running")


if __name__ == "__main__":
    unittest.main()
