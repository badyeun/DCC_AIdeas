"""Small real-process tests: no model weights, network or GPU required."""

import os
import contextlib
import io
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from mission3_ver9.budget import AbsoluteDeadline, BudgetExceeded, TimeBudget


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class TimeBudgetTests(unittest.TestCase):
    def test_deadline_reservation_and_reconstruction(self):
        clock = FakeClock()
        budget = TimeBudget(total_seconds=100, clock=clock, monotonic_clock=clock)
        self.assertIs(AbsoluteDeadline, TimeBudget)
        self.assertEqual(budget.remaining_seconds(), 100)
        self.assertEqual(budget.check(reserve_seconds=20), 75)
        clock.now += 70
        child = TimeBudget.from_deadline(budget.start_epoch, budget.deadline_epoch,
                                        clock=clock, monotonic_clock=clock)
        self.assertEqual(child.remaining_seconds(), 30)
        self.assertEqual(child.check(20), 5)
        clock.now += 5
        with self.assertRaises(BudgetExceeded):
            child.check(20)
        self.assertEqual(child.check(), 20)
        clock.now += 100
        self.assertEqual(child.remaining_seconds(), 0)
        with self.assertRaises(BudgetExceeded):
            child.check()

    def test_backward_wall_clock_does_not_extend_budget(self):
        wall, monotonic = FakeClock(), FakeClock()
        budget = TimeBudget(total_seconds=100, clock=wall, monotonic_clock=monotonic)
        wall.now -= 100
        monotonic.now += 20
        self.assertEqual(budget.remaining_seconds(), 80)

    def test_invalid_budgets_and_reserves(self):
        clock = FakeClock()
        for value in (0, -1, 10801, float("inf"), float("nan"), "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TimeBudget(total_seconds=value, clock=clock)
        for start, deadline in ((1000, 1000), (1000, 999), (1000, 11801),
                                (1001, 1002), (-1, 10), (1000, float("inf"))):
            with self.subTest(start=start, deadline=deadline), self.assertRaises(ValueError):
                TimeBudget.from_deadline(start, deadline, clock=clock)
        budget = TimeBudget(total_seconds=100, clock=clock)
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(reserve=value), self.assertRaises(ValueError):
                budget.check(value)

    def test_capture_inherited_streams_and_nonzero_exit(self):
        budget = TimeBudget(total_seconds=30)
        result = budget.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "ok\n", ""))
        inherited = budget.run([sys.executable, "-c", "pass"])
        self.assertIsNone(inherited.stdout)
        result = budget.run([sys.executable, "-c", "raise SystemExit(7)"], check=False)
        self.assertEqual(result.returncode, 7)
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            budget.run([sys.executable, "-c", "raise SystemExit(9)"])
        self.assertEqual(failure.exception.returncode, 9)

    def test_expired_budget_never_starts_child_or_touches_artifact(self):
        clock = FakeClock()
        budget = TimeBudget(total_seconds=10, clock=clock, monotonic_clock=clock)
        clock.now += 5
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "baseline.txt"
            artifact.write_text("already selected", encoding="utf-8")
            with self.assertRaises(BudgetExceeded):
                budget.run([sys.executable, "-c", "raise AssertionError('must not run')"])
            self.assertEqual(artifact.read_text(encoding="utf-8"), "already selected")

    @unittest.skipUnless(os.name == "posix" and Path("/proc").exists(), "Linux process-group test")
    def test_timeout_kills_term_ignoring_child_and_preserves_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "baseline.txt"
            artifact.write_text("keep", encoding="utf-8")
            pid_file = Path(directory) / "child.pid"
            child_code = (
                "import os,signal,time,pathlib; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
                "time.sleep(30)"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                "time.sleep(30)"
            )
            started = time.monotonic()
            with self.assertRaises(BudgetExceeded):
                TimeBudget(total_seconds=30).run([sys.executable, "-c", parent_code], timeout=0.6)
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertEqual(artifact.read_text(encoding="utf-8"), "keep")
            self.assertTrue(pid_file.exists(), "descendant must have started before timeout")
            pid = int(pid_file.read_text())
            # Linux can briefly keep an orphan zombie pending init's reaping.
            # A zombie has exited and cannot retain a live CUDA context.
            stopped = False
            for _ in range(100):
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
                except FileNotFoundError:
                    stopped = True
                    break
                if state == "Z":
                    stopped = True
                    break
                time.sleep(0.01)
            self.assertTrue(stopped, f"descendant {pid} survived timeout cleanup")

    @unittest.skipUnless(os.name == "posix", "POSIX signal handling")
    def test_ignoring_parent_is_killed_within_two_second_grace(self):
        command = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"]
        started = time.monotonic()
        with self.assertRaises(BudgetExceeded):
            TimeBudget(total_seconds=30).run(command, timeout=0.2, capture_output=True)
        self.assertLess(time.monotonic() - started, 2.6)

    def test_stream_output_is_live_without_newline_and_merges_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "output-was-visible"

            class LiveOutput(io.StringIO):
                def write(self, text):
                    written = super().write(text)
                    if "ready" in self.getvalue():
                        gate.write_text("continue", encoding="utf-8")
                    return written

            # The child can finish only after the parent's Python stdout has
            # received output. Buffering until communicate() returns deadlocks.
            code = (
                "import pathlib,sys,time; "
                f"gate=pathlib.Path({str(gate)!r}); "
                "sys.stdout.write('ready|'); sys.stdout.flush(); "
                "start=time.monotonic(); "
                "\nwhile not gate.exists() and time.monotonic()-start<3: time.sleep(.01)"
                "\nassert gate.exists(), 'parent did not stream output live'"
                "\nsys.stderr.write('stderr|'); sys.stderr.flush()"
                "\nsys.stdout.write('done\\n'); sys.stdout.flush()"
            )
            output = LiveOutput()
            with contextlib.redirect_stdout(output):
                result = TimeBudget(total_seconds=30).run([sys.executable, "-u", "-c", code],
                                                          stream_output=True, timeout=5)
            self.assertEqual(output.getvalue(), "ready|stderr|done\n")
            self.assertEqual(result.returncode, 0)
            self.assertIsNone(result.stdout)
            self.assertIsNone(result.stderr)

    def test_stream_decodes_split_multibyte_output_and_nonzero_exit(self):
        data = "학습 완료\n".encode("utf-8")
        code = (f"import os,time; data={data!r}; os.write(1,data[:1]); "
                "time.sleep(.05); os.write(1,data[1:]); raise SystemExit(7)")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = TimeBudget(total_seconds=30).run([sys.executable, "-c", code],
                                                      stream_output=True, check=False, encoding="utf-8")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(output.getvalue(), "학습 완료\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as error:
            TimeBudget(total_seconds=30).run([sys.executable, "-c", "raise SystemExit(9)"], stream_output=True)
        self.assertEqual(error.exception.returncode, 9)

    def test_stream_conflicting_options_rejected_before_start(self):
        command = [sys.executable, "-c", "raise AssertionError('must not start')"]
        for options in ({"capture_output": True}, {"stdout": subprocess.PIPE},
                        {"stderr": subprocess.PIPE}, {"text": False}, {"universal_newlines": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TimeBudget(total_seconds=30).run(command, stream_output=True, **options)

    @unittest.skipUnless(os.name == "posix", "POSIX signal handling")
    def test_stream_timeout_is_independent_of_newlines_and_term_handlers(self):
        command = [sys.executable, "-u", "-c",
                   "import signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                   "sys.stdout.write('no newline'); sys.stdout.flush(); time.sleep(30)"]
        output = io.StringIO()
        started = time.monotonic()
        with contextlib.redirect_stdout(output), self.assertRaises(BudgetExceeded):
            TimeBudget(total_seconds=30).run(command, timeout=0.2, stream_output=True)
        self.assertLess(time.monotonic() - started, 2.6)
        self.assertEqual(output.getvalue(), "no newline")

    @unittest.skipUnless(os.name == "posix", "POSIX process/pipe handling")
    def test_stream_pipe_kept_open_after_leader_exit_cannot_hang_close(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "escaped.pid"
            child_code = (f"import os,pathlib,time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
                          "time.sleep(30)")
            parent_code = ("import subprocess,sys,time; "
                           f"subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
                           "time.sleep(.1)")
            started = time.monotonic()
            try:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(BudgetExceeded):
                    TimeBudget(total_seconds=30).run([sys.executable, "-c", parent_code],
                                                     timeout=.5, stream_output=True)
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertTrue(pid_file.exists())
            finally:
                # This test deliberately escapes the managed group to exercise
                # open-pipe cleanup. Reap its live process explicitly afterwards.
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
