"""Share a wall-clock deadline across setup, training, and final export.

Create the monotonic deadline once per notebook invocation, before package
installation. Do not restore a monotonic timestamp from an earlier Colab VM.
``Budget`` reserves finalization time when ``training=True``; subprocess
supervision enforces the actual invocation deadline. A ``None`` deadline
disables time limits while retaining normal completion and error handling.
"""
from __future__ import annotations

import math
from numbers import Real
import os
import signal
import subprocess
import time


class BudgetExceeded(TimeoutError):
    """Work stopped without declaring an incomplete artifact complete."""

    def __init__(self, stage, *, deadline_monotonic=None):
        self.stage = str(stage)
        self.deadline_monotonic = deadline_monotonic
        super().__init__(f"Wall-clock budget exhausted during {self.stage}")


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


class Budget:
    """A deadline with conservative admission checks for expensive stages.

    ``remaining(training=True)`` subtracts the export/evaluation reserve.
    ``check`` raises at an exhausted deadline and otherwise returns seconds left.
    ``can_start`` applies the safety factor to an estimated duration; it does not
    extend the deadline or admit work when no time remains.
    A ``None`` deadline admits work without a time limit and reports ``None``
    remaining, so saved reports stay valid JSON without infinity values.
    """

    def __init__(self, deadline_monotonic, finalization_reserve_s=1200,
                 safety_factor=1.25, clock=time.monotonic):
        self.deadline_monotonic = (None if deadline_monotonic is None else
                                   _finite_nonnegative(deadline_monotonic, "deadline_monotonic"))
        self.finalization_reserve_s = _finite_nonnegative(finalization_reserve_s, "finalization_reserve_s")
        self.safety_factor = _finite_nonnegative(safety_factor, "safety_factor")
        if self.safety_factor < 1:
            raise ValueError("safety_factor must be at least one")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.clock = clock

    def remaining(self, training=False):
        if not isinstance(training, bool):
            raise ValueError("training must be a boolean")
        if self.deadline_monotonic is None:
            return None
        now = _finite_nonnegative(self.clock(), "clock result")
        reserve = self.finalization_reserve_s if training else 0
        return max(0.0, self.deadline_monotonic - reserve - now)

    def check(self, stage, training=False):
        remaining = self.remaining(training=training)
        if remaining is not None and remaining <= 0:
            raise BudgetExceeded(stage, deadline_monotonic=self.deadline_monotonic)
        return remaining

    def can_start(self, estimated_seconds, training=True):
        estimate = _finite_nonnegative(estimated_seconds, "estimated_seconds")
        remaining = self.remaining(training=training)
        return remaining is None or (remaining > 0 and estimate * self.safety_factor <= remaining)


def _signal_process(process, force=False):
    """Signal only this supervisor's new process group (or Windows child)."""
    try:
        if os.name == "posix":
            # Popen(start_new_session=True) makes this child's PID its group ID.
            # Signal the group even if its leader exited, to stop descendants.
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif process.poll() is None:
            process.kill() if force else process.terminate()
    except ProcessLookupError:
        pass


def _stop_process(process, deadline_monotonic):
    """Use at most the remaining shutdown margin for graceful stop and reaping."""
    _signal_process(process)
    remaining = max(0.0, deadline_monotonic - time.monotonic())
    try:
        process.wait(timeout=min(5.0, remaining / 2))
    except subprocess.TimeoutExpired:
        pass
    # A descendant may ignore SIGTERM even when the group leader has exited.
    _signal_process(process, force=True)
    try:
        process.wait(timeout=max(0.0, deadline_monotonic - time.monotonic()))
    except subprocess.TimeoutExpired:
        # SIGKILL/TerminateProcess has already been issued; never wait unbounded.
        pass


def run_supervised(command, deadline_monotonic, shutdown_margin_s=10, cwd=None, env=None):
    """Run a command with inherited output and an optional wall-clock watchdog.

    Returns zero on success. Nonzero exit raises ``CalledProcessError``; timeout
    raises ``BudgetExceeded`` after terminating the child. POSIX descendants are
    stopped as a group; Windows supervision stops only its own direct child.
    The shutdown margin is inside the supplied deadline, not added after it.
    ``None`` waits for completion without a watchdog; error cleanup stays bounded.
    Commands must be argument sequences and are never executed through a shell.
    """
    deadline = (None if deadline_monotonic is None else
                _finite_nonnegative(deadline_monotonic, "deadline_monotonic"))
    margin = _finite_nonnegative(shutdown_margin_s, "shutdown_margin_s")
    if margin <= 0:
        raise ValueError("shutdown_margin_s must be positive")
    if isinstance(command, (str, bytes)) or not isinstance(command, (list, tuple)) or not command:
        raise ValueError("command must be a nonempty argument sequence")
    if deadline is not None and deadline - margin - time.monotonic() <= 0:
        raise BudgetExceeded("subprocess startup", deadline_monotonic=deadline)
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=os.name == "posix")
    try:
        timeout = None if deadline is None else max(0.0, deadline - margin - time.monotonic())
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_process(process, deadline if deadline is not None else time.monotonic() + margin)
        if deadline is None:
            raise
        raise BudgetExceeded("subprocess execution", deadline_monotonic=deadline) from None
    except BaseException:
        cleanup_deadline = time.monotonic() + margin
        _stop_process(process, cleanup_deadline if deadline is None else min(deadline, cleanup_deadline))
        raise
    if returncode:
        cleanup_deadline = time.monotonic() + margin
        _stop_process(process, cleanup_deadline if deadline is None else min(deadline, cleanup_deadline))
        raise subprocess.CalledProcessError(returncode, command)
    return returncode
