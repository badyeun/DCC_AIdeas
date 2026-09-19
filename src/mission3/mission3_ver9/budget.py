"""One shared wall-clock budget for setup, training and evaluation.

Create ``budget = TimeBudget()`` before the first automatic notebook step; pass
``start_epoch`` and ``deadline_epoch`` to every child and reconstruct with
``TimeBudget.from_deadline(start_epoch, deadline_epoch)``. Never create a fresh
three-hour budget for a later phase. ``check(reserve_seconds=1500)`` returns the
usable seconds before the reserved final phase, or raises ``BudgetExceeded``.
``run`` inherits stdout/stderr by default and returns a CompletedProcess. It
stops the entire child process group on timeout (Colab/Linux), retaining files
already written by the caller. Normal nonzero exits follow subprocess.run's
``check`` convention. No machine-learning dependencies are imported here.
"""

from __future__ import annotations

import math
import codecs
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence


MAX_SECONDS = 3 * 60 * 60
SAFETY_SECONDS = 5.0
TERMINATION_SECONDS = 2.0


class BudgetExceeded(TimeoutError):
    """A phase cannot continue within its shared absolute deadline."""


class _OutputRelay:
    """Forward a pipe without tying timeout handling to a blocking readline.

    Read the descriptor directly in nonblocking mode. Closing a TextIOWrapper
    while another thread holds its read lock can hang when a descendant keeps
    the pipe open, so this relay never calls the wrapper's blocking read APIs.
    """

    def __init__(self, stream):
        self.stream = stream
        self.output = sys.stdout
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._forward, name="mission3-output", daemon=True)
        os.set_blocking(stream.fileno(), False)

    def start(self):
        self.thread.start()

    def _forward(self):
        decoder = codecs.getincrementaldecoder(self.stream.encoding or "utf-8")(
            errors=self.stream.errors or "replace")
        try:
            while not self.cancel.is_set():
                try:
                    chunk = os.read(self.stream.fileno(), 65536)
                except BlockingIOError:
                    self.cancel.wait(0.02)
                    continue
                if not chunk:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        self.output.write(tail)
                        self.output.flush()
                    break
                text = decoder.decode(chunk, final=False)
                if text:
                    self.output.write(text)
                    self.output.flush()
        except BaseException as error:
            if not self.cancel.is_set():
                self.error = error
        finally:
            self.finished.set()

    def close(self):
        self.cancel.set()
        # The nonblocking reader exits promptly even if an escaped descendant
        # retains stdout. A stalled notebook output writer is a daemon thread.
        self.thread.join(timeout=0.1)


def _finite_seconds(value: float, name: str, *, positive: bool = False) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        condition = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {condition}")
    return value


class TimeBudget:
    """Absolute epoch deadline, capped at 10,800 seconds from the run start.

    ``remaining_seconds`` includes the five-second shutdown safety margin;
    ``check`` and ``run`` exclude it. ``reserve_seconds`` additionally leaves
    time for subsequent phases. ``clock`` is injectable for deterministic tests.
    A monotonic bound prevents a backward wall-clock adjustment extending a run.
    """

    def __init__(
        self,
        *,
        start_epoch: float | None = None,
        total_seconds: float = MAX_SECONDS,
        deadline_epoch: float | None = None,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        total_seconds = _finite_seconds(total_seconds, "total_seconds", positive=True)
        if total_seconds > MAX_SECONDS:
            raise ValueError(f"total_seconds cannot exceed {MAX_SECONDS}")
        now = clock()
        self.start_epoch = _finite_seconds(now if start_epoch is None else start_epoch, "start_epoch")
        self.deadline_epoch = _finite_seconds(
            self.start_epoch + total_seconds if deadline_epoch is None else deadline_epoch,
            "deadline_epoch",
        )
        if not self.start_epoch < self.deadline_epoch <= self.start_epoch + total_seconds:
            raise ValueError("deadline_epoch must be after start_epoch and within total_seconds")
        if self.start_epoch > now:
            raise ValueError("start_epoch cannot be in the future")
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._monotonic_deadline = monotonic_clock() + max(0.0, self.deadline_epoch - now)

    @classmethod
    def from_deadline(cls, start_epoch: float, deadline_epoch: float, **kwargs) -> "TimeBudget":
        """Reconstruct the same deadline in a child; does not restart the clock."""
        return cls(start_epoch=start_epoch, deadline_epoch=deadline_epoch, **kwargs)

    def remaining_seconds(self) -> float:
        """Seconds remaining to the absolute deadline, clamped to zero."""
        return max(0.0, min(
            self.deadline_epoch - self._clock(),
            self._monotonic_deadline - self._monotonic_clock(),
        ))

    def check(self, reserve_seconds: float = 0, *, phase: str = "work") -> float:
        """Return usable seconds, excluding shutdown margin and phase reserve."""
        reserve_seconds = _finite_seconds(reserve_seconds, "reserve_seconds")
        usable = self.remaining_seconds() - SAFETY_SECONDS - reserve_seconds
        if usable <= 0:
            raise BudgetExceeded(f"No time left for {phase}; preserving existing artifacts")
        return usable

    def run(
        self,
        command: Sequence[str],
        *,
        reserve_seconds: float = 0,
        timeout: float | None = None,
        check: bool = True,
        capture_output: bool = False,
        stream_output: bool = False,
        **popen_kwargs,
    ) -> subprocess.CompletedProcess:
        """Run a command within this budget, killing its process group if needed.

        ``timeout`` can shorten a phase but cannot extend the shared deadline.
        Standard Popen options such as ``cwd``, ``env``, ``text``, ``stdout`` and
        ``stderr`` are accepted. ``start_new_session`` is always true so timeout
        cleanup cannot signal the notebook's own process group.

        ``stream_output=True`` forwards merged stdout/stderr to Python's current
        stdout, which makes child logs visible in an in-process Colab notebook.
        It does not retain an unbounded copy of training output. A separate
        thread reads the pipe while the supervisor independently enforces the
        process deadline. It cannot be combined with ``capture_output``.
        """
        usable = self.check(reserve_seconds, phase="subprocess")
        if timeout is not None:
            timeout = _finite_seconds(timeout, "timeout", positive=True)
            usable = min(usable, timeout)
        if "start_new_session" in popen_kwargs:
            raise ValueError("start_new_session is managed by TimeBudget.run")
        if stream_output and capture_output:
            raise ValueError("stream_output cannot be combined with capture_output")
        if stream_output:
            if "stdout" in popen_kwargs or "stderr" in popen_kwargs:
                raise ValueError("stream_output cannot be combined with stdout or stderr")
            if popen_kwargs.get("text") is False or popen_kwargs.get("universal_newlines") is False:
                raise ValueError("stream_output requires text mode")
            popen_kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            popen_kwargs.setdefault("errors", "replace")
        if capture_output:
            if "stdout" in popen_kwargs or "stderr" in popen_kwargs:
                raise ValueError("capture_output cannot be combined with stdout or stderr")
            popen_kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        phase_deadline = time.monotonic() + usable
        process = subprocess.Popen(command, start_new_session=True, **popen_kwargs)
        relay = None
        try:
            if stream_output:
                relay = _OutputRelay(process.stdout)
                relay.start()
            usable = min(
                self.check(reserve_seconds, phase="subprocess"),
                phase_deadline - time.monotonic(),
            )
            if usable <= 0:
                raise BudgetExceeded("Subprocess startup exhausted the available budget")
            if relay is None:
                stdout, stderr = process.communicate(timeout=usable)
            else:
                # Never block the supervisor on output, line endings or EOF.
                process.wait(timeout=usable)
                drain_seconds = min(self.check(reserve_seconds, phase="output drain"),
                                    phase_deadline - time.monotonic())
                if not relay.finished.wait(timeout=max(0.0, drain_seconds)):
                    raise subprocess.TimeoutExpired(command, usable)
                if relay.error is not None:
                    raise relay.error
                stdout, stderr = None, None
        except subprocess.TimeoutExpired as exc:
            self._stop(process)
            failure = BudgetExceeded(f"Subprocess exceeded its time budget: {command!r}")
            failure.stdout, failure.stderr = exc.stdout, exc.stderr
            raise failure from exc
        except BaseException:
            self._stop(process)
            raise
        finally:
            if relay is not None:
                relay.close()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result

    def _stop(self, process: subprocess.Popen) -> None:
        """TERM then KILL the session's process group within two seconds."""
        stop_by = time.monotonic() + min(TERMINATION_SECONDS, self.remaining_seconds())

        def send(sig: int) -> None:
            try:
                if os.name == "posix":
                    # Signal even if the group leader exited: grandchildren may remain.
                    os.killpg(process.pid, sig)
                elif process.poll() is None:
                    process.terminate() if sig == signal.SIGTERM else process.kill()
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        try:
            process.wait(timeout=max(0.0, stop_by - time.monotonic() - 0.25))
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A TERM-ignoring child can survive even after its parent has exited.
            send(getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            process.wait(timeout=max(0.0, stop_by - time.monotonic()))
        except subprocess.TimeoutExpired:
            # KILL was already sent; never let OS-level reaping extend the deadline.
            pass


# Both names describe the same shared, absolute budget.
AbsoluteDeadline = TimeBudget
