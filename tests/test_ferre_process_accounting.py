"""
Process-slot accounting for `astra.pipelines.aspcap.ferre`.

`ferre()` reports its resource use to the parent dispatch loop over a pipe: `n_processes=+1` when it
takes a process slot, `n_processes=-1` when it gives it back. The dispatch loop uses the running
total to enforce `max_processes`.

A hang that leads to a resume is the *same logical job continuing*, so the slot must be held across
the whole resume chain. Releasing it when the hung subprocess dies and re-acquiring it when the
resume starts leaves a window in which the dispatch loop sees a free slot and hands it to a
different grid -- producing two concurrent FERRE processes (and 2x the threads) under
`max_processes=1`.

The invariants, for one complete `ferre()` call including every recursive resume:

  1. exactly one `n_processes=-1` is emitted for the whole chain,
  2. the running total never dips below the level the chain holds until it terminates, and
  3. the net delta is 0 when `ferre()` acquired its own slot (`communicate_on_start=True`) or -1 when
     the caller acquired it beforehand (`communicate_on_start=False`).

A slot must also never be leaked on an error path: an unreleased slot permanently reduces the
dispatch loop's capacity and, at `max_processes=1`, deadlocks it.

These drive the real `ferre()` and its real monitor thread; only the FERRE subprocess itself is
faked, so the hang cases genuinely trip the watchdog rather than simulating it.
"""

import os

os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

import builtins
import tempfile
import threading
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class RecordingPipe:
    """Stands in for the multiprocessing pipe, keeping the n_processes running total over time."""

    def __init__(self):
        self.messages = []
        self.running_total = []
        self._total = 0

    def send(self, message):
        self.messages.append(message)
        if "n_processes" in message:
            self._total += message["n_processes"]
            self.running_total.append(self._total)

    @property
    def acquires(self):
        return sum(1 for m in self.messages if m.get("n_processes") == 1)

    @property
    def releases(self):
        return sum(1 for m in self.messages if m.get("n_processes") == -1)

    @property
    def net(self):
        return sum(m.get("n_processes", 0) for m in self.messages)


class _FakeStream:
    """stdout/stderr for a fake FERRE. Optionally goes silent to simulate a hang."""

    def __init__(self, lines, stall_after=None, kill_event=None):
        self._lines = list(lines)
        self._i = 0
        self._stall_after = stall_after
        self._kill_event = kill_event

    def readline(self):
        if self._stall_after is not None and self._i >= self._stall_after:
            # Silent until the monitor thread notices and kills us, exactly as a hung FERRE would be.
            self._kill_event.wait(timeout=30)
            return ""
        if self._i < len(self._lines):
            line = self._lines[self._i]
            self._i += 1
            return line
        return ""

    def read(self):
        return ""

    def close(self):
        pass


class _FakeProcess:
    def __init__(self, lines, stall_after=None):
        self.kill_event = threading.Event()
        self.stdout = _FakeStream(lines, stall_after, self.kill_event)
        self.stderr = _FakeStream([])
        self._returncode = 0

    def kill(self):
        self._returncode = -9
        self.kill_event.set()

    def wait(self):
        return self._returncode


# A run that dispatches and completes objects, then reaches EOF normally.
CLEAN_RUN = (
    [
        " next object #         1\n",
        "           1 0_1000_2000_0_None\n",
        " next object #         2\n",
        "           2 1_1001_2001_0_None\n",
        "",
    ],
    None,
)

# Dispatches and completes one object -- so n_complete > 0 makes a resume eligible -- then goes
# silent, which trips the max_t_communicate watchdog.
HANGING_RUN = (
    [
        " next object #         1\n",
        "           1 0_1000_2000_0_None\n",
        " next object #         2\n",
    ],
    3,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_ferre(process_script, communicate_on_start=True, resumes_available=0, is_list_mode=False):
    """
    Drive `ferre()` against faked subprocesses and return the RecordingPipe.

    process_script: [(stdout_lines, stall_after), ...] -- one per successive FERRE subprocess; the
                    final entry is reused if more are spawned than provided.
    resumes_available: how many times `re_process_partial_ferre` hands back more work before
                    reporting that nothing remains.
    """
    from astra.pipelines import aspcap

    pipe = RecordingPipe()
    counters = {"popen": 0, "reprocess": 0}

    def fake_popen(*args, **kwargs):
        index = min(counters["popen"], len(process_script) - 1)
        counters["popen"] += 1
        lines, stall_after = process_script[index]
        return _FakeProcess(lines, stall_after)

    def fake_reprocess(input_nml_path, pwd=None, exclude_indices=None):
        index = counters["reprocess"]
        counters["reprocess"] += 1
        if index < resumes_available:
            new_path = f"{input_nml_path}.{index + 1}"
            open(new_path, "a").close()
            return (new_path, [f"ignored{i}" for i in range(3)])
        return (None, None)

    with tempfile.TemporaryDirectory() as directory:
        # A real parameter.input, so the timeout-reporting path can actually run.
        with open(os.path.join(directory, "parameter.input"), "w") as fp:
            for i in range(10):
                fp.write(f"{i}_100{i}_200{i}_0_None   0.0 0.0 0.0 0.0 0.0 0.0 0.0 5000.0\n")

        basename = "input_list.nml" if is_list_mode else "input.nml"
        input_nml_path = os.path.join(directory, basename)
        open(input_nml_path, "w").close()

        with mock.patch.object(aspcap.subprocess, "Popen", side_effect=fake_popen), \
             mock.patch.object(aspcap, "re_process_partial_ferre", side_effect=fake_reprocess), \
             mock.patch.object(aspcap, "merge_partial_ferre_outputs", return_value=None), \
             mock.patch.object(aspcap, "debugger", lambda *a, **k: None):
            aspcap.ferre(
                input_nml_path,
                directory,
                n_obj=10,
                n_threads=8,
                pipe=pipe,
                communicate_on_start=communicate_on_start,
                max_t_communicate=2,      # keep the hang cases quick
                max_t_elapsed=None,
                max_sigma_outlier=None,
                max_t_grid_load=None,
                max_resume_attempts=3,
            )

    pipe.subprocess_count = counters["popen"]
    return pipe


def assert_slot_accounting(pipe, communicate_on_start=True):
    """Assert all three invariants for a completed ferre() chain."""
    expected_net = 0 if communicate_on_start else -1
    held_level = 1 if communicate_on_start else 0

    assert pipe.releases == 1, (
        f"expected exactly one n_processes=-1 for the whole chain, got {pipe.releases}; "
        f"running total was {pipe.running_total}"
    )
    assert pipe.net == expected_net, (
        f"expected net n_processes delta {expected_net}, got {pipe.net}"
    )
    # Every reading but the final one must stay at or above the level the chain holds. A dip means
    # the slot was handed back mid-chain and could have been claimed by another job.
    intermediate = pipe.running_total[:-1] if pipe.running_total else []
    dips = [value for value in intermediate if value < held_level]
    assert not dips, (
        f"process slot was released before the chain finished (running total {pipe.running_total} "
        f"dipped below {held_level}); the dispatch loop could hand that slot to another job"
    )


# ---------------------------------------------------------------------------
# Runs that never hang
# ---------------------------------------------------------------------------

def test_clean_run_acquires_and_releases_its_own_slot():
    pipe = run_ferre([CLEAN_RUN], communicate_on_start=True)
    assert_slot_accounting(pipe, communicate_on_start=True)
    assert pipe.acquires == 1


def test_clean_run_releases_slot_acquired_by_caller():
    """With communicate_on_start=False the caller sent the +1, so ferre() only sends the -1."""
    pipe = run_ferre([CLEAN_RUN], communicate_on_start=False)
    assert_slot_accounting(pipe, communicate_on_start=False)
    assert pipe.acquires == 0


# ---------------------------------------------------------------------------
# Hangs, with and without a resume
# ---------------------------------------------------------------------------

def test_hang_with_nothing_to_resume_releases_slot():
    pipe = run_ferre([HANGING_RUN], communicate_on_start=True, resumes_available=0)
    assert_slot_accounting(pipe, communicate_on_start=True)


def test_resume_holds_slot_instead_of_releasing_and_reacquiring():
    """
    The regression test. Before the fix this produced a running total of [1, 0, 1, 0]: the slot was
    handed back when the hung subprocess died and taken again when the resume started, letting the
    dispatch loop give it to a different grid in between.
    """
    pipe = run_ferre([HANGING_RUN, CLEAN_RUN], communicate_on_start=True, resumes_available=1)
    assert_slot_accounting(pipe, communicate_on_start=True)
    assert pipe.subprocess_count == 2, "expected a second FERRE subprocess for the resume"
    assert pipe.acquires == 1, "the resumed sub-run must not acquire a second slot"


def test_nested_resumes_hold_a_single_slot():
    pipe = run_ferre(
        [HANGING_RUN, HANGING_RUN, CLEAN_RUN], communicate_on_start=True, resumes_available=2
    )
    assert_slot_accounting(pipe, communicate_on_start=True)
    assert pipe.subprocess_count == 3
    assert pipe.acquires == 1


def test_exhausted_resume_attempts_release_exactly_one_slot():
    """Giving up after max_resume_attempts must still leave the accounting balanced."""
    pipe = run_ferre([HANGING_RUN], communicate_on_start=True, resumes_available=99)
    assert_slot_accounting(pipe, communicate_on_start=True)
    assert pipe.subprocess_count > 1, "expected at least one resume attempt"
    assert pipe.acquires == 1


def test_resume_with_caller_held_slot():
    pipe = run_ferre([HANGING_RUN, CLEAN_RUN], communicate_on_start=False, resumes_available=1)
    assert_slot_accounting(pipe, communicate_on_start=False)
    assert pipe.acquires == 0


# ---------------------------------------------------------------------------
# Error paths must not leak the slot
# ---------------------------------------------------------------------------

def test_exception_after_acquiring_still_releases_slot():
    """
    An unreleased slot permanently reduces dispatch capacity and deadlocks at max_processes=1. This
    raises while writing the stdout file -- after the slot is taken and after the point the release
    used to be sent, which is precisely the window the fix moved the release past.
    """
    from astra.pipelines import aspcap

    pipe = RecordingPipe()
    real_open = builtins.open

    with tempfile.TemporaryDirectory() as directory:
        input_nml_path = os.path.join(directory, "input.nml")
        real_open(input_nml_path, "w").close()

        def exploding_open(path, *args, **kwargs):
            if str(path).endswith("/stdout"):
                raise OSError("simulated failure writing stdout")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(
                aspcap.subprocess, "Popen",
                side_effect=lambda *a, **k: _FakeProcess(CLEAN_RUN[0])), \
             mock.patch.object(aspcap, "debugger", lambda *a, **k: None), \
             mock.patch.object(builtins, "open", side_effect=exploding_open):
            aspcap.ferre(
                input_nml_path,
                directory,
                n_obj=2,
                n_threads=8,
                pipe=pipe,
                communicate_on_start=True,
                max_t_communicate=None,
                max_t_elapsed=None,
                max_sigma_outlier=None,
                max_t_grid_load=None,
            )

    assert pipe.releases == 1, "the slot must be handed back even when an exception escapes"
    assert pipe.net == 0, f"expected balanced accounting, got net {pipe.net}"


def test_failure_to_acquire_does_not_emit_a_release():
    """
    A slot that was never taken must not be released: that would drive the dispatch loop's counter
    negative and invent capacity, which is the same over-subscription bug in reverse.
    """
    from astra.pipelines import aspcap

    class FailingPipe(RecordingPipe):
        def send(self, message):
            super().send(message)
            if message.get("n_processes") == 1:
                raise RuntimeError("pipe failed during acquire")

    pipe = FailingPipe()
    with mock.patch.object(aspcap, "debugger", lambda *a, **k: None):
        aspcap.ferre(
            "/nonexistent/input.nml", "/tmp", 1, 1, pipe, communicate_on_start=True
        )

    assert pipe.releases == 0, "must not release a slot that was never successfully acquired"
    assert pipe.net >= 0, "accounting must never go negative"
