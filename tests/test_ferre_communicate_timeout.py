"""
Two-phase communication timeout for `astra.pipelines.aspcap.ferre`.

FERRE is legitimately silent in two situations, and neither means it has stopped working:

  * while loading its grid (tens of GB, no output during the read), and
  * between dispatch and the first completed object -- with `NOBJ >= NTHREADS` every thread is
    handed an object at once and they finish in a burst, so nothing is reported until roughly the
    whole first wave lands.

`max_t_communicate` was measuring exactly that window, which made it a per-object *latency* budget
rather than a liveness check. Measured against a real 28 GB grid, first-completion latency was ~917s
at NTHREADS=128 versus a 1000s budget -- and production runs, with colder caches, lost entire grids
to it (128 objects dispatched, zero completed, killed, everything written as NaN).

Once results start flowing the picture inverts: completions and dispatches interleave continuously
and the largest observed gap drops to ~182s, so a long silence there really is a problem.

Hence a two-phase budget: `max_t_communicate_first_result` until a (sub-)execution produces its
first result, `max_t_communicate` afterwards.

The distinction matters most in list (`-l`) mode, used by the abundances stage. Every entry reloads
the grid and has its own silent first wave, but `t_overhead` is only ever set once -- so a gate
based on it, or on a cumulative completion count, protects only the first element. In a real 0.9.4
run this killed all 21 elements after the first, one per execution, each costing a full grid reload.
"""

import os

os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

import tempfile
import threading
from unittest import mock

import pytest


# Verbatim from real FERRE stdout; the banner is printed once per execution, and once per entry in
# list mode. "Done reading!" marks the end of a grid load.
FERRE_BANNER = "                     f e r r e          v4.8.10     \n"
DONE_READING = " Done reading!\n"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class RecordingPipe:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


# How long a stalled fake FERRE stays silent before giving up and reporting EOF. Must sit between
# the "tight" and "generous" budgets used below, so that a watchdog that should fire has time to,
# and one that should not simply reaches the end of the stall.
STALL_SECONDS = 8


class _FakeStream:
    """Emits lines, then optionally goes silent to simulate FERRE working without reporting."""

    def __init__(self, lines, stall_after=None, kill_event=None):
        self._lines = list(lines)
        self._i = 0
        self._stall_after = stall_after
        self._kill_event = kill_event

    def readline(self):
        if self._stall_after is not None and self._i >= self._stall_after:
            # Returns early if the watchdog kills us, otherwise sits silent for the full stall.
            self._kill_event.wait(timeout=STALL_SECONDS)
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
        self.was_killed = False
        self._returncode = 0

    def kill(self):
        self.was_killed = True
        self._returncode = -9
        self.kill_event.set()

    def wait(self):
        return self._returncode


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_ferre(
    lines, stall_after, max_t_communicate, max_t_communicate_first_result,
    max_t_elapsed=None, max_sigma_outlier=None, n_obj=10, max_resume_attempts=0,
):
    """
    Drive ferre() against a faked subprocess; returns (result, elapsed_seconds).

    `result` is a namespace with two independent flags, since the "no communication" and
    per-object "sigma outlier" checks are separate watchdogs that can each fire on their own:
      * result.communicate_fired -- the max_t_communicate[_first_result] check
      * result.outlier_fired     -- the max_t_elapsed / max_sigma_outlier check

    Keyed off the watchdogs' own log lines rather than whether the process object saw a kill():
    ferre() calls kill() unconditionally as cleanup once the reader loop ends, so a clean run and a
    watchdog kill are indistinguishable from the process object alone.

    It also carries what the kill handed to the resume, which is what decides whether hung objects
    are actually dropped from the retry or fed straight back into it:
      * result.exclude_indices  -- passed to re_process_partial_ferre (None if it was never called)
      * result.timeout_pks      -- spectrum pks reported as having caused the timeout
    """
    from time import time as _time
    from types import SimpleNamespace
    from astra.pipelines import aspcap

    process = _FakeProcess(lines, stall_after)
    logged = []
    pipe = RecordingPipe()
    resume_calls = []

    def _fake_re_process(existing_input_nml_path, pwd=None, exclude_indices=None):
        resume_calls.append(list(exclude_indices or []))
        return (None, None)

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "parameter.input"), "w") as fp:
            for i in range(n_obj):
                fp.write(f"{i}_100{i}_200{i}_0_None   0.0 0.0 0.0 0.0 0.0 0.0 0.0 5000.0\n")
        input_nml_path = os.path.join(directory, "input.nml")
        # A real control file always names its PFILE, and the timeout-reporting path reads it to map
        # object indices back to spectrum pks -- without it that path raises and reports nothing.
        with open(input_nml_path, "w") as fp:
            fp.write(" PFILE = 'parameter.input'\n")

        started = _time()
        with mock.patch.object(aspcap.subprocess, "Popen", return_value=process), \
             mock.patch.object(aspcap, "re_process_partial_ferre", side_effect=_fake_re_process), \
             mock.patch.object(aspcap, "merge_partial_ferre_outputs", return_value=None), \
             mock.patch.object(aspcap, "debugger",
                               lambda *a, **k: logged.append(" ".join(map(str, a)))):
            aspcap.ferre(
                input_nml_path,
                directory,
                n_obj=n_obj,
                n_threads=8,
                pipe=pipe,
                communicate_on_start=True,
                max_t_communicate=max_t_communicate,
                max_t_communicate_first_result=max_t_communicate_first_result,
                max_t_elapsed=max_t_elapsed,
                max_sigma_outlier=max_sigma_outlier,
                max_t_grid_load=None,
                max_resume_attempts=max_resume_attempts,
            )

    assert not any("MONITOR DIED" in m for m in logged), "the monitor thread raised"
    result = SimpleNamespace(
        communicate_fired=any("hanging no communication" in m for m in logged),
        # The sigma-outlier kill logs "hanging [<indices>]" -- distinct from the
        # "hanging on <path>" summary line emitted afterwards regardless of cause.
        outlier_fired=any(m.startswith("hanging [") for m in logged),
        exclude_indices=resume_calls[0] if resume_calls else None,
        timeout_pks=[m["timeout_on_spectrum_pk"] for m in pipe.messages
                     if "timeout_on_spectrum_pk" in m],
    )
    return result, _time() - started


# ---------------------------------------------------------------------------
# The marker regex
# ---------------------------------------------------------------------------

def test_execution_start_regex_matches_banner_and_load_end():
    from astra.pipelines.aspcap import REGEX_EXECUTION_START
    assert REGEX_EXECUTION_START.search(FERRE_BANNER)
    assert REGEX_EXECUTION_START.search(DONE_READING)


def test_execution_start_regex_ignores_dispatch_and_completion_lines():
    """Must not fire on ordinary progress, or the first-result budget would never expire."""
    from astra.pipelines.aspcap import REGEX_EXECUTION_START
    assert not REGEX_EXECUTION_START.search(" next object #         1\n")
    assert not REGEX_EXECUTION_START.search("           1 0_1000_2000_0_None\n")
    assert not REGEX_EXECUTION_START.search(" median snr =   119.737000\n")


# ---------------------------------------------------------------------------
# Before the first result: the generous budget applies
# ---------------------------------------------------------------------------

def test_silence_before_first_result_is_not_killed_at_the_steady_state_budget():
    """
    The regression test. Startup, grid load, dispatch, then silence -- the shape that lost whole
    grids in production. With a steady-state budget of 2s and a first-result budget of 60s, the
    process must survive: it has not reported a result yet, so this silence is expected.
    """
    lines = [
        FERRE_BANNER,
        DONE_READING,
        " next object #         1\n",
        " next object #         2\n",
    ]
    result, elapsed = run_ferre(
        lines, stall_after=4, max_t_communicate=2, max_t_communicate_first_result=60
    )
    assert not result.communicate_fired, (
        "FERRE was killed during the pre-first-result window; that silence is expected and should "
        "be governed by max_t_communicate_first_result, not max_t_communicate"
    )
    # Sanity: it really did sit silent well past the steady-state budget rather than exiting early.
    assert elapsed > 4, f"expected a real stall of >4s, only took {elapsed:.1f}s"


def test_silence_before_first_result_is_killed_once_its_own_budget_expires():
    """The generous budget is a longer leash, not an absence of one."""
    lines = [
        FERRE_BANNER,
        DONE_READING,
        " next object #         1\n",
    ]
    result, _ = run_ferre(
        lines, stall_after=3, max_t_communicate=60, max_t_communicate_first_result=2
    )
    assert result.communicate_fired, "a stall exceeding max_t_communicate_first_result must still be caught"


# ---------------------------------------------------------------------------
# After the first result: the tight budget applies
# ---------------------------------------------------------------------------

def test_silence_after_first_result_is_killed_at_the_steady_state_budget():
    """
    Once a result has been reported, completions and dispatches interleave continuously, so a long
    gap is genuinely suspicious and must be caught promptly -- not held to the generous budget.
    """
    lines = [
        FERRE_BANNER,
        DONE_READING,
        " next object #         1\n",
        "           1 0_1000_2000_0_None\n",   # first result -> tighten the budget
        " next object #         2\n",
    ]
    result, _ = run_ferre(
        lines, stall_after=5, max_t_communicate=2, max_t_communicate_first_result=60
    )
    assert result.communicate_fired, (
        "after a result was reported the steady-state budget applies; this stall should have been "
        "caught at max_t_communicate rather than waiting for max_t_communicate_first_result"
    )


# ---------------------------------------------------------------------------
# List mode: every entry gets its own first-result window
# ---------------------------------------------------------------------------

def test_new_sub_execution_restores_the_first_result_budget():
    """
    In list (-l) mode each entry reloads the grid and has its own silent first wave. After entry one
    completes, a naive "have we ever completed anything?" gate would leave entry two exposed to the
    tight budget during its reload -- which is exactly what killed all 21 elements after the first
    in a real 0.9.4 abundances run, one per execution.
    """
    lines = [
        FERRE_BANNER,
        DONE_READING,
        " next object #         1\n",
        "           1 0_1000_2000_0_None\n",   # entry one produces a result
        FERRE_BANNER,                          # entry two begins: budget must reset
        DONE_READING,
        " next object #         1\n",
    ]
    result, elapsed = run_ferre(
        lines, stall_after=7, max_t_communicate=2, max_t_communicate_first_result=60
    )
    assert not result.communicate_fired, (
        "a new sub-execution must restore the first-result budget; otherwise every list-mode entry "
        "after the first is killed during its own grid load"
    )
    assert elapsed > 4, f"expected a real stall of >4s, only took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# The per-object outlier watchdog (max_t_elapsed / max_sigma_outlier) has the same flaw
# ---------------------------------------------------------------------------
#
# This is a second, independent watchdog from max_t_communicate: it flags an individual object as
# hung if its wait time exceeds max_t_elapsed, or is an outlier by more than max_sigma_outlier
# standard deviations from the median of *completed* objects. Before any object has completed,
# there is no such median to compare against -- the code falls back to a hardcoded
# median=120.0, stddev=10.0 -- so it is not measuring anything real. In a production run this fired
# repeatedly on a grid whose true first-completion latency was on the order of max_t_elapsed
# itself, flagging ~128 objects as "hanging" and exhausting all 5 resume attempts (each one
# reproducing the same failure on a shrunken batch) while never completing a single object.
#
# The fix mirrors the communication budget: skip this test entirely while awaiting_first_result is
# true, and rely on max_t_communicate_first_result as the sole guard for that window.

def test_outlier_watchdog_does_not_fire_before_first_result():
    """
    The regression test for the second watchdog. Two objects dispatched, neither ever completes.
    max_t_elapsed=1 is tiny enough that the old code kills this almost immediately; with the fix,
    awaiting_first_result stays True (nothing has completed), so this check must not run at all.
    """
    lines = [
        FERRE_BANNER,
        DONE_READING,
        " next object #         1\n",
        " next object #         2\n",
    ]
    result, elapsed = run_ferre(
        lines, stall_after=4,
        max_t_communicate=60, max_t_communicate_first_result=60,
        max_t_elapsed=1, max_sigma_outlier=None,
    )
    assert not result.outlier_fired, (
        "an object waiting past max_t_elapsed before any result was killed; with zero completions "
        "the median/stddev it is compared against is a hardcoded fallback, not measured data, so "
        "this window must be governed by max_t_communicate_first_result instead"
    )
    assert elapsed > 4, f"expected a real stall of >4s, only took {elapsed:.1f}s"


def test_outlier_watchdog_still_fires_once_real_completion_data_exists():
    """
    A real outlier must still be caught promptly once genuine completion-time data exists.

    This covers the `max_sigma_outlier=None` route, where the absolute `max_t_elapsed` cap is the
    only condition. Note the sample size: an earlier version of this test used a single completion
    and asserted the watchdog fired, which encoded the very bug that cost GKd its abundances --
    one sample gives stddev 0 and cannot support an outlier judgement at all.

    Six objects are left outstanding rather than two, so that a real share of the thread pool is
    wedged. With fewer than half the pool waiting this is a drained tail, not a stall, and is
    deliberately left to max_t_communicate -- see the test below.
    """
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 17)]
    lines += [f"          {i} {i-1}_100{i-1}_200{i-1}_0_None\n" for i in range(1, 11)]

    result, _ = run_ferre(
        lines, stall_after=len(lines),
        max_t_communicate=60, max_t_communicate_first_result=60,
        max_t_elapsed=1, max_sigma_outlier=None,
        n_obj=16,
    )
    assert result.outlier_fired, (
        "with a real completed sample, an object exceeding max_t_elapsed should still be flagged "
        "-- this watchdog must be suppressed only while the sample is degenerate, not always"
    )


def test_outlier_watchdog_leaves_a_drained_tail_to_max_t_communicate():
    """
    A tail of a few slow objects must not be killed by the outlier test.

    The outstanding set shrinks as an execution drains, so once the work runs out "every outstanding
    object looks slow" is trivially true of the last one or two -- and this test cannot tell a
    genuinely stuck object from a merely slow one. Killing there costs a grid reload and abandons
    stars that were about to finish. Sustained silence is much stronger evidence, so the tail is
    left to max_t_communicate, which reaches the same end state (all outstanding objects excluded)
    on better grounds.

    Same shape as the test above, but only two of eight threads are still occupied.
    """
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 13)]
    lines += [f"          {i} {i-1}_100{i-1}_200{i-1}_0_None\n" for i in range(1, 11)]

    result, _ = run_ferre(
        lines, stall_after=len(lines),
        max_t_communicate=5, max_t_communicate_first_result=5,
        max_t_elapsed=1, max_sigma_outlier=None,
        n_obj=12,
    )
    assert not result.outlier_fired, (
        "a two-object tail was killed by the outlier test; with most of the pool idle this is a "
        "drained execution rather than a stall, and belongs to max_t_communicate"
    )
    assert result.communicate_fired, (
        "the tail must still be cleared by the communication budget, otherwise a genuinely stuck "
        "tail would hang forever"
    )


# ---------------------------------------------------------------------------
# Small-sample degeneracy in the sigma-outlier test
#
# `awaiting_first_result` covers the zero-completion case. These cover the case that actually
# destroyed data in the 2026-08-31 run: a *handful* of completions, which is just as degenerate.
#
# GKd params dispatched 707 objects. Exactly one completed (at 1079.9s -- these fits are genuinely
# that slow). np.std of a single sample is 0, which the old code replaced with an absolute 10.0.
# The sigma test then read "10 sigma" as median + 100s = ~1180s, and since the whole wave had been
# dispatched together and all of it was passing 1180s at once, 126 perfectly healthy objects were
# flagged in a single event and permanently abandoned. That happened 43 times across the run --
# 127 objects each -- which is what left GKd's abundances at ~50% NaN on every element.
#
# The times below are scaled down to keep the suite fast; the mechanism (one completion arming a
# statistical test) is identical.
# ---------------------------------------------------------------------------

def test_a_single_completion_does_not_arm_the_outlier_test():
    """
    One completion, then a stall with objects still waiting. The old code armed the sigma test off
    that single sample and flagged the whole waiting set; it must now stay quiet.
    """
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 13)]
    lines += ["           1 0_1000_2000_0_None\n"]          # exactly one result

    result, _ = run_ferre(
        lines,
        stall_after=len(lines),
        max_t_communicate=60,
        max_t_communicate_first_result=60,
        max_t_elapsed=1,
        max_sigma_outlier=0.1,   # scaled so a short stall would trip a degenerate stddev
        n_obj=12,
    )
    assert not result.outlier_fired, (
        "a single completion is not a distribution: stddev collapses to 0 and the old absolute "
        "floor made every still-waiting object look like a large outlier"
    )
    assert not result.communicate_fired, "the communicate budget was generous; it must not fire"


def test_enough_completions_still_catch_a_real_straggler():
    """The guard must not disable the watchdog: with a real sample, a genuine outlier still dies."""
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 15)]
    # MIN_OUTLIER_SAMPLES completions, so median/stddev describe something real.
    lines += [f"          {i} {i-1}_100{i-1}_200{i-1}_0_None\n" for i in range(1, 11)]

    result, _ = run_ferre(
        lines,
        stall_after=len(lines),
        max_t_communicate=60,
        max_t_communicate_first_result=60,
        max_t_elapsed=1,
        max_sigma_outlier=0.1,
        n_obj=14,
    )
    assert result.outlier_fired, (
        "with a real completed sample the outlier test must still fire on objects left waiting"
    )


def test_minimum_sample_threshold_is_above_one():
    """A sample of one can never be a distribution -- pin that the constant reflects it."""
    from astra.pipelines.aspcap import MIN_OUTLIER_SAMPLES
    assert MIN_OUTLIER_SAMPLES > 1


def test_stddev_floor_scales_with_the_median_not_an_absolute_10s():
    """
    The floor must be relative. GKd params completions clustered near 1080s; an absolute 10s floor
    put "10 sigma" only 100s above the median, inside the natural spread of a synchronized wave.
    Measured healthy batches show a spread of 20-35% of the median, so the floor belongs there.
    """
    from astra.pipelines.aspcap import MIN_RELATIVE_STDDEV

    median = 1080.0          # observed GKd params completion time
    max_sigma_outlier = 10   # the production setting

    old_threshold = median + max_sigma_outlier * 10.0        # absolute floor
    new_threshold = median + max_sigma_outlier * max(0.0, MIN_RELATIVE_STDDEV * median)

    assert old_threshold < 1200, "the old floor sat inside the wave's own spread"
    assert new_threshold > 3 * median, "the floor must scale with the timescale being measured"


def test_unmatched_completions_do_not_disable_the_outlier_test():
    """
    A completion whose `next object #` entry cannot be matched records NaN as its elapsed time.
    Nothing ever removes those, so if they were allowed into the median/stddev the whole sample
    would evaluate to NaN and this watchdog would be off for the rest of the execution. The real
    2026-08-31 run logged 6439 such `nan nan` evaluations, so this is the common case, not an edge.

    Here the completion names deliberately do not correspond to dispatched indices, which is what
    drives the NaN path, alongside enough well-formed completions to arm the test.
    """
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 15)]
    # Unmatchable completions -> NaN elapsed times mixed in with real ones.
    lines += [f"          {i} 99{i}_9000_9000_0_None\n" for i in range(1, 4)]
    lines += [f"          {i} {i-1}_100{i-1}_200{i-1}_0_None\n" for i in range(1, 11)]

    result, _ = run_ferre(
        lines, stall_after=len(lines),
        max_t_communicate=60, max_t_communicate_first_result=60,
        max_t_elapsed=1, max_sigma_outlier=None,
        n_obj=14,
    )
    assert result.outlier_fired, (
        "NaN elapsed times must be ignored, not allowed to poison the distribution and silently "
        "switch the watchdog off"
    )


# ---------------------------------------------------------------------------
# Naming culprits on the no-communication kill
#
# The sigma test now only fires once *every* outstanding object is hung -- i.e. when no thread is
# progressing -- so that one grid reload sheds up to NTHREADS bad objects instead of one. That
# makes the two watchdogs race on the same clock: the last thread to wedge starts both the
# communication budget and its own max_t_elapsed clock at the same moment, and the communication
# check runs first in the loop body. So in exactly the many-bad-objects case, the communication
# path wins.
#
# It used to kill blind. re_process_partial_ferre rebuilds the retry from the names already written
# to the output files, which never include the hung objects -- so with an empty exclusion list they
# went straight back into the resumed batch and hung again, burning every resume attempt.
# ---------------------------------------------------------------------------

def test_no_communication_kill_names_the_outstanding_objects():
    """A wedge caught by the communication budget must hand its in-flight objects to the resume."""
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 13)]
    lines += [f"          {i} {i-1}_100{i-1}_200{i-1}_0_None\n" for i in range(1, 11)]

    result, _ = run_ferre(
        lines, stall_after=len(lines),
        # Tight enough to fire on the stall; the sigma test is off so this is the only watchdog.
        max_t_communicate=1, max_t_communicate_first_result=1,
        max_t_elapsed=None, max_sigma_outlier=None,
        n_obj=12, max_resume_attempts=1,
    )
    assert result.communicate_fired, "expected the communication budget to fire on the stall"
    assert result.exclude_indices is not None, "the kill should have prepared a resume"
    # Objects 11 and 12 were dispatched and never completed; exclude_indices is 0-indexed.
    assert sorted(result.exclude_indices) == [10, 11], (
        f"expected the two outstanding objects to be excluded, got {result.exclude_indices} -- "
        "an empty list feeds the hung objects straight back into the resumed batch"
    )


def test_no_communication_kill_before_any_result_still_names_them():
    """
    The early window is governed solely by max_t_communicate_first_result, since the sigma test is
    suppressed until it has real completion data. A wedge there must still name culprits, otherwise
    a batch that hangs on its first wave has nothing to exclude and cannot usefully resume at all.
    """
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 4)]

    result, _ = run_ferre(
        lines, stall_after=len(lines),
        max_t_communicate=1, max_t_communicate_first_result=1,
        max_t_elapsed=1, max_sigma_outlier=None,
        n_obj=6, max_resume_attempts=1,
    )
    assert result.communicate_fired, "expected the first-result budget to fire on the stall"
    assert not result.outlier_fired, "the sigma test must stay suppressed before any completion"
    assert sorted(result.exclude_indices or []) == [0, 1, 2], (
        f"expected all three in-flight objects to be excluded, got {result.exclude_indices}"
    )


def test_no_communication_kill_reports_the_culprits_as_causing_the_timeout():
    """
    The excluded objects must also be reported back, so their rows carry flag_caused_timeout.

    This one passes with or without the exclusion above -- the reporting path falls back to every
    outstanding object when nothing was excluded. What it guards is the *other* branch, which the
    exclusion now makes live: keys_to_flag re-derives 1-indexed keys from 0-indexed
    exclude_indices, so an off-by-one there would silently report nothing at all.
    """
    lines = [FERRE_BANNER, DONE_READING]
    lines += [f" next object #        {i}\n" for i in range(1, 13)]
    lines += [f"          {i} {i-1}_100{i-1}_200{i-1}_0_None\n" for i in range(1, 11)]

    result, _ = run_ferre(
        lines, stall_after=len(lines),
        max_t_communicate=1, max_t_communicate_first_result=1,
        max_t_elapsed=None, max_sigma_outlier=None,
        n_obj=12, max_resume_attempts=1,
    )
    # parameter.input names are "<index>_100<index>_200<index>_0_None", so spectrum_pk is 200<index>
    # for the two objects (10, 11) left outstanding.
    assert sorted(result.timeout_pks) == [20010, 20011], (
        f"expected the outstanding objects reported as causing the timeout, got {result.timeout_pks}"
    )
