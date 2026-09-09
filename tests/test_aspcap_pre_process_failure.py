"""
Regression tests for how `_aspcap_stage` handles a failure inside `pre_process_ferre`.

Background
----------
`_safe_pre_process_ferre` used to catch any exception, log it, and fall through -- which
means it returned `None`. Its caller in `_aspcap_stage` unconditionally unpacked that
result into a 5-tuple:

    input_nml_path, pwd, n_obj, n_ferre_threads, skipped = future.result()

so a failure became `TypeError: cannot unpack non-iterable NoneType object`, raised two
layers away from the real cause. Astra's `@task` wrapper then caught *that*, logged it,
and let the CLI exit 0 -- so a full output filesystem produced a run that SLURM recorded
as COMPLETED / ExitCode 0:0 while writing no results at all.

The fix returns the exception instead of `None`, and has the caller collect those and
raise at the end of the stage, preserving the original exception as the cause.
"""
import pytest

import astra.pipelines.aspcap as aspcap_module


ENOSPC = 28


@pytest.fixture
def restore_pre_process():
    """Swap `pre_process_ferre` for the duration of a test, then put it back."""
    original = aspcap_module.pre_process_ferre
    yield
    aspcap_module.pre_process_ferre = original


def _plan(pwd="/fake/run/params/apo25m_5_GKg"):
    return [{"pwd": pwd, "spectra": []}]


def test_returns_exception_not_none_on_failure(restore_pre_process):
    """The wrapper must hand the real exception back, never None."""

    def raise_enospc(*args, **kwargs):
        raise OSError(ENOSPC, "No space left on device", "/fake/run/params")

    aspcap_module.pre_process_ferre = raise_enospc

    result = aspcap_module._safe_pre_process_ferre(_plan())

    assert result is not None, (
        "returning None is the original bug: the caller unpacks this into a 5-tuple, "
        "so None becomes a TypeError that masks the real error"
    )
    assert isinstance(result, OSError)
    assert result.errno == ENOSPC


def test_original_cause_survives_for_the_caller(restore_pre_process):
    """The returned exception keeps enough detail to diagnose a disk-full run."""

    def raise_enospc(*args, **kwargs):
        raise OSError(ENOSPC, "No space left on device", "/fake/run/params")

    aspcap_module.pre_process_ferre = raise_enospc

    result = aspcap_module._safe_pre_process_ferre(_plan())

    # This is what the RuntimeError raised by `_aspcap_stage` interpolates, and what a
    # user actually needs to see to know the filesystem filled up.
    assert "No space left on device" in str(result)


def test_success_path_is_untouched(restore_pre_process):
    """A working `pre_process_ferre` must still pass its 5-tuple straight through."""
    expected = ("/fake/input.nml", "/fake/pwd", 128, 64, [])

    def succeed(*args, **kwargs):
        return expected

    aspcap_module.pre_process_ferre = succeed

    result = aspcap_module._safe_pre_process_ferre(_plan())

    assert result == expected
    assert not isinstance(result, Exception)
    # The caller unpacks exactly these five names; if that ever changes, this breaks loudly.
    input_nml_path, pwd, n_obj, n_ferre_threads, skipped = result
    assert (input_nml_path, pwd, n_obj, n_ferre_threads, skipped) == expected


def test_failure_is_distinguishable_from_success(restore_pre_process):
    """
    The caller branches on `isinstance(result, Exception)`. An exception must never be
    mistakable for a valid 5-tuple, and vice versa.
    """

    def raise_enospc(*args, **kwargs):
        raise OSError(ENOSPC, "No space left on device")

    aspcap_module.pre_process_ferre = raise_enospc
    failure = aspcap_module._safe_pre_process_ferre(_plan())

    aspcap_module.pre_process_ferre = lambda *a, **k: ("nml", "pwd", 1, 1, [])
    success = aspcap_module._safe_pre_process_ferre(_plan())

    assert isinstance(failure, Exception)
    assert not isinstance(success, Exception)
