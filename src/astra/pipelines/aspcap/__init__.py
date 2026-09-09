import os
import numpy as np
import concurrent.futures
import subprocess
import re
import json
import fcntl
import pickle
import traceback
from multiprocessing import Pipe, Lock
from datetime import datetime
from tempfile import mkdtemp
import threading
from typing import Optional, Iterable, List, Tuple, Callable, Union, Sequence
from tqdm import tqdm
from time import time, sleep

from astra import __version__, task
from astra.utils import log, expand_path
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
from astra.models.aspcap import ASPCAP, FerreCoarse, FerreStellarParameters, FerreChemicalAbundances, Source
from astra.models.spectrum import Spectrum
from astra.pipelines.ferre.processing import pre_process_ferre, post_process_ferre, re_process_partial_ferre, merge_partial_ferre_outputs
from astra.pipelines.ferre.utils import parse_header_path, parse_ferre_spectrum_name, read_control_file
from astra.pipelines.aspcap.initial import get_initial_guesses, get_initial_arjl_guesses
from astra.pipelines.aspcap.coarse import plan_coarse_stellar_parameters_stage
from astra.pipelines.aspcap.stellar_parameters import plan_stellar_parameters_stage
from astra.pipelines.aspcap.abundances import plan_abundances_stage, get_species
#from astra.pipelines.aspcap.stellar_parameters import stellar_parameters, post_stellar_parameters
from astra.pipelines.aspcap.utils import ABUNDANCE_RELATIVE_TO_H

from astra.pipelines.aspcap.debugger import debugger, HOSTNAME, RAND

#cd /scratch/general/nfs1/u6020307/pbs/aspcap-2025-02-12-ax1pzxtc

#subprocess.check_output("echo `hostname`"))

def _is_list_mode(path):
    return "input_list.nml" in path

def _safe_pre_process_ferre(*args, **kwargs):
    try:
        return pre_process_ferre(*args, **kwargs)
    except Exception as e:
        # Return the exception rather than falling through to an implicit None. The caller
        # unpacks this result into a 5-tuple, so a None turned any failure here into
        # `TypeError: cannot unpack non-iterable NoneType object` two layers away, which the
        # @task wrapper then logged and swallowed -- so a full disk produced a run that
        # reported success while writing nothing.
        try:
            pwd = args[0][0]["pwd"]
        except Exception:
            pwd = "(unknown pwd)"
        # Log the pwd and traceback, not the whole plan: a plan holds thousands of spectra
        # and dumping it buries the actual error in tens of kilobytes of repr.
        debugger(
            f"Exception in pre_process_ferre for {pwd}: "
            f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}"
        )
        return e


def _safe_post_process_ferre(*args, **kwargs):
    try:
        return post_process_ferre(*args, **kwargs)
    except Exception as e:
        debugger(f"Exception in post_process_ferre {args} {kwargs}: {e}")
        return []

def _safe_ferre(*args, **kwargs):
    try:
        return ferre(*args, **kwargs)
    except Exception as e:
        debugger(f"Exception in ferre {args} {kwargs}: {e}")



@task
def aspcap(
    spectra: Iterable[ApogeeCoaddedSpectrumInApStar],
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    #header_paths: Optional[Union[Sequence[str], str]] = "/uufs/chpc.utah.edu/common/home/u6020307/vast/aspcap-grids/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    parent_dir: Optional[str] = None,
    n_threads: Optional[int] = 42, # 32 in normal mode
    max_processes: Optional[int] = 3, # 16 previously, 4 in normal mode,
    max_threads: Optional[int] = 128,
    max_concurrent_loading: Optional[int] = 4,
    soft_thread_ratio: Optional[float] = 1,
    use_ferre_list_mode: Optional[bool] = True,
    live_renderable: Optional[object] = None,
    **kwargs
) -> Iterable[ASPCAP]:
    """
    Run the ASPCAP pipeline on some spectra.

    :param spectra:
        The spectra to analyze with ASPCAP.

    :param initial_guess_callable: [optional]
        A callable that returns an initial guess for the stellar parameters.

    :param header_paths: [optional]
        The path to a file containing the paths to the FERRE header files. This file should contain one path per line.

    :param weight_path: [optional]
        The path to the FERRE weight file to use during the coarse and main stellar parameter stage.

    :param element_weight_paths: [optional]
        A path containing FERRE weight files for different elements, which will be used in the chemical abundances stage.

    :param parent_dir: [optional]
        The parent directory where these FERRE executions will be planned. If `None` is given then this will default
        to a temporary directory in `$MWM_ASTRA/X.Y.Z/pipelines/aspcap/`.

    :param n_threads: [optional]
        The number of threads to use per FERRE process.

    :param max_processes: [optional]
        The maximum number of FERRE processes to run at once.

    :param max_threads: [optional]
        The maximum number of threads to run at once. This is a soft limit that can be temporarily exceeded by `soft_thread_ratio`
        to allow new FERRE processes to load into memory while existing threads are still running.

    :param max_concurrent_loading: [optional]
        The maximum number of FERRE grids to load at once. This is to prevent disk I/O from becoming a bottleneck.

    :param soft_thread_ratio: [optional]
        The ratio of threads to processes that can be temporarily exceeded to allow new FERRE processes to load into memory while
        existing threads are still running.

    :param use_ferre_list_mode: [optional]
        Use the `-l` list mode in FERRE for the abundances stage. In theory this is more efficient. In practice FERRE can hang
        forever in list mode when it does not hang in normal mode.

    :param live_renderable: [optional]
        A live renderable object that can be updated with progress information. This is useful for Jupyter notebooks or other
        live-rendering environments.

    Keyword arguments
    -----------------
    All additional keyword arguments will be passed through to `astra.pipelines.ferre.pre_process.pre_process.ferre`.
    Some handy keywords include:
    continuum_order: int = 4,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    """

    if spectra[0]._meta.name.startswith("arjl"):
        # header_paths = "$MWM_ASTRA/pipelines/aspcap/arjl_header_paths.list"  # did not have permission! CHNAGE LATER!
        header_paths = "$MWM_ASTRA/ARjl_grids/arjl_header_paths.list"
        if initial_guess_callable is None:
            initial_guess_callable = get_initial_arjl_guesses
        kwargs.update(ferre_kwds=dict(continuum_order=2))
    else:
        if initial_guess_callable is None:
            initial_guess_callable = get_initial_guesses

    if parent_dir is None:
        _dir = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/aspcap/")
        os.makedirs(_dir, exist_ok=True)
        parent_dir = mkdtemp(prefix=f"{datetime.now().strftime('%Y-%m-%d')}-", dir=_dir)
        os.chmod(parent_dir, 0o755)

    from time import sleep
    parent, child = Pipe()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(max_threads, max_processes)) as executor:
        stage_args = [executor, parent, child, parent_dir, max_processes, max_threads, max_concurrent_loading, soft_thread_ratio]
        if isinstance(live_renderable, str):
            class FakeProgress:
                def __init__(self, path):
                    self.path = path
                    self.task_counter = 0
                    if not os.path.exists(path):
                        with open(path, "w"):
                            pass
                    return None

                def append(self, data):
                    try:
                        r = json.dumps(data) + "\n"
                        with open(self.path, "a") as fp:
                            fp.write(r)
                        return True
                    except Exception as e:
                        debugger(f"Exception appending to file: {e}")
                        return False

                def update(self, *args, **kwargs):
                    return self.append(("update", args, kwargs))

                def add_task(self, *args, **kwargs):
                    self.task_counter += 1
                    self.append(("add_task", self.task_counter, args, kwargs))
                    return self.task_counter

            progress = FakeProgress(live_renderable)
            stage_args += [progress]

        elif live_renderable is not None:
            from rich.panel import Panel
            from rich.progress import (Progress, BarColumn, MofNCompleteColumn, TimeElapsedColumn)
            progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn()
            )
            live_renderable.add_row(Panel.fit(progress, title="ASPCAP", padding=(2, 2)))
            stage_args += [progress]

        ferre_kwds = kwargs.pop("ferre_kwds", {})
        coarse_plans, spectra_with_no_initial_guess = plan_coarse_stellar_parameters_stage(
            spectra=spectra,
            parent_dir=parent_dir,
            header_paths=header_paths,
            initial_guess_callable=initial_guess_callable,
            weight_path=weight_path,
            n_threads=n_threads,
            **ferre_kwds
        )
        for spectrum in spectra_with_no_initial_guess:
            yield ASPCAP.from_spectrum(spectrum, flag_no_suitable_initial_guess=True)


        coarse_results, coarse_failures = _aspcap_stage("coarse", coarse_plans, *stage_args)
        debugger(f"aspcap: coarse stage returned: {len(coarse_results)} results, {len(coarse_failures)} failures")
        debugger(f"aspcap: about to yield from coarse_failures (n={len(coarse_failures)})")
        yield from coarse_failures
        debugger(f"aspcap: done yielding coarse_failures; calling plan_stellar_parameters_stage")

        stellar_parameter_plans, best_coarse_results = plan_stellar_parameters_stage(
            spectra=spectra,
            parent_dir=parent_dir,
            coarse_results=coarse_results,
            weight_path=weight_path,
            n_threads=n_threads,
            **ferre_kwds
        )
        debugger(f"aspcap: plan_stellar_parameters_stage returned: {len(stellar_parameter_plans)} plans, {len(best_coarse_results)} best coarse results")
        param_results, param_failures = _aspcap_stage("params", stellar_parameter_plans, *stage_args)
        yield from param_failures

        abundance_plans = plan_abundances_stage(
            spectra=spectra,
            parent_dir=parent_dir,
            stellar_parameter_results=param_results,
            element_weight_paths=element_weight_paths,
            n_threads=n_threads,
            use_ferre_list_mode=use_ferre_list_mode,
            **ferre_kwds
        )

        abundance_results, abundance_failures = _aspcap_stage(
            "abundances",
            abundance_plans,
            *stage_args,
            use_ferre_list_mode=use_ferre_list_mode,
            # Abundances is much chattier than the parameter stages (measured completion gaps:
            # 0.06-0.55s median, 57.6s worst), and this budget is now also what clears a stuck tail.
            ferre_kwds=dict(max_sigma_outlier=15, max_t_elapsed=300, max_t_communicate=300)
        )

        # Bring it all together baby.
        result_kwds = {}
        for r in param_results:
            coarse = best_coarse_results[r["spectrum_pk"]]
            v_sini = 10**(r.get("log10_v_sini", np.nan))
            e_v_sini = r.get("e_log10_v_sini", np.nan) * v_sini * np.log(10)
            v_micro = 10**(r.get("log10_v_micro", np.nan))
            e_v_micro = r.get("e_log10_v_micro", np.nan) * v_micro * np.log(10)
            r.update(
                raw_teff=r["teff"],
                raw_e_teff=r["e_teff"],
                raw_logg=r["logg"],
                raw_e_logg=r["e_logg"],
                raw_v_micro=v_micro,
                raw_e_v_micro=e_v_micro,
                raw_v_sini=v_sini,
                raw_e_v_sini=e_v_sini,
                raw_m_h_atm=r["m_h"],
                raw_e_m_h_atm=r["e_m_h"],
                raw_alpha_m_atm=r.get("alpha_m", np.nan),
                raw_e_alpha_m_atm=r.get("e_alpha_m", np.nan),
                raw_c_m_atm=r.get("c_m", np.nan),
                raw_e_c_m_atm=r.get("e_c_m", np.nan),
                raw_n_m_atm=r.get("n_m", np.nan),
                raw_e_n_m_atm=r.get("e_n_m", np.nan),
                m_h_atm=r["m_h"],
                e_m_h_atm=r["e_m_h"],
                alpha_m_atm=r.get("alpha_m", np.nan),
                e_alpha_m_atm=r.get("e_alpha_m", np.nan),
                c_m_atm=r.get("c_m", np.nan),
                e_c_m_atm=r.get("e_c_m", np.nan),
                n_m_atm=r.get("n_m", np.nan),
                e_n_m_atm=r.get("e_n_m", np.nan),
                v_sini=v_sini,
                e_v_sini=e_v_sini,
                v_micro=v_micro,
                e_v_micro=e_v_micro,
                coarse_teff=coarse.teff,
                coarse_logg=coarse.logg,
                coarse_v_micro=10**(coarse.log10_v_micro or np.nan),
                coarse_v_sini=10**(coarse.log10_v_sini or np.nan),
                coarse_m_h_atm=coarse.m_h,
                coarse_alpha_m_atm=coarse.alpha_m,
                coarse_c_m_atm=coarse.c_m,
                coarse_n_m_atm=coarse.n_m,
                coarse_rchi2=coarse.rchi2,
                coarse_penalized_rchi2=coarse.penalized_rchi2,
                coarse_ferre_flags=coarse.ferre_flags,
                coarse_short_grid_name=coarse.short_grid_name,
                initial_teff=coarse.initial_teff,
                initial_logg=coarse.initial_logg,
                initial_v_micro=10**(coarse.initial_log10_v_micro or np.nan),
                initial_v_sini=10**(coarse.initial_log10_v_sini or np.nan),
                initial_m_h_atm=coarse.initial_m_h,
                initial_alpha_m_atm=coarse.initial_alpha_m,
                initial_c_m_atm=coarse.initial_c_m,
                initial_n_m_atm=coarse.initial_n_m,
                ferre_time_coarse=coarse.t_elapsed,
                ferre_time_params=r["t_elapsed"],
                pwd=parent_dir,
            )
            result_kwds[r["spectrum_pk"]] = r

        for r in abundance_results:
            species = get_species(r["weight_path"])
            label = species.lower() if species.lower() == "c_12_13" else f"{species.lower()}_h"

            for key in ("m_h", "alpha_m", "c_m", "n_m"):
                if not r.get(f"flag_{key}_frozen", False):
                    break
            else:
                raise ValueError(f"Can't figure out which label to use")

            value, e_value = (r[key], r[f"e_{key}"])

            if not ABUNDANCE_RELATIVE_TO_H[species] and value is not None:
                # [X/M] = [X/H] - [M/H]
                # [X/H] = [X/M] + [M/H]
                value += result_kwds[r["spectrum_pk"]]["m_h_atm"]
                e_value = np.sqrt(e_value**2 + result_kwds[r["spectrum_pk"]]["e_m_h_atm"]**2)

            kwds = {
                f"{label}_rchi2": r["rchi2"],
                f"{label}": value,
                f"e_{label}": e_value,
                f"raw_{label}": value,
                f"raw_e_{label}": e_value,
                f"{label}_flags": FerreChemicalAbundances(**r).ferre_flags
            }
            result_kwds[r["spectrum_pk"]].update(kwds)

        spectra_by_pk = {s.spectrum_pk: s for s in spectra}
        for spectrum_pk, kwds in result_kwds.items():
            yield ASPCAP.from_spectrum(spectra_by_pk[spectrum_pk], **kwds)

        f = np.random.randn()
        if not isinstance(parent, str):
            debugger(f"{f:.2f} closing parent")
            parent.close()
            debugger(f"{f:.2f} closed parent. closing child")
            child.close()

        debugger(f"{f:.2f} closing child. shutting down executor")
        #executor.shutdown(wait=False, cancel_futures=True)
        #debugger(f"{f:.2f} shut down executor")

def _aspcap_stage(
    stage,
    plans,
    executor,
    parent,
    child,
    parent_dir,
    max_processes,
    max_threads,
    max_concurrent_loading,
    soft_thread_ratio,
    progress=None,
    ferre_kwds=None,
    use_ferre_list_mode=False,
):
    pb = None
    if progress is not None:
        full_names = {
            "coarse": "Coarse parameters",
            "params": "Stellar parameters",
            "abundances": "Chemical abundances"
        }
        stage_name = f"{full_names.get(stage, stage)}"
        if os.getenv("CLUSTER", False):
            stage_name += f" @ {HOSTNAME}"
        stage_task_id = progress.add_task(f"[bold blue]{stage_name}[/bold blue]")

    def get_task_name(path):
        # Display label for the progress bar. Callers must pass the full input_nml_path:
        # the positional unpacking below counts back from the file name, so handing it a
        # directory shifts every component by one.
        if stage == "abundances" and not use_ferre_list_mode:
            *__, stage_name, grid_name, species, base_name = path.split("/")
            return f"{grid_name}/{species}"
        else:
            *__, stage_name, task_name, base_name = path.split("/")
            return task_name

    def get_timing_key(pwd):
        # Key for the `timings` dict, which is per-stage and holds one entry per FERRE
        # execution. This deliberately does NOT reuse get_task_name: that function is fed
        # a directory here, and its off-by-one then collapsed every grid in a stage onto
        # the same key (e.g. "params"), so each execution silently overwrote the previous
        # one's per-spectrum timings and only the last execution's spectra kept a time.
        # For coarse and params the working directory is unique per execution, so it is
        # a sound key on its own.
        if stage == "abundances" and not use_ferre_list_mode:
            # Not so here: pre_process_ferre hands back the grid directory, which every
            # species under that grid shares, so it cannot tell two executions apart.
            # Return None so nothing is recorded and the time stays null, rather than
            # crediting one species with another's runtime. Fixing this needs a key that
            # carries the species (see ferre_time_abundances, which is never assigned).
            return None
        return os.path.normpath(pwd)

    # FERRE can be limited by at least three mechanisms:
    # 1. Too many threads requested (CPU limited).
    # 2. Too many processes started (RAM limited).
    # 3. Too many grids load at once (disk I/O limited).
    successes, failures, ferre_tasks = ([], {}, {})
    current_processes, current_threads, currently_loading = (0, 0, 0)
    pre_processed_futures, ferre_futures, post_processed_futures = ([], [], [])
    n_started_executions, n_planned_executions, timings = (0, len(plans), {})
    spectrum_primary_keys_causing_timeout = []
    pre_process_exceptions = []

    at_capacity = lambda p, t, c: (
        p >= max_processes,
        (t >= (soft_thread_ratio * max_threads)) and (p > 0),
        c >= max_concurrent_loading
    )

    total = 0
    for plan in plans:
        (
            executor
            .submit(_safe_pre_process_ferre, plan)
            .add_done_callback(lambda future: pre_processed_futures.insert(0, future))
        )
        total += sum(map(len, (p["spectra"] for p in plan)))
    if progress is not None:
        progress.update(stage_task_id, completed=0, total=total)
    else:
        pb = tqdm(total=total, desc=f"ASPCAP {stage}")
        pb.__enter__()


    def check_capacity(current_processes, current_threads, currently_loading):

        debugger(f"check capacity {current_processes} {current_threads} {currently_loading} {len(ferre_futures)}")
        while parent.poll():
            #debugger("awaiting message")
            state = parent.recv()
            if timeout_on_spectrum_pk := state.get("timeout_on_spectrum_pk", None):
                spectrum_primary_keys_causing_timeout.append(timeout_on_spectrum_pk)
            else:
                delta_n_loading = state.get("n_loading", 0)
                delta_n_complete = state.get("n_complete", 0)
                currently_loading += delta_n_loading
                current_threads += state.get("n_threads", 0)
                current_processes += state.get("n_processes", 0)
                debugger(f"state: {state}")
                if progress is not None:
                    progress_kwds = dict(advance=delta_n_complete)
                    task_name = get_task_name(state['input_nml_path'])
                    if delta_n_loading != 0:
                        color = "yellow" if delta_n_loading > 0 else "white"
                        progress_kwds.update(description=f"  [{color}]{task_name}")
                    progress.update(ferre_tasks[task_name], **progress_kwds)
                    progress.update(stage_task_id, advance=delta_n_complete)

                    try:
                        if stage == "abundances" and not use_ferre_list_mode and progress._tasks[ferre_tasks[task_name]].completed:
                            progress.update(ferre_tasks[task_name], visible=False, refresh=True)
                    except:
                        None

                    #for task_id in ferre_tasks.values():
                    #progress.update(task_id, completed=True, visible=False, refresh=True)

                elif pb is not None:
                    worker_limit, thread_limit, loading_limit = at_capacity(current_processes, current_threads, currently_loading)
                    pb.set_description(
                        f"ASPCAP {stage} ("
                        f"thread {current_threads}/{max_threads}{'*' if thread_limit else ''}; "
                        f"proc {current_processes}/{max_processes}{'*' if worker_limit else ''}; "
                        f"load {currently_loading}/{max_concurrent_loading}{'*' if loading_limit else ''}; "
                        f"job {n_started_executions}/{n_planned_executions})"
                    )
                    pb.update(delta_n_complete)
            #debugger("ok")

        #debugger("getting ferre future")
        try:
            ferre_future = next(concurrent.futures.as_completed(ferre_futures, timeout=0))
        except (concurrent.futures.TimeoutError, StopIteration):
            None
        else:
            debugger(f"got a ferre future")
            (input_nml_path, pwd, return_code, t_overhead, t_elapsed) = ferre_future.result()

            debugger(f"READY TO POST_PROCESS: {input_nml_path} {return_code} {pwd}")
            """
            try:
                if "input_list.nml" in input_nml_path:
                    from glob import glob
                    for sub_path in glob(os.path.dirname(input_nml_path) + "/*/input.nml"):
                        for basename in ("parameter.output", "rectified_flux.output", "rectified_model_flux.output"):
                            p = os.path.join(os.path.dirname(sub_path), basename)
                            debugger(f"{os.path.dirname(sub_path)} {p} {os.path.exists(p)} {os.path.getsize(p) if os.path.exists(p) else None}")

                else:
                    for basename in ("parameter.output", "rectified_flux.output", "rectified_model_flux.output"):
                        p = os.path.join(os.path.dirname(input_nml_path), basename)
                        debugger(f"{p} {os.path.exists(p)} {os.path.getsize(p) if os.path.exists(p) else None}")

                os.system("ps -ef | grep ferre")
                post_process_ferre(input_nml_path, pwd)

                debugger("Did it")
            except:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            """

            # Keyed by the execution's working directory so that concurrent grids within
            # a stage do not overwrite each other. `pwd` is what post-processing reports
            # back on each result, so store under the same value it will look up with.
            timing_key = get_timing_key(pwd)
            if timing_key is not None:
                timings[timing_key] = (t_overhead, t_elapsed)
            post_processed_futures.append(executor.submit(_safe_post_process_ferre, input_nml_path, pwd))
            ferre_futures.remove(ferre_future)
            debugger("removed ferre futures")

        #debugger(f"CHECKING CAPACITY: {current_processes} {current_threads} {currently_loading}")

        # check that there isn't some aggregating error in the capacity check.
        if current_processes == 0 and currently_loading == 0 and current_threads > 0:
            debugger("resetting thread count")
            current_threads = 0

        return (current_processes, current_threads, currently_loading)

    while n_planned_executions > n_started_executions:
        try:
            # Let's oscillate between the first (largest) and last (smallest) elements: (-0 and -1)
            # This means we are distributing the grid loading time while other threads are doing useful things.
            future = pre_processed_futures.pop(-((n_started_executions + 1) % 2))
        except IndexError:
            continue
        else:
            result = future.result()
            if isinstance(result, Exception):
                # Pre-processing failed for this plan, so there is nothing to hand to FERRE.
                # Count it as started so this loop still terminates, and keep the exception
                # so the stage can fail loudly below rather than quietly yielding nothing.
                pre_process_exceptions.append(result)
                n_started_executions += 1
                continue
            input_nml_path, pwd, n_obj, n_ferre_threads, skipped = result

            # Spectra might be skipped because the file could not be found, or if there were too many bad pixels.
            for spectrum, kwds in skipped:
                # TODO: check whether this progress should be communicated through the pipe
                if progress is not None:
                    progress.update(stage_task_id, advance=1)
                if pb is not None:
                    pb.update(1)
                failures[spectrum.spectrum_pk] = ASPCAP.from_spectrum(spectrum, **kwds)

            while True:
                debugger("l513")
                current_processes, current_threads, currently_loading = check_capacity(current_processes, current_threads, currently_loading)
                if not any(at_capacity(current_processes, current_threads, currently_loading)):
                    break

            if n_obj > 0:
                ferre_futures.append(executor.submit(_safe_ferre, input_nml_path, pwd, n_obj, n_ferre_threads, child, communicate_on_start=False, **(ferre_kwds or {})))
                # Do the communication here ourselves because otherwise we will submit too many jobs before they start.
                if progress is not None:
                    task_name = get_task_name(input_nml_path)
                    ferre_tasks[task_name] = progress.add_task(task_name, total=n_obj)

                child.send(dict(input_nml_path=input_nml_path, n_processes=1, n_loading=1, n_threads=n_ferre_threads))

            n_started_executions += 1

    # All submitted. Now wait for them to finish.
    while len(ferre_futures) > 0:
        current_processes, current_threads, currently_loading = check_capacity(current_processes, current_threads, currently_loading)

    debugger(f"waiting for ferre futures {len(ferre_futures)} {len(post_processed_futures)}")
    for future in concurrent.futures.as_completed(post_processed_futures):
        # If the number of spectra being processed in one job gets too large, we might need to write the timing information to a temporary file
        # in the child thread, and have the parent pick it up.
        debugger("got a result")
        for result in future.result():
            debugger(f"result -> {result}")
            # Assign timings to the results.
            result_pwd = result.get("pwd")
            key = get_timing_key(result_pwd) if result_pwd else None
            try:
                t_overhead, t_elapsed_all = timings[key]
                t_elapsed = t_elapsed_all[result["ferre_name"]]
            except:
                # Timings are best-effort, but losing them silently is how they went
                # unnoticed before: say which spectrum and key missed.
                debugger(
                    f"no timing for {result.get('ferre_name')!r} under key {key!r} "
                    f"(have {sorted(timings)})"
                )
                t_elapsed = t_overhead = np.nan
            finally:
                debugger("ok")
                result["t_overhead"] = t_overhead
                result["t_elapsed"] = np.sum(np.atleast_1d(t_elapsed))

            if result["spectrum_pk"] in spectrum_primary_keys_causing_timeout:
                result["flag_caused_timeout"] = True
                debugger(f"assigned {result['spectrum_pk']} as causing timeout")

            # A spectrum whose row was never actually written by FERRE (excluded as a named
            # culprit, or abandoned after resume attempts ran out) comes back with a NaN rchi2 --
            # FERRE pads unreached rows with NaN and nothing overwrites them. Anyone who was
            # merely delayed and successfully retried has a real, finite rchi2 by this point, so
            # this only flags spectra whose result is actually incomplete/unreliable.
            if not np.isfinite(result.get("rchi2", np.nan)):
                result["flag_affected_by_timeout"] = True

            successes.append(result)

    debugger(f"doing thing {post_processed_futures}")
    if progress is not None:
        for task_id in ferre_tasks.values():
            progress.update(task_id, completed=True, visible=False, refresh=True)
    else:
        pb.__exit__(None, None, None)
    debugger(f"_aspcap_stage[{stage}] about to return: {len(successes)} successes, {len(failures)} failures")
    if pre_process_exceptions:
        # A pre-processing failure means those spectra produced no results at all. Returning
        # normally here is what allowed a full disk to look like a successful run, so surface
        # it -- keeping the first real exception as the cause.
        first = pre_process_exceptions[0]
        raise RuntimeError(
            f"ASPCAP {stage}: pre-processing failed for {len(pre_process_exceptions)} of "
            f"{n_planned_executions} execution plan(s); those spectra have no results. "
            f"First error: {first.__class__.__name__}: {first}"
        ) from first
    return (successes, list(failures.values()))


REGEX_NEXT_OBJECT = re.compile(r"next object #\s+(\d+)")
REGEX_COMPLETED = re.compile(r"\s+(\d+)\s(\d+_[\d\w_]+)") # assumes input ids start with an integer and underscore
# Marks the beginning of a stretch in which FERRE is legitimately silent: its startup banner (printed
# once per execution, and once per entry in list (-l) mode) and the end of the grid load, after which
# every thread is busy on its first object and nothing has completed yet.
REGEX_EXECUTION_START = re.compile(r"f e r r e\s+v|Done reading")

# Minimum number of completed objects before the median/stddev of their fit times describes a
# real distribution. With a single completion np.std is exactly 0, and the old absolute 10s
# floor then made the sigma test roughly 100x too sensitive on a ~1000s median: everything in
# a synchronized wave crossed "10 sigma" within seconds of each other and was permanently
# abandoned. Below this many samples the test is skipped entirely and max_t_communicate stays
# the guard against a genuinely dead process.
MIN_OUTLIER_SAMPLES = 10

# Floor the spread relative to the median rather than at an absolute 10s. Healthy batches
# measure a spread of 20-35% of the median, so 25% sits inside the real distribution -- it will
# not loosen the test where genuine statistics exist, but it stops the degenerate collapse when
# every observed completion happened to take about the same time.
MIN_RELATIVE_STDDEV = 0.25


def ferre(
    input_nml_path,
    cwd,
    n_obj,
    n_threads,
    pipe,
    max_sigma_outlier=15,  # recent tests found max to be 8.8, so some buffer
    max_t_elapsed=3000,  # tail smaple in recent test found to reach 1800s
    max_t_grid_load=3600,  # 600, 1000 -- raised: loading the largest grids (~30GB) has been observed
                            # taking up to ~1600s, leaving little margin at 1000s against normal I/O variance.
    max_t_communicate=1000,  # 600,
    max_t_communicate_first_result=3600,
    communicate_on_start=True,
    max_resume_attempts=5,
    _resume_attempt=0,
):

    # Whether the process slot this call is responsible for has been handed back yet -- either by us,
    # or by a resumed sub-run we delegated it to. The bare `except` below uses this to avoid leaking
    # the slot (which would wedge the dispatch loop at capacity) without double-releasing it.
    # Declared outside the try so it is always bound, even if we fail immediately.
    # communicate_on_start=False means the caller already sent the +1, so we owe a release from the
    # very first instruction; otherwise we owe nothing until our own +1 below succeeds.
    slot_released = communicate_on_start

    try:
        if communicate_on_start:
            pipe.send(dict(input_nml_path=input_nml_path, n_processes=1, n_loading=1, n_threads=max(0, n_threads)))
            slot_released = False

        is_list_mode = _is_list_mode(input_nml_path)

        ferre_hanging = threading.Event()
        stdout, n_complete, t_start, t_last_communication, t_overhead, t_awaiting, t_elapsed, exclude_indices, n_threads_to_release = ([], 0, time(), time(), None, {}, {}, [], max(0, n_threads))
        # Guards concurrent access to t_awaiting and t_elapsed between the main reader loop and the monitor thread.
        state_lock = threading.Lock()
        # True while we are between the start of a (sub-)execution and its first completed object.
        # Silence is expected in that window -- the grid load produces no output, and afterwards every
        # thread is busy on its first fit with nothing finished yet -- so it gets a much larger
        # communication budget than the steady state, where completions and dispatches interleave.
        awaiting_first_result = True

        command = ["ferre.x"]
        if is_list_mode:
            command += ["-l", os.path.basename(input_nml_path)]
        else:
            if input_nml_path.startswith(cwd):
                command.append(input_nml_path[len(cwd):].lstrip("/"))
            else:
                command.append(input_nml_path)

        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, close_fds=True)

        def monitor():
            try:
                while not ferre_hanging.is_set():
                    # Snapshot the shared dicts under the lock so subsequent reads are race-free.
                    with state_lock:
                        t_awaiting_snapshot = dict(t_awaiting)
                        t_elapsed_snapshot = {k: list(v) for k, v in t_elapsed.items()}

                    debugger(f"monitor {max_sigma_outlier} {max_t_elapsed} {t_awaiting_snapshot} in {cwd}")

                    # Two-phase budget. Before a (sub-)execution's first result, silence is expected
                    # and its duration scales with grid size and NTHREADS, so allow much longer;
                    # afterwards completions and dispatches interleave continuously and a long gap
                    # really does mean trouble. Keying off awaiting_first_result rather than
                    # t_overhead matters for list (-l) mode, where every entry reloads the grid and
                    # has its own silent first wave, but t_overhead is only ever set once.
                    budget = max_t_communicate_first_result if awaiting_first_result else max_t_communicate
                    if (budget is not None and (time() - t_last_communication) > budget):
                        debugger(f"hanging no communication (awaiting_first_result={awaiting_first_result}, budget={budget})")
                        # Total silence means everything outstanding is stuck, so name them here --
                        # otherwise the resume feeds the same hung objects straight back in.
                        # Populate before ferre_hanging, which is what the reader loop breaks on.
                        # FERRE's indices are 1-indexed; exclude_indices is 0-indexed.
                        exclude_indices.extend([(k - 1) for k in t_awaiting_snapshot])
                        debugger(f"no communication, excluding {len(t_awaiting_snapshot)} outstanding objects")
                        ferre_hanging.set()
                        try:
                            process.kill()
                        except:
                            None
                        break


                    if (
                        ((max_sigma_outlier is not None or max_t_elapsed is not None) and t_awaiting_snapshot)
                    or  (max_t_grid_load is not None and t_overhead is None and ((time() - t_start) > max_t_grid_load))
                    ):
                        n_await = len(t_awaiting_snapshot)
                        n_execution = 0 if len(t_elapsed_snapshot) == 0 else max(list(map(len, t_elapsed_snapshot.values())))
                        n_complete = sum([len(v) == n_execution for v in t_elapsed_snapshot.values()])

                        t_elapsed_per_spectrum_execution = []
                        for k, v in t_elapsed_snapshot.items():
                            t_elapsed_per_spectrum_execution.extend(v)

                        debugger(f"checking on {n_await} things {len(t_awaiting_snapshot)} {len(t_elapsed_per_spectrum_execution)} {max_t_elapsed} {max_sigma_outlier}")

                        if (
                        (len(t_awaiting_snapshot) > 0)
                        and (max_t_elapsed is not None or max_sigma_outlier is not None)
                        # This test compares each object's wait time against the median/stddev of
                        # completed objects -- before the first result, there is no such
                        # distribution to compare against (median/stddev below are a hardcoded
                        # fallback, not measured), so any wait looks like an arbitrarily large
                        # outlier. That is the same failure mode max_t_communicate_first_result
                        # exists for; skip this test until it has real data, and let that budget be
                        # the sole guard for the pre-first-result window.
                        and not awaiting_first_result
                        ):

                            # A completion whose `next object #` entry could not be matched records
                            # NaN (see where t_elapsed is appended). Those carry no timing
                            # information, and one of them would otherwise turn the median and
                            # stddev of the whole sample into NaN -- which, since nothing ever
                            # removes them, would disable this watchdog for the rest of the
                            # execution. Judge on the samples that actually measured something.
                            finite_samples = [
                                t for t in t_elapsed_per_spectrum_execution if np.isfinite(t)
                            ]
                            n_samples = len(finite_samples)
                            if n_samples == 0:
                                median = stddev = np.nan
                            else:
                                median = np.median(finite_samples)
                                stddev = np.std(finite_samples)

                            # Only judge outliers once the completed sample actually describes a
                            # distribution. `awaiting_first_result` covers the zero-completion case;
                            # this covers the barely-any-completions case, which is just as
                            # degenerate -- one completion gives stddev 0, and the old 10s floor
                            # then flagged an entire healthy 127-object wave at once.
                            if not (
                                n_samples >= MIN_OUTLIER_SAMPLES
                                and np.isfinite(median)
                                and np.isfinite(stddev)
                            ):
                                sleep(1)
                                continue

                            stddev = max(stddev, MIN_RELATIVE_STDDEV * median)

                            t_awaiting_elapsed = { k: (time() + v) for k, v in t_awaiting_snapshot.items() }
                            if not t_awaiting_elapsed:
                                # Snapshot was non-empty but the comprehension produced nothing — defensive guard.
                                sleep(1)
                                continue
                            waiting_elapsed = max(t_awaiting_elapsed.values())
                            sigma_outlier = (waiting_elapsed - median)/stddev


                            # We want to be sure that we have a reasonable estimate of the wait time for existing things.
                            # We can use previous executions to estimate this, if it is part of a list mode.
                            is_hanging = [
                                # The indices we get from FERRE stdout are 1-indexed, not 0-indexed.
                                (k - 1) for k, v in t_awaiting_elapsed.items()
                                if (
                                    (max_t_elapsed is None or v > max_t_elapsed)
                                and (max_sigma_outlier is None or ((v - median)/stddev) > max_sigma_outlier)
                                )
                            ]
                            debugger(f"median / stddev {median:.2} {stddev:.2f} {t_awaiting_elapsed} {waiting_elapsed:.2f} {sigma_outlier} {is_hanging}")

                            # TODO: strace the process and check that it is waiting on FUTEX_PRIVATE_WAIT before killing it?
                            # Need to kill and re-run the process when either out of objects
                            # or resources taken up by hanging objects.
                            #
                            # Two conditions, not one. Requiring every outstanding object to look hung
                            # lets healthy work keep flowing while a few threads are wedged. But the
                            # outstanding set also shrinks as an execution drains, so near the tail
                            # "all of them" can mean one or two objects -- and this test cannot tell a
                            # genuinely stuck object from a merely slow one. Requiring a real share of
                            # the pool as well keeps the tail out of this test's hands and leaves it to
                            # max_t_communicate, where sustained silence is far stronger evidence. The
                            # floor scales with whatever can actually be in flight, so it does not
                            # disable the test on executions with fewer objects than threads.
                            n_out = len(t_awaiting_elapsed)
                            n_hanging = len(is_hanging)
                            min_hanging = max(1, min(max(0, n_threads), n_obj) // 2)
                            if n_hanging >= max(1, n_out) and n_hanging >= min_hanging:
                                exclude_indices.extend(is_hanging)
                                debugger(f"hanging {is_hanging}")
                                ferre_hanging.set()
                                try:
                                    process.kill()
                                except Exception as e:
                                    debugger(f"exception trying to kill dying: {e}")
                                break
                        elif (t_overhead is None and max_t_grid_load is not None and (time() - t_start) > max_t_grid_load):
                            debugger(f"hanging on grid load {input_nml_path}")
                            ferre_hanging.set()
                            try:
                                process.kill()
                            except Exception as e:
                                debugger(f"exception trying to kill dying: {e}")
                            break

                    sleep(1)
            except Exception as e:
                debugger(f"MONITOR DIED {input_nml_path}")
                debugger(e)
                # We better kill ferre because we can't monitor it anymore.
                ferre_hanging.set()
                try:
                    process.kill()
                except Exception as e:
                    debugger(f"exception trying to kill dying: {e}")


        monitor = threading.Thread(target=monitor)
        monitor.daemon = True
        monitor.start()

        while True:
            line = process.stdout.readline()

            if REGEX_EXECUTION_START.search(line):
                # A new (sub-)execution is starting, or has just finished loading its grid. Either
                # way an expected silent stretch follows, so restart the clock on the larger budget.
                # In list (-l) mode this happens once per entry, which is what keeps the elements
                # after the first from being killed during their own grid loads.
                t_last_communication = time()
                awaiting_first_result = True

            if match := REGEX_NEXT_OBJECT.search(line):
                t_last_communication = time()
                with state_lock:
                    t_awaiting[int(match.group(1))] = -time()
                if t_overhead is None:
                    t_overhead = time() - t_start
                    pipe.send(dict(input_nml_path=input_nml_path, n_loading=-1))

            if match := REGEX_COMPLETED.search(line):
                t_last_communication = time()
                # Results are flowing for this (sub-)execution: tighten to the steady-state budget.
                awaiting_first_result = False

                key = match.group(2)
                with state_lock:
                    t_elapsed.setdefault(key, [])
                    try:
                        # The leading counter FERRE prints on a completion line is not the object
                        # index, so it cannot be used to clear the entry that `next object #` added.
                        # The object name is authoritative: its prefix is the index in parameter.input.
                        t = t_awaiting.pop(int(key.split("_")[0]) + 1) + time()
                    except:
                        # A miss leaves an entry that ages forever and is now excluded on a wedge,
                        # costing a healthy spectrum -- so make it visible rather than silent.
                        debugger(f"could not clear t_awaiting for completed {key} in {cwd}")
                        t = np.nan
                    t_elapsed[key].append(t)
                n_complete += 1
                n_remaining = n_obj - n_complete

                delta_n_threads = -1 if n_remaining < n_threads and n_threads_to_release > 0 else 0
                n_threads_to_release += delta_n_threads
                pipe.send(dict(input_nml_path=input_nml_path, n_complete=1, n_threads=delta_n_threads))

            if not line or ferre_hanging.is_set():
                break

            stdout.append(line)

        stderr = process.stderr.read()
        return_code = int(process.wait())
        try:
            process.kill()
        except:
            None
        process.stdout.close()
        process.stderr.close()

        # n_processes=-1 (releasing the process slot) is sent explicitly in each branch below rather
        # than unconditionally here, because a hang that leads to a resume is the SAME logical job
        # continuing -- releasing the slot here and re-acquiring it when the resume starts would open
        # a window where the dispatch loop can hand the "freed" slot to a completely different job,
        # exceeding max_processes for as long as both run concurrently.

        with open(os.path.join(cwd, f"stdout"), "w") as fp:
            fp.write("".join(stdout))

        with open(os.path.join(cwd, f"stderr"), "w") as fp:
            fp.write(stderr)

        if ferre_hanging.is_set():
            n_execution = 0 if len(t_elapsed) == 0 else max(list(map(len, t_elapsed.values())))
            n_spectra_done_in_last_execution = len([v for v in t_elapsed.values() if len(v) == n_execution])

            if is_list_mode:
                # List mode fits one element at a time over the same stars, so a kill always lands
                # inside one specific element. Rebuild THAT element without the objects that hung and
                # queue it ahead of the elements that have not started yet.
                #
                # The two things this must not do, both of which the previous version did:
                #   * restart the element unchanged -- it re-hits the same star and dies in the same
                #     place, which is what produced runs of 45 input_list.nml.N files and left every
                #     element truncated at an identical star index; and
                #   * drop the element wholesale -- each element is fitted through its own mask over a
                #     different window of the spectrum, so a star that cannot be fitted for one element
                #     is often perfectly measurable for the next.
                #
                # Because every resume now removes at least one object, the remaining work strictly
                # decreases and the recursion terminates on its own. That progress guarantee is what
                # bounds it -- deliberately not a retry counter, which would abandon the survivors.
                if n_spectra_done_in_last_execution > 0:
                    pipe.send(dict(input_nml_path=input_nml_path, n_complete=-n_spectra_done_in_last_execution))

                with open(input_nml_path, "r") as fp:
                    paths = [line.strip() for line in fp.readlines() if line.strip()]

                # n_execution counts how many executions the furthest-along spectrum has been through,
                # so it indexes the element that was running when we killed FERRE.
                current_index = min(max(n_execution - 1, 0), max(len(paths) - 1, 0))
                current_relative_path = paths[current_index] if paths else None
                remaining_relative_paths = paths[current_index + 1:]

                resumed_relative_path = None
                new_element_nml_path = None
                if current_relative_path is not None:
                    try:
                        element_dir = os.path.dirname(current_relative_path)
                        new_element_nml_path, _ = re_process_partial_ferre(
                            os.path.join(cwd, current_relative_path),
                            cwd,
                            exclude_indices=exclude_indices,
                            # Keep the rebuilt flux/e_flux slices inside the element's own directory:
                            # in this layout they are shared with every other element at the parent.
                            relocate_into=element_dir or None,
                        )
                        if new_element_nml_path is not None:
                            resumed_relative_path = os.path.relpath(new_element_nml_path, cwd)
                    except Exception as e:
                        debugger(f"exception rebuilding list element {current_relative_path}: {e}")

                if resumed_relative_path is None:
                    debugger(
                        f"list element {current_relative_path} has nothing left to run "
                        f"(excluded {len(exclude_indices)}); moving on to {len(remaining_relative_paths)} remaining"
                    )

                next_relative_paths = (
                    ([resumed_relative_path] if resumed_relative_path else []) + remaining_relative_paths
                )

                prefix, suffix = input_nml_path.split(".nml")
                suffix = suffix.lstrip(".")
                suffix = (int(suffix) + 1) if suffix != "" else 1
                new_path = f"{prefix}.nml.{suffix}"

                # Release these threads so it's balanced out when the sub-ferre process takes them. The
                # process slot itself is deliberately NOT released here -- see the note above.
                pipe.send(dict(input_nml_path=input_nml_path, n_threads=-n_threads))

                if len(next_relative_paths) > 0:
                    with open(new_path, "w") as fp:
                        fp.write("\n".join(next_relative_paths) + "\n")

                    debugger(
                        f"resuming list {new_path} with {len(next_relative_paths)} element(s), "
                        f"excluded {len(exclude_indices)} object(s) from {current_relative_path}"
                    )

                    # Re-declare the loading/thread need for the resumed sub-run, but not n_processes
                    # (see the note above) -- communicate_on_start=False suppresses its own declaration.
                    pipe.send(dict(input_nml_path=input_nml_path, n_loading=1, n_threads=n_threads))
                    *_, this_t_overhead, this_t_elapsed = ferre(
                        new_path,
                        cwd,
                        n_obj - n_complete + n_spectra_done_in_last_execution,
                        n_threads,
                        pipe,
                        max_sigma_outlier=max_sigma_outlier,
                        max_t_elapsed=max_t_elapsed,
                        # Previously omitted, so every resumed sub-run silently reverted to the
                        # function defaults instead of the caller's configuration.
                        max_t_grid_load=max_t_grid_load,
                        max_t_communicate=max_t_communicate,
                        max_t_communicate_first_result=max_t_communicate_first_result,
                        communicate_on_start=False,
                    )
                    slot_released = True  # the resumed sub-run owns the release now

                    t_overhead = (t_overhead or 0) + this_t_overhead
                    for k, v in this_t_elapsed.items():
                        t_elapsed.setdefault(k, [])
                        t_elapsed[k].extend(v)

                    # Fold the rebuilt element's results back into the output files named by its
                    # ORIGINAL namelist, which is what post-processing reads (post_process_ferre
                    # walks the original input_list.nml, so it opens C/parameter.output, never
                    # C/parameter.output.1). Without this the resumed rows are stranded in the
                    # `.N` files and then overwritten with NaN, because `vaffoff` pads every name
                    # that is missing from the output with a NaN row.
                    #
                    # This must run AFTER the recursive call returns: if the same element hangs
                    # again inside it, that level merges its `.2` into `.1` before returning, so
                    # merging `.1` into the original here folds in both.
                    if new_element_nml_path is not None:
                        try:
                            merge_partial_ferre_outputs(
                                os.path.join(cwd, current_relative_path),
                                new_element_nml_path,
                                cwd,
                            )
                        except Exception as e:
                            debugger(f"exception merging resumed list element {current_relative_path}: {e}")
                else:
                    # Every element is accounted for -- truly done, release the process slot.
                    pipe.send(dict(input_nml_path=input_nml_path, n_processes=-1))
                    slot_released = True

            else:
                debugger(f"hanging on {input_nml_path} {cwd} {return_code} {t_overhead} {t_elapsed}")

                # Report the spectra that were being waited on when we killed FERRE, so they can be
                # flagged as having caused the timeout.
                try:
                    if is_list_mode:
                        # input_nml_path is itself a list file whose lines point to per-element
                        # sub-.nml files (Al/input.nml, C/input.nml, ...), each with its own PFILE
                        # relative to its own subdirectory -- there is no cwd/parameter.input here.
                        # n_execution identifies which list entry was in progress (same logic used
                        # to resume at "unprocessed_input_nml_paths" above). If it's still 0, nothing
                        # in the list ever produced a completion (e.g. died during the first grid
                        # load) and there's no reliable way to name which element was active.
                        if n_execution == 0:
                            raise RuntimeError("no completions yet in list mode; cannot identify active sub-execution")
                        with open(input_nml_path, "r") as fp:
                            list_paths = list(map(str.strip, fp.readlines()))
                        active_nml_path = os.path.join(cwd, list_paths[n_execution - 1])
                    else:
                        # Non-list mode, including resumed sub-runs: input_nml_path IS the .nml
                        # actually executed (e.g. foo.nml.1 on a resume), so its own PFILE is
                        # authoritative -- a resumed run's PFILE is parameter.input.1, a filtered
                        # subset, not the original parameter.input.
                        active_nml_path = input_nml_path

                    active_pwd = os.path.dirname(active_nml_path)
                    control_kwds = read_control_file(active_nml_path)
                    parameter_input_path = os.path.join(active_pwd, control_kwds["PFILE"])
                    input_names = np.loadtxt(parameter_input_path, usecols=(0, ), dtype=str)

                    # Flag only the spectra actually identified as the cause (exclude_indices, from
                    # the sigma-outlier test) when we have them. When the kill instead came from
                    # max_t_communicate/max_t_communicate_first_result there is no named culprit, so
                    # fall back to flagging everything still in flight.
                    if exclude_indices:
                        keys_to_flag = [i + 1 for i in exclude_indices if (i + 1) in t_awaiting]
                    else:
                        keys_to_flag = list(t_awaiting.keys())

                    for index_1_based in keys_to_flag:
                        parsed = parse_ferre_spectrum_name(input_names[int(index_1_based) - 1])
                        pipe.send(dict(timeout_on_spectrum_pk=parsed["spectrum_pk"]))

                except Exception as e:
                    debugger(f"exception reporting timeout spectra: {e}")

                # Resume the remainder in a new FERRE execution, excluding the objects that hung.
                # Without this we would abandon every object that had not been reached yet, which for
                # a large grid can be thousands of spectra lost because one or two of them stalled.
                new_input_nml_path = None
                if _resume_attempt < max_resume_attempts and (n_complete > 0 or exclude_indices):
                    try:
                        new_input_nml_path, ignore_names = re_process_partial_ferre(
                            input_nml_path, cwd, exclude_indices=exclude_indices
                        )
                    except Exception as e:
                        debugger(f"exception preparing resume for {input_nml_path}: {e}")
                        new_input_nml_path = None
                else:
                    debugger(
                        f"not resuming {input_nml_path}: attempt {_resume_attempt}/{max_resume_attempts}, "
                        f"n_complete={n_complete}, n_excluded={len(exclude_indices)}"
                    )

                if new_input_nml_path is None:
                    # Nothing left to run (or we have given up). Close out the progress for whatever
                    # never ran, and release the process slot -- this really is done now.
                    pipe.send(dict(input_nml_path=input_nml_path, n_threads=-n_threads_to_release, n_complete=n_obj - n_complete, n_processes=-1))
                    slot_released = True
                else:
                    n_remaining = n_obj - len(set(ignore_names))
                    debugger(
                        f"resuming {new_input_nml_path} with {n_remaining} objects "
                        f"(excluded {len(exclude_indices)}, attempt {_resume_attempt + 1}/{max_resume_attempts})"
                    )
                    # Account for the objects we are permanently abandoning, then release our threads
                    # so the resumed process can take them. The process slot itself is NOT released
                    # (no n_processes=-1) -- this is the same logical job continuing, not a new one.
                    n_abandoned = n_obj - n_complete - n_remaining
                    if n_abandoned > 0:
                        pipe.send(dict(input_nml_path=input_nml_path, n_complete=n_abandoned))
                    pipe.send(dict(input_nml_path=input_nml_path, n_threads=-n_threads_to_release))
                    pipe.send(dict(input_nml_path=input_nml_path, n_loading=1, n_threads=n_threads))

                    *_, this_t_overhead, this_t_elapsed = ferre(
                        new_input_nml_path,
                        cwd,
                        n_remaining,
                        n_threads,
                        pipe,
                        max_sigma_outlier=max_sigma_outlier,
                        max_t_elapsed=max_t_elapsed,
                        max_t_grid_load=max_t_grid_load,
                        max_t_communicate=max_t_communicate,
                        max_t_communicate_first_result=max_t_communicate_first_result,
                        max_resume_attempts=max_resume_attempts,
                        _resume_attempt=_resume_attempt + 1,
                        communicate_on_start=False,
                    )
                    slot_released = True  # the resumed sub-run owns the release now
                    t_overhead = (t_overhead or 0) + (this_t_overhead or 0)
                    for k, v in this_t_elapsed.items():
                        t_elapsed.setdefault(k, [])
                        t_elapsed[k].extend(v)

                    # Post-processing only reads the output files named by the original input file,
                    # so fold the resumed results back into them.
                    try:
                        merge_partial_ferre_outputs(input_nml_path, new_input_nml_path, cwd)
                    except Exception as e:
                        debugger(f"exception merging resumed outputs for {input_nml_path}: {e}")

                debugger(t_awaiting)
                debugger(f"done hanging")

        else:
            # Close out the process in case we didn't grep all the targets from stdout
            # e.g.: /uufs/chpc.utah.edu/common/home/sdss51/sdsswork/mwm/spectro/astra/0.7.0/pipelines/aspcap/2025-02-23-nl4_j6g0/params/lco25m_d_GKd
            pipe.send(dict(input_nml_path=input_nml_path, n_complete=n_obj - n_complete, n_processes=-1))
            slot_released = True


        # Set ferre_hanging to kill the daemon thread.
        ferre_hanging.set()
        return (input_nml_path, cwd, return_code, t_overhead, t_elapsed)
    except:
        # Release first, and defensively: an unreleased slot permanently reduces the dispatch loop's
        # capacity and can deadlock it, so this must not be skipped because some later cleanup step
        # raised (e.g. a name that was never bound if we failed early). Guarded by slot_released so we
        # never double-release one already handed back, or handed off to a resumed sub-run.
        try:
            if not slot_released:
                pipe.send(dict(input_nml_path=input_nml_path, n_processes=-1))
        except:
            None
        try:
            ferre_hanging.set()
        except:
            None
        return (input_nml_path, cwd, -10, locals().get("t_overhead"), locals().get("t_elapsed") or {})
