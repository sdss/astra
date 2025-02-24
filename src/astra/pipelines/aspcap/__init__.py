import os
import numpy as np
import concurrent.futures
import subprocess
import re
import json
import fcntl
import pickle
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
from astra.pipelines.ferre.processing import pre_process_ferre, post_process_ferre
from astra.pipelines.ferre.utils import parse_header_path
from astra.pipelines.aspcap.initial import get_initial_guesses
from astra.pipelines.aspcap.coarse import plan_coarse_stellar_parameters_stage
from astra.pipelines.aspcap.stellar_parameters import plan_stellar_parameters_stage
from astra.pipelines.aspcap.abundances import plan_abundances_stage, get_species
#from astra.pipelines.aspcap.stellar_parameters import stellar_parameters, post_stellar_parameters
from astra.pipelines.aspcap.utils import ABUNDANCE_RELATIVE_TO_H

RAND = np.random.randint(0, 1000)
from socket import gethostname

HOSTNAME = gethostname()

def debugger(*foo):
    with open(f"/scratch/general/nfs1/u6020307/pbs/{HOSTNAME}-{RAND}.log", "a") as fp:
        fp.write(" ".join(map(str, foo)) + "\n")

#cd /scratch/general/nfs1/u6020307/pbs/aspcap-2025-02-12-ax1pzxtc

#subprocess.check_output("echo `hostname`"))

def _is_list_mode(path):
    return "input_list.nml" in path

def _safe_pre_process_ferre(*args, **kwargs):
    try:
        return pre_process_ferre(*args, **kwargs)
    except:
        debugger(f"Exception in pre_process_ferre {args} {kwargs}")
        #input_nml_path, pwd, n_obj, n_ferre_threads, skipped


def _safe_post_process_ferre(*args, **kwargs):
    try:
        return post_process_ferre(*args, **kwargs)
    except:
        debugger(f"Exception in post_process_ferre {args} {kwargs}")
        return []

def _safe_ferre(*args, **kwargs):
    try:
        return ferre(*args, **kwargs)
    except:
        debugger(f"Exception in ferre {args} {kwargs}")



@task
def aspcap(
    spectra: Iterable[ApogeeCoaddedSpectrumInApStar], 
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    #header_paths: Optional[Union[Sequence[str], str]] = "/uufs/chpc.utah.edu/common/home/u6020307/vast/aspcap-grids/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    parent_dir: Optional[str] = None, 
    n_threads: Optional[int] = 32,
    max_processes: Optional[int] = 16,
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

        coarse_plans, spectra_with_no_initial_guess = plan_coarse_stellar_parameters_stage(
            spectra=spectra, 
            parent_dir=parent_dir,
            header_paths=header_paths, 
            initial_guess_callable=initial_guess_callable,
            weight_path=weight_path,
            n_threads=n_threads
        )
        for spectrum in spectra_with_no_initial_guess:
            yield ASPCAP.from_spectrum(spectrum, flag_no_suitable_initial_guess=True)

        coarse_results, coarse_failures = _aspcap_stage("coarse", coarse_plans, *stage_args)    
        yield from coarse_failures

        stellar_parameter_plans, best_coarse_results = plan_stellar_parameters_stage(
            spectra=spectra, 
            parent_dir=parent_dir,
            coarse_results=coarse_results,
            weight_path=weight_path,
            n_threads=n_threads
        )

        param_results, param_failures = _aspcap_stage("params", stellar_parameter_plans, *stage_args)
        yield from param_failures

        abundance_plans = plan_abundances_stage(
            spectra=spectra,
            parent_dir=parent_dir,
            stellar_parameter_results=param_results,
            element_weight_paths=element_weight_paths,
            n_threads=n_threads,
            use_ferre_list_mode=use_ferre_list_mode
        )            
        
        abundance_results, abundance_failures = _aspcap_stage("abundances", abundance_plans, *stage_args, use_ferre_list_mode=use_ferre_list_mode, ferre_kwds=dict(max_sigma_outlier=10, max_t_elapsed=60))

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
                coarse_result_flags=coarse.ferre_flags,
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
        if stage == "abundances" and not use_ferre_list_mode:
            *__, stage_name, grid_name, species, base_name = path.split("/")
            return f"{grid_name}/{species}"
        else:
            *__, stage_name, task_name, base_name = path.split("/")
            return task_name
        
    # FERRE can be limited by at least three mechanisms:
    # 1. Too many threads requested (CPU limited).
    # 2. Too many processes started (RAM limited).
    # 3. Too many grids load at once (disk I/O limited).
    successes, failures, ferre_tasks = ([], {}, {})
    current_processes, current_threads, currently_loading = (0, 0, 0)
    pre_processed_futures, ferre_futures, post_processed_futures = ([], [], [])
    n_started_executions, n_planned_executions, timings = (0, len(plans), {})
    spectrum_primary_keys_causing_timeout = []

    at_capacity = lambda p, t, c: (
        p >= max_processes,
        t >= (soft_thread_ratio * max_threads),
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

            # TODO: Should `timings` and `post_process_ferre` take directories or input_nml_paths?
            task_name = get_task_name(os.path.dirname(input_nml_path))
            timings[task_name] = (t_overhead, t_elapsed)
            post_processed_futures.append(executor.submit(_safe_post_process_ferre, input_nml_path, pwd))
            ferre_futures.remove(ferre_future)
            debugger("removed ferre futures")

        #debugger(f"CHECKING CAPACITY: {current_processes} {current_threads} {currently_loading}")
        return (current_processes, current_threads, currently_loading)

    while n_planned_executions > n_started_executions:
        try:
            # Let's oscillate between the first (largest) and last (smallest) elements: (-0 and -1)
            # This means we are distributing the grid loading time while other threads are doing useful things.
            future = pre_processed_futures.pop(-((n_started_executions + 1) % 2))
        except IndexError:
            continue
        else:
            input_nml_path, pwd, n_obj, n_ferre_threads, skipped = future.result()
            
            # Spectra might be skipped because the file could not be found, or if there were too many bad pixels.
            for spectrum, kwds in skipped:
                # TODO: check whether this progress should be communicated through the pipe 
                if progress is not None:
                    progress.update(stage_task_id, advance=1)
                if pb is not None:
                    pb.update(1)
                failures[spectrum.spectrum_pk] = ASPCAP.from_spectrum(spectrum, **kwds)

            while True:
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
            try:
                key = get_task_name(result["pwd"])
                t_overhead, t_elapsed_all = timings[key]
                t_elapsed = t_elapsed_all[result["ferre_name"]]
            except:
                debugger("failure")
                t_elapsed = t_overhead = np.nan
            finally:
                debugger("ok")
                result["t_overhead"] = t_overhead
                result["t_elapsed"] = np.sum(np.atleast_1d(t_elapsed))
            
            if result["spectrum_pk"] in spectrum_primary_keys_causing_timeout:
                result["flag_caused_timeout"] = True
                debugger(f"assigned {result['spectrum_pk']} as causing timeout")
            
            successes.append(result)
    
    debugger(f"doing thing {post_processed_futures}")
    if progress is not None:
        for task_id in ferre_tasks.values():
            progress.update(task_id, completed=True, visible=False, refresh=True)
    else:
        pb.__exit__(None, None, None)
    return (successes, list(failures.values()))


REGEX_NEXT_OBJECT = re.compile(r"next object #\s+(\d+)")
REGEX_COMPLETED = re.compile(r"\s+(\d+)\s(\d+_[\d\w_]+)") # assumes input ids start with an integer and underscore


def ferre(
    input_nml_path,
    cwd,
    n_obj,
    n_threads,
    pipe,
    max_sigma_outlier=10,
    max_t_elapsed=600,
    max_t_grid_load=600,
    max_t_communicate=600,
    communicate_on_start=True
):

    try:            
        if communicate_on_start:
            pipe.send(dict(input_nml_path=input_nml_path, n_processes=1, n_loading=1, n_threads=max(0, n_threads)))

        is_list_mode = _is_list_mode(input_nml_path)

        ferre_hanging = threading.Event()
        stdout, n_complete, t_start, t_last_communication, t_overhead, t_awaiting, t_elapsed, exclude_indices, n_threads_to_release = ([], 0, time(), time(), None, {}, {}, [], max(0, n_threads))

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
                    debugger(f"monitor {max_sigma_outlier} {max_t_elapsed} {t_awaiting} in {cwd}")

                    if (max_t_communicate is not None and (time() - t_last_communication) > max_t_communicate):
                        debugger(f"hanging no communication")        
                        ferre_hanging.set()
                        try:
                            process.kill()
                        except:
                            None
                        break                    


                    if (
                        ((max_sigma_outlier is not None or max_t_elapsed is not None) and t_awaiting)
                    or  (max_t_grid_load is not None and t_overhead is None and ((time() - t_start) > max_t_grid_load))
                    ):
                        n_await = len(t_awaiting)
                        n_execution = 0 if len(t_elapsed) == 0 else max(list(map(len, t_elapsed.values())))
                        n_complete = sum([len(v) == n_execution for v in t_elapsed.values()])
                        
                        t_elapsed_per_spectrum_execution = []
                        for k, v in t_elapsed.items():
                            t_elapsed_per_spectrum_execution.extend(v)

                        debugger(f"checking on {n_await} things {len(t_awaiting)} {len(t_elapsed_per_spectrum_execution)} {max_t_elapsed} {max_sigma_outlier}")

                        if (
                        (len(t_awaiting) > 0)
                        and (max_t_elapsed is not None or max_sigma_outlier is not None)
                        ):
                            
                            if len(t_elapsed_per_spectrum_execution) == 0:
                                median = 120.0
                                stddev = 10.0
                            else:
                                median = np.median(t_elapsed_per_spectrum_execution)
                                stddev = np.std(t_elapsed_per_spectrum_execution)
                                if stddev == 0:
                                    stddev = 10.0

                            t_awaiting_elapsed = { k: (time() + v) for k, v in t_awaiting.items() }                        
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
                            # Need to kill and re-run the process.
                            if is_hanging: 
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

            if match := REGEX_NEXT_OBJECT.search(line):
                t_last_communication = time()
                t_awaiting[int(match.group(1))] = -time()
                if t_overhead is None:
                    t_overhead = time() - t_start
                    pipe.send(dict(input_nml_path=input_nml_path, n_loading=-1))
                                
            if match := REGEX_COMPLETED.search(line):
                t_last_communication = time()
                
                key = match.group(2)
                t_elapsed.setdefault(key, [])
                try:
                    t = t_awaiting.pop(int(match.group(1))) + time()
                except:
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

        pipe.send(dict(input_nml_path=input_nml_path, n_processes=-1)) 

        with open(os.path.join(cwd, f"stdout"), "w") as fp:
            fp.write("".join(stdout))

        with open(os.path.join(cwd, f"stderr"), "w") as fp:
            fp.write(stderr)

        if ferre_hanging.is_set():
            n_execution = 0 if len(t_elapsed) == 0 else max(list(map(len, t_elapsed.values())))
            n_spectra_done_in_last_execution = len([v for v in t_elapsed.values() if len(v) == n_execution])

            if is_list_mode and n_spectra_done_in_last_execution > 0:
                pipe.send(dict(input_nml_path=input_nml_path, n_complete=-n_spectra_done_in_last_execution))

                with open(input_nml_path, "r") as fp:
                    paths = list(map(str.strip, fp.readlines()))
                    unprocessed_input_nml_paths = paths[max(n_execution - 1, 0):]
                            
                    # Re-process that one failed thing
                    """
                    if updated_nml_path is not None:
                        _, __, this_t_overhead, this_t_elapsed = ferre(updated_nml_path, cwd, n_obj // n_execution - n_spectra_done_in_last_execution, -n_threads, pipe)
                        t_overhead += this_t_overhead
                        for k, v in this_t_elapsed.items():
                            t_elapsed.setdefault(k, [])
                            t_elapsed[k].extend(v)
                    #debugger(f"DONE REPROCESSING {failed_input_nml_path} {updated_nml_path}")
                    """
                
                prefix, suffix = input_nml_path.split(".nml")
                suffix = suffix.lstrip(".")
                suffix = (int(suffix) + 1) if suffix != "" else 1
                new_path = f"{prefix}.nml.{suffix}"

                # Release these threads and process so it's balanced out when the sub-ferre process takes them
                pipe.send(dict(input_nml_path=input_nml_path, n_threads=-n_threads)) 

                # Need to check if the last two NML list paths had the same number of rows.
                # If they do, it means we are in an infinite loop.
                if suffix >= 3:
                    with open(f"{prefix}.nml.{suffix - 1}", "r") as fp:
                        n_prev = len(fp.readlines())
                    with open(f"{prefix}.nml.{suffix - 2}", "r") as fp:
                        n_prev_prev = len(fp.readlines())
                    
                    if n_prev == n_prev_prev:
                        debugger(f"detected infinite loop {input_nml_path} {n_prev} {n_prev_prev}")
                        failed_relative_path, *unprocessed_input_nml_paths = unprocessed_input_nml_paths
                        """
                        # Try to run FERRE in non-list mode on the failed path
                        *_, this_t_overhead, this_t_elapsed = ferre(failed_relative_path, cwd, n_obj, n_threads, pipe, max_sigma_outlier, max_t_elapsed)
                        t_overhead = (t_overhead or 0) + this_t_overhead
                        for k, v in this_t_elapsed.items():
                            t_elapsed.setdefault(k, [])
                            t_elapsed[k].extend(v)
                        """
                        # TODO: previously when we did it like this, we got into some weird infinite bug where everything hung forever.
                        # let's skip over it and move on.

                if len(unprocessed_input_nml_paths) > 0:
                    with open(new_path, "w") as fp:
                        fp.write("\n".join(unprocessed_input_nml_paths))                
                    
                    *_, this_t_overhead, this_t_elapsed = ferre(new_path, cwd, n_obj - n_complete + n_spectra_done_in_last_execution, n_threads, pipe, max_sigma_outlier, max_t_elapsed)

                    t_overhead = (t_overhead or 0) + this_t_overhead
                    for k, v in this_t_elapsed.items():
                        t_elapsed.setdefault(k, [])
                        t_elapsed[k].extend(v)
            
            else:
                # Just kill the process. Don't bother with the failed things.
                # TODO: Send back which items we were waiting on at the time, so we can mark them as the problematic ones.
                debugger(f"hanging on {input_nml_path} {cwd} {return_code} {t_overhead} {t_elapsed}")
                # Note that we don't need to send n_processes=-1 because we already sent it above before we wrote out stdout/stderr
                pipe.send(dict(input_nml_path=input_nml_path, n_threads=-n_threads_to_release, n_complete=n_obj - n_complete))
                # Send back the spectrum causing the hanging.
                
                try:
                    parameter_input_path = os.path.join(os.path.dirname(input_nml_path), "parameter.input")
                    input_names = np.loadtxt(parameter_input_path, usecols=(0, ), dtype=str)
            
                    for index_1_based in t_awaiting.keys():
                        parsed = parse_ferre_spectrum_name(input_names[int(index_1_based) - 1])
                        pipe.send(dict(timeout_on_spectrum_pk=parsed["spectrum_pk"]))

                except:
                    None

                debugger(t_awaiting)
                debugger(f"done hanging")

        # Set ferre_hanging to kill the daemon thread.
        ferre_hanging.set()
        return (input_nml_path, cwd, return_code, t_overhead, t_elapsed)
    except:
        ferre_hanging.set()
        return (input_nml_path, cwd, -10, t_overhead, t_elapsed)