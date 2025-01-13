import os
import numpy as np
import concurrent.futures
import subprocess
import re
from multiprocessing import Pipe, Lock
from datetime import datetime
from tempfile import mkdtemp
from typing import Optional, Iterable, List, Tuple, Callable, Union
from tqdm import tqdm
from time import time, sleep

from astra import __version__, task
from astra.utils import log, expand_path
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
from astra.models.aspcap import ASPCAP, FerreCoarse, FerreStellarParameters, FerreChemicalAbundances, Source
from astra.models.spectrum import Spectrum
from astra.pipelines.ferre.pre_process import pre_process_ferres
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import parse_header_path
from astra.pipelines.aspcap.initial import get_initial_guesses
from astra.pipelines.aspcap.coarse import plan_coarse_stellar_parameters_stage
from astra.pipelines.aspcap.stellar_parameters import plan_stellar_parameters_stage
from astra.pipelines.aspcap.abundances import plan_abundances_stage, get_species
#from astra.pipelines.aspcap.stellar_parameters import stellar_parameters, post_stellar_parameters
from astra.pipelines.aspcap.utils import ABUNDANCE_RELATIVE_TO_H

from signal import SIGTERM

@task
def aspcap(
    spectra: Iterable[ApogeeCoaddedSpectrumInApStar], 
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    parent_dir: Optional[str] = None, 
    n_threads: Optional[int] = 32,
    max_processes: Optional[int] = 16,
    max_threads: Optional[int] = 128,
    max_concurrent_loading: Optional[int] = 4,
    soft_thread_ratio: Optional[float] = 1,
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

    parent, child = Pipe()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(max_threads, max_processes)) as executor:
        stage_args = [executor, parent, child, parent_dir, max_processes, max_threads, max_concurrent_loading, soft_thread_ratio]
        if live_renderable is not None:
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
            n_threads=n_threads
        )
        abundance_results, abundance_failures = _aspcap_stage("abundances", abundance_plans, *stage_args)

        parent.close()
        child.close()
        executor.shutdown(wait=False, cancel_futures=True)

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
    
    # TODO: Confirm we can access all the spectra from the single `ASPCAP` object.
    spectra_by_pk = {s.spectrum_pk: s for s in spectra}
    for spectrum_pk, kwds in result_kwds.items():
        yield ASPCAP.from_spectrum(spectra_by_pk[spectrum_pk], **kwds)


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
):
    pb = None
    if progress is not None:
        full_names = {
            "coarse": "Coarse parameters",
            "params": "Stellar parameters",
            "abundances": "Chemical abundances"
        }
        stage_task_id = progress.add_task(f"[bold blue]{full_names.get(stage, stage)}[/bold blue]")

    get_task_name = lambda input_nml_path: os.path.dirname(input_nml_path).split("/")[-1]

    # FERRE can be limited by at least three mechanisms:
    # 1. Too many threads requested (CPU limited).
    # 2. Too many processes started (RAM limited).
    # 3. Too many grids load at once (disk I/O limited).
    current_processes, current_threads, currently_loading = (0, 0, 0)
    pre_processed_futures, ferre_futures, post_processed_futures = ([], [], [])
    n_executions, n_executions_total, timings = (0, len(plans), {})

    # TODO: the soft thread ratio is to account for the fact that it takes time to load the grid, and we can load the grid while
    #       we are still thread limited. By the time the grid is loaded, we won't be thread limited anymore.
    #       It's a massive hack that we should revisit.
    at_capacity = lambda p, t, c: (
        p >= max_processes,
        t >= (soft_thread_ratio * max_threads),
        c >= max_concurrent_loading
    )
    max_workers = max(max_processes, max_threads)
    successes, failures = [], {}

    total = 0
    for plan in plans:
        (
            executor
            .submit(pre_process_ferres, plan)
            .add_done_callback(lambda future: pre_processed_futures.insert(0, future))
        )
        total += sum(map(len, (p["spectra"] for p in plan)))
    
    if progress is not None:
        progress.update(stage_task_id, completed=0, total=total)
    else:
        pb = tqdm(total=total, desc=f"ASPCAP {stage}")
        pb.__enter__()

    ferre_tasks = {}

    def check_capacity(current_processes, current_threads, currently_loading):
        worker_limit, thread_limit, loading_limit = at_capacity(current_processes, current_threads, currently_loading)

        if pb is not None:
            pb.set_description(
                f"ASPCAP {stage} ("
                f"thread {current_threads}/{max_threads}{'*' if thread_limit else ''}; "
                f"proc {current_processes}/{max_processes}{'*' if worker_limit else ''}; "
                f"load {currently_loading}/{max_concurrent_loading}{'*' if loading_limit else ''}; "
                f"job {n_executions}/{n_executions_total})"
            )

        while parent.poll():
            input_nml_path, n_obj_update, n_threads_update = parent.recv()
            if n_obj_update == 0:
                currently_loading -= 1
                if progress is not None:
                    progress.update(
                        ferre_tasks[input_nml_path],
                        description=f"  [white]{get_task_name(input_nml_path)}",
                        refresh=True
                    )
            if not input_nml_path.endswith("input_list.nml"):
                current_threads -= n_threads_update
            if pb is not None:
                pb.update(n_obj_update)
            if progress is not None:
                progress.update(ferre_tasks[input_nml_path], advance=n)
                progress.update(stage_task_id, advance=n)
    
        try:
            ferre_future = next(concurrent.futures.as_completed(ferre_futures, timeout=0))
        except:
            None
        else:
            (input_nml_path, return_code, t_overhead, t_elapsed) = ferre_future.result()
            if return_code not in (0, 1):
                log.exception(f"FERRE failed on {input_nml_path}: {return_code}")
            timings[os.path.dirname(input_nml_path)] = (t_overhead, t_elapsed)
            # TODO: Should `timings` and `post_process_ferre` take directories or input_nml_paths?
            post_processed_futures.append(
                executor.submit(
                    post_process_ferre,
                    input_nml_path
                )
            )
            ferre_futures.remove(ferre_future)
            current_processes -= 1
        return (current_processes, current_threads, currently_loading)

    while n_executions_total > n_executions:
        try:
            # Let's oscillate between the first (largest) and last (smallest) elements: (-0 and -1)
            # This means we are distributing the grid loading time while other threads are doing useful things.
            future = pre_processed_futures.pop(-((n_executions + 1) % 2))
        except IndexError:
            continue
        else:
            input_nml_path, total, n_ferre_threads, skipped = future.result()
            
            # Spectra might be skipped because the file could not be found, or if there were too many bad pixels.
            for spectrum, kwds in skipped:
                if progress is not None:
                    progress.update(stage_task_id, advance=1)
                if pb is not None:
                    pb.update(1)
                failures[spectrum.spectrum_pk] = ASPCAP.from_spectrum(spectrum, **kwds)

            while any(at_capacity(current_processes, current_threads, currently_loading)):
                current_processes, current_threads, currently_loading = check_capacity(current_processes, current_threads, currently_loading)

            if total > 0:
                ferre_futures.append(executor.submit(ferre, input_nml_path, total, n_ferre_threads, child))
                if progress is not None:
                    ferre_tasks[input_nml_path] = progress.add_task(
                        f"  [yellow]{get_task_name(input_nml_path)}",
                        total=total
                    )
                currently_loading, current_threads, current_processes = (currently_loading + 1, current_threads + n_ferre_threads, current_processes + 1)
            n_executions += 1

    # All submitted. Now wait for them to finish.
    while current_processes:
        current_processes, current_threads, currently_loading = check_capacity(current_processes, current_threads, currently_loading)
            

    for future in concurrent.futures.as_completed(post_processed_futures):
        for result in future.result():
            # Assign timings to the results.
            try:
                if stage == "abundances":
                    key = os.path.dirname(result["pwd"])
                else:
                    key = result["pwd"]
                t_overhead, t_elapsed_all = timings[key]
                t_elapsed = t_elapsed_all[result["ferre_name"]]
            except:
                t_elapsed = t_overhead = np.nan
            finally:
                result["t_overhead"] = t_overhead
                result["t_elapsed"] = t_elapsed
            successes.append(result)
    
    if progress is not None:
        for task_id in ferre_tasks.values():
            progress.update(task_id, completed=True, visible=False, refresh=True)
    else:
        pb.__exit__(None, None, None)
    return (successes, list(failures.values()))


regex_next_object = re.compile(r"next object #\s+(\d+)")
regex_completed = re.compile(r"\s+(\d+)\s(\d+_[\d\w_]+)") # assumes input ids start with an integer and underscore


def ferre(input_nml_path, n_obj, n_threads, pipe):

    # TODO: list mode!
    is_abundances_mode = input_nml_path.endswith("input_list.nml")

    try:
        stdout, n_complete = ([], 0)
        t_start, t_overhead, t_awaiting, t_elapsed = (time(), None, {}, {})

        command = ["ferre.x"]
        if is_abundances_mode:
            command += ["-l", os.path.basename(input_nml_path)]
        cwd = os.path.dirname(input_nml_path)

        process = subprocess.Popen(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )

        for line in iter(process.stdout.readline, ""):
            if match := regex_next_object.search(line):
                t_awaiting[int(match.group(1))] = -time()
                if t_overhead is None:
                    t_overhead = time() - t_start
                    pipe.send((input_nml_path, 0, 0))
            
            if match := regex_completed.search(line):
                key = match.group(2)
                t_elapsed.setdefault(key, 0)
                t_elapsed[key] += t_awaiting.pop(int(match.group(1))) + time()
                n_complete += 1
                n_remaining = n_obj - n_complete
                n_threads_released = 1 if n_remaining < n_threads else 0
                pipe.send((input_nml_path, 1, n_threads_released))
            stdout.append(line)

        # In case there were some we did not pick up the timings for, or we somehow sent false-positives.                      
        if n_complete != n_obj:
            pipe.send((input_nml_path, n_obj - n_complete, 0))

        with open(os.path.join(cwd, "stdout"), "w") as fp:
            fp.write("".join(stdout))

        process.stdout.close()
        return_code = int(process.wait())
        stderr = process.stderr.read()
        process.stderr.close()
        try:
            process.kill()
            os.killpg(os.getpgid(process.pid), SIGTERM)
        except:
            None

        with open(os.path.join(cwd, "stderr"), "w") as fp:
            fp.write(stderr)

        if return_code not in (0, 1): # FERRE returns return code 1 even if everything is OK.
            log.error(f"FERRE returned code {return_code} in {input_nml_path}:")
            log.error(stderr)
    except:
        print(f"FAILED ON {input_nml_path}")
        raise
    return (input_nml_path, return_code, t_overhead, t_elapsed)




@task
def old_aspcap(
    spectra: Iterable[Spectrum], 
    parent_dir: Optional[str] = None, 
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    operator_kwds: Optional[dict] = None,
    **kwargs
) -> Iterable[ASPCAP]:
    """
    Run the ASPCAP pipeline on some spectra.
    
    .. warning:: 
        This is task for convenience. 
        
        If you want efficiency, you should use the `pre_` and `post_` tasks for each stage in the pipeline.
    
    :param spectra:
        The spectra to analyze with ASPCAP.
    
    :param parent_dir: [optional]
        The parent directory where these FERRE executions will be planned. If `None` is given then this will default
        to a temporary directory in `$MWM_ASTRA/X.Y.Z/pipelines/aspcap/`.
    
    :param initial_guess_callable: [optional]
        A callable that returns an initial guess for the stellar parameters. 
    
    :param header_paths: [optional]
        The path to a file containing the paths to the FERRE header files. This file should contain one path per line.
    
    :param weight_path: [optional]
        The path to the FERRE weight file to use during the coarse and main stellar parameter stage.

    :param element_weight_paths: [optional]
        A path containing FERRE weight files for different elements, which will be used in the chemical abundances stage.

    :param operator_kwds: [optional]
        A dictionary of keywords to supply to the `astra.pipelines.ferre.operator.FerreOperator` class.
    
    Keyword arguments
    -----------------
    All additional keyword arguments will be passed through to `astra.pipelines.ferre.pre_process.pre_process.ferre`. 
    Some handy keywords include:
    continuum_order: int = 4,
    continuum_reject: float = 0.3,
    continuum_observations_flag: int = 1,
    """

    if parent_dir is None:
        dir = expand_path(f"$MWM_ASTRA/{__version__}/pipelines/aspcap/")
        os.makedirs(dir, exist_ok=True)
        parent_dir = mkdtemp(prefix=f"{datetime.now().strftime('%Y-%m-%d')}-", dir=dir)
        os.chmod(parent_dir, 0o755)

    if initial_guess_callable is None:
        initial_guess_callable = get_initial_guesses

    # Convenience without accidentally `flatten()`ing a `ModelSelect`
    if isinstance(spectra, SpectrumMixin):
        spectra = [spectra]

    # Use the list() to make sure this is executed before other stages.
    coarse_stellar_parameter_results = list(
        coarse_stellar_parameters(
            spectra,
            parent_dir=parent_dir,
            initial_guess_callable=initial_guess_callable,
            header_paths=header_paths,
            weight_path=weight_path,
            operator_kwds=operator_kwds,
            **kwargs
        )
    )

    # Here we don't need list() because the stellar parameter results will get processed first
    # in the `create_aspcap_results` function, and then the chemical abundance results.
    # TODO: This might become a bit of a clusterfuck if the FERRE jobs fail. Maybe revisit this.
    stellar_parameter_results = list(stellar_parameters(
        spectra,
        parent_dir=parent_dir,
        weight_path=weight_path,
        operator_kwds=operator_kwds,
        **kwargs
    ))

    chemical_abundance_results = list(abundances(
        spectra,
        parent_dir=parent_dir,
        element_weight_paths=element_weight_paths,
        operator_kwds=operator_kwds,
        #ferre_list_mode=True, # This can be supplied by the user
        **kwargs
    ))
    yield from create_aspcap_results(stellar_parameter_results, chemical_abundance_results)


@task
def post_process_aspcap(parent_dir, **kwargs) -> Iterable[ASPCAP]:
    """
    Run all the post-processing steps for each ASPCAP stage.
    
    :param parent_dir:
        The parent directory where these FERRE executions will be planned. If `None` is given then this will default
        to a temporary directory in `$MWM_ASTRA/X.Y.Z/pipelines/aspcap/`.    
    """

    coarse_results = list(post_coarse_stellar_parameters(parent_dir, **kwargs))
    stellar_parameter_results = list(post_stellar_parameters(parent_dir, **kwargs))
    chemical_abundance_results = list(post_abundances(parent_dir, **kwargs))
    yield from create_aspcap_results(stellar_parameter_results, chemical_abundance_results)



def create_aspcap_results_from_parent_dir(
        parent_dir: str, 
        create_star_products: Optional[bool] = True, 
        overwrite=False,
        limit=None,
        **kwargs
    ):
    
    from astra.products.pipeline import create_star_pipeline_product

    source_pks = []
    for item in _create_aspcap_results_from_parent_dir(parent_dir, **kwargs):
        source_pks.append(item.source_pk)

    # TODO: parallelize this
    source_pks = tuple(set(source_pks))
    if create_star_products:
        log.info(f"Creating aspcapStar products for {len(source_pks)}")
        q = (
            Source
            .select()
            .where(Source.pk.in_(source_pks))
        )
        for source in q:
            create_star_pipeline_product(
                source,
                ASPCAP,
                overwrite=overwrite,
                limit=limit,
            )

    
    



@task
def _create_aspcap_results_from_parent_dir(parent_dir, **kwargs) -> Iterable[ASPCAP]:

    # Get the stellar parameter results first
    parent_dir = expand_path(parent_dir)
    
    stellar_parameter_results = list(
        FerreStellarParameters
        .select()
        .where(FerreStellarParameters.pwd.startswith(f"{parent_dir}/params/"))
    )

    t_coarse = (
        FerreCoarse
        .select(
            FerreCoarse.spectrum_pk,
            fn.sum(FerreCoarse.t_elapsed),        
            fn.sum(FerreCoarse.ferre_time_elapsed)
        )
        .where(FerreCoarse.spectrum_pk.in_([ea.spectrum_pk for ea in stellar_parameter_results]))
        .group_by(FerreCoarse.spectrum_pk)
        .tuples()
    )

    chemical_abundance_results = (
        FerreChemicalAbundances
        .select()
        .where(FerreChemicalAbundances.upstream_pk.in_([ea.task_pk for ea in stellar_parameter_results]))
    )

    t_coarse = { k: v for k, *v in t_coarse }
    data, ferre_time_elapsed, t_elapsed = ({}, {}, {})

    for result in tqdm(stellar_parameter_results, desc="Collecting stellar parameters"):
        data.setdefault(result.task_pk, {})

        t_elapsed.setdefault(result.task_pk, 0)
        t_elapsed[result.task_pk] += (result.t_elapsed or 0) + (t_coarse.get(result.spectrum_pk, [0])[0] or 0)
        ferre_time_elapsed.setdefault(result.task_pk, 0)
        ferre_time_elapsed[result.task_pk] += (result.ferre_time_elapsed or 0) + (t_coarse.get(result.spectrum_pk, [0])[-1] or 0)
        
        v_sini = 10**(result.log10_v_sini or np.nan)
        e_v_sini = (result.e_log10_v_sini or np.nan) * v_sini * np.log(10)
        v_micro = 10**(result.log10_v_micro or np.nan)
        e_v_micro = (result.e_log10_v_micro or np.nan) * v_micro * np.log(10)

        data[result.task_pk].update({
            "source_pk": result.source_pk,
            "spectrum_pk": result.spectrum_pk,
            "tag": result.tag,
            "t_elapsed": t_elapsed[result.task_pk],
            "short_grid_name": result.short_grid_name,
            "teff": result.teff,
            "e_teff": result.e_teff,
            "logg": result.logg,
            "e_logg": result.e_logg,
            "v_micro": v_micro,
            "e_v_micro": e_v_micro,
            "v_sini": v_sini,
            "e_v_sini": e_v_sini,
            "m_h_atm": result.m_h,
            "e_m_h_atm": result.e_m_h,
            "alpha_m_atm": result.alpha_m,
            "e_alpha_m_atm": result.e_alpha_m,
            "c_m_atm": result.c_m,
            "e_c_m_atm": result.e_c_m,
            "n_m_atm": result.n_m,
            "e_n_m_atm": result.e_n_m,

            "initial_flags": result.initial_flags,
            "continuum_order": result.continuum_order,
            "continuum_reject": result.continuum_reject,
            "interpolation_order": result.interpolation_order,

            "snr": result.snr,
            "rchi2": result.rchi2,
            "result_flags": result.ferre_flags,
            "ferre_log_snr_sq": result.ferre_log_snr_sq,     
            "ferre_time_elapsed": ferre_time_elapsed[result.task_pk],
            "stellar_parameters_task_pk": result.task_pk,

            "raw_teff": result.teff,
            "raw_e_teff": result.e_teff,
            "raw_logg": result.logg,
            "raw_e_logg": result.e_logg,
            "raw_v_micro": v_micro,
            "raw_e_v_micro": e_v_micro,
            "raw_v_sini": v_sini,
            "raw_e_v_sini": e_v_sini,
            "raw_m_h_atm": result.m_h,
            "raw_e_m_h_atm": result.e_m_h,
            "raw_alpha_m_atm": result.alpha_m,
            "raw_e_alpha_m_atm": result.e_alpha_m,
            "raw_c_m_atm": result.c_m,
            "raw_e_c_m_atm": result.e_c_m,
            "raw_n_m_atm": result.n_m,
            "raw_e_n_m_atm": result.e_n_m,
        })


    skipped_results = {}
    for result in tqdm(chemical_abundance_results, total=0, desc="Collecting abundances"):
        ferre_time_elapsed.setdefault(result.upstream_pk, 0)
        ferre_time_elapsed[result.upstream_pk] += (result.ferre_time_elapsed or 0)

        if result.upstream_pk not in data:
            skipped_results.setdefault(result.upstream_pk, [])
            skipped_results[result.upstream_pk].append(result)
            continue

        data[result.upstream_pk].update(**_result_to_kwds(result, data[result.upstream_pk]))        

    if len(skipped_results) > 0:
        # There shouldn't be too many of these, so let's just save them.
        log.warn(f"There were {len(skipped_results)} chemical abundance results that might updating to existing ASPCAP records. Doing it now..")
        '''
        q = (
            ASPCAP
            .select()
            .where(ASPCAP.stellar_parameters_task_pk.in_(list(skipped_results.keys())))
        )
        chemical_abundance_task_pk_names = [f for f in ASPCAP._meta.fields.keys() if f.endswith("_task_pk") and f != "stellar_parameters_task_pk"]
        
        for result in q:
            any_update_made = False
            chemical_abundance_task_pks = dict(zip(chemical_abundance_task_pk_names, [getattr(result, f) for f in chemical_abundance_task_pk_names]))            
            for chemical_abundance_result in skipped_results[result.stellar_parameters_task_pk]:
                if chemical_abundance_result.task_pk in chemical_abundance_task_pks.values(): 
                    continue
                
                any_update_made = True
                for k, v in _result_to_kwds(chemical_abundance_result, result.__data__).items():
                    setattr(result, k, v)

            if any_update_made:
                log.info(f"Saving result {result}")                
                result.save()
        '''
        
    for stellar_parameter_task_pk, kwds in data.items():
        yield ASPCAP(**kwds)
    

'''
@task
def create_aspcap_results(
    stellar_parameter_results: Optional[Iterable[FerreStellarParameters]] = (
        FerreStellarParameters
        .select()
        .distinct(FerreStellarParameters.spectrum_pk)
        .join(ASPCAP, JOIN.LEFT_OUTER, on=(ASPCAP.stellar_parameters_task_pk == FerreStellarParameters.task_pk))
        .where(ASPCAP.stellar_parameters_task_pk.is_null())
    ), 
    chemical_abundance_results: Optional[Iterable[FerreChemicalAbundances]] = (
        FerreChemicalAbundances
        .select()
    ), 
    **kwargs
) -> Iterable[ASPCAP]:
    """
    Create ASPCAP results based on the results from the stellar parameter stage, and the chemical abundances stage.

    These result iterables are linked through the `FerreChemicalAbundances.upstream_pk` being equal to the
    `FerreStellarParameters.task_pk` attributes. One ASPCAP result will be created for each stellar parameter result,
    even if there are no abundances available for that stellar parameter result.

    :param stellar_parameter_results:
        An iterable of `FerreStellarParameters`.
    
    :param chemical_abundance_results:
        An iterable of `FerreChemicalAbundances`
    """

    t_coarse = (
        FerreCoarse
        .select(
            FerreCoarse.spectrum_pk,
            fn.sum(FerreCoarse.t_elapsed),        
            fn.sum(FerreCoarse.ferre_time_elapsed)
        )
        .where(FerreCoarse.spectrum_pk.in_([ea.spectrum_pk for ea in stellar_parameter_results]))
        .group_by(FerreCoarse.spectrum_pk)
        .tuples()
    )
    t_coarse = { k: v for k, *v in t_coarse }

    data, ferre_time_elapsed, t_elapsed = ({}, {}, {})

    for result in tqdm(stellar_parameter_results, desc="Collecting stellar parameters"):
        data.setdefault(result.task_pk, {})

        t_elapsed.setdefault(result.task_pk, 0)
        t_elapsed[result.task_pk] += (result.t_elapsed or 0) + (t_coarse.get(result.spectrum_pk, [0])[0] or 0)
        ferre_time_elapsed.setdefault(result.task_pk, 0)
        ferre_time_elapsed[result.task_pk] += (result.ferre_time_elapsed or 0) + (t_coarse.get(result.spectrum_pk, [0])[-1] or 0)
        
        v_sini = 10**(result.log10_v_sini or np.nan)
        e_v_sini = (result.e_log10_v_sini or np.nan) * v_sini * np.log(10)
        v_micro = 10**(result.log10_v_micro or np.nan)
        e_v_micro = (result.e_log10_v_micro or np.nan) * v_micro * np.log(10)

        data[result.task_pk].update({
            "source_pk": result.source_pk,
            "spectrum_pk": result.spectrum_pk,
            "tag": result.tag,
            "t_elapsed": t_elapsed[result.task_pk],
            "short_grid_name": result.short_grid_name,
            "teff": result.teff,
            "e_teff": result.e_teff,
            "logg": result.logg,
            "e_logg": result.e_logg,
            "v_micro": v_micro,
            "e_v_micro": e_v_micro,
            "v_sini": v_sini,
            "e_v_sini": e_v_sini,
            "m_h_atm": result.m_h,
            "e_m_h_atm": result.e_m_h,
            "alpha_m_atm": result.alpha_m,
            "e_alpha_m_atm": result.e_alpha_m,
            "c_m_atm": result.c_m,
            "e_c_m_atm": result.e_c_m,
            "n_m_atm": result.n_m,
            "e_n_m_atm": result.e_n_m,

            "initial_flags": result.initial_flags,
            "continuum_order": result.continuum_order,
            "continuum_reject": result.continuum_reject,
            "interpolation_order": result.interpolation_order,

            "snr": result.snr,
            "rchi2": result.rchi2,
            "result_flags": result.ferre_flags,
            "ferre_log_snr_sq": result.ferre_log_snr_sq,     
            "ferre_time_elapsed": ferre_time_elapsed[result.task_pk],
            "stellar_parameters_task_pk": result.task_pk,

            "raw_teff": result.teff,
            "raw_e_teff": result.e_teff,
            "raw_logg": result.logg,
            "raw_e_logg": result.e_logg,
            "raw_v_micro": v_micro,
            "raw_e_v_micro": e_v_micro,
            "raw_v_sini": v_sini,
            "raw_e_v_sini": e_v_sini,
            "raw_m_h_atm": result.m_h,
            "raw_e_m_h_atm": result.e_m_h,
            "raw_alpha_m_atm": result.alpha_m,
            "raw_e_alpha_m_atm": result.e_alpha_m,
            "raw_c_m_atm": result.c_m,
            "raw_e_c_m_atm": result.e_c_m,
            "raw_n_m_atm": result.n_m,
            "raw_e_n_m_atm": result.e_n_m,
        })


    skipped_results = {}
    for result in tqdm(chemical_abundance_results, desc="Collecting abundances"):
        ferre_time_elapsed.setdefault(result.upstream_pk, 0)
        ferre_time_elapsed[result.upstream_pk] += (result.ferre_time_elapsed or 0)

        if result.upstream_pk not in data:
            skipped_results.setdefault(result.upstream_pk, [])
            skipped_results[result.upstream_pk].append(result)
            continue

        data[result.upstream_pk].update(**_result_to_kwds(result, data[result.upstream_pk]))        

    if len(skipped_results) > 0:
        # There shouldn't be too many of these, so let's just save them.
        log.warn(f"There were {len(skipped_results)} chemical abundance results that might updating to existing ASPCAP records. Doing it now..")
        """
        q = (
            ASPCAP
            .select()
            .where(ASPCAP.stellar_parameters_task_pk.in_(list(skipped_results.keys())))
        )
        chemical_abundance_task_pk_names = [f for f in ASPCAP._meta.fields.keys() if f.endswith("_task_pk") and f != "stellar_parameters_task_pk"]
        
        for result in q:
            any_update_made = False
            chemical_abundance_task_pks = dict(zip(chemical_abundance_task_pk_names, [getattr(result, f) for f in chemical_abundance_task_pk_names]))            
            for chemical_abundance_result in skipped_results[result.stellar_parameters_task_pk]:
                if chemical_abundance_result.task_pk in chemical_abundance_task_pks.values(): 
                    continue
                
                any_update_made = True
                for k, v in _result_to_kwds(chemical_abundance_result, result.__data__).items():
                    setattr(result, k, v)

            if any_update_made:
                log.info(f"Saving result {result}")                
                result.save()
        """
        
    for stellar_parameter_task_pk, kwds in data.items():
        yield ASPCAP(**kwds)
    
    
def _result_to_kwds(result, existing_kwds):

    #data.setdefault(result.upstream_pk, {})
    species = get_species(result.weight_path)
    
    if species.lower() == "c_12_13":
        label = species.lower()
    else:
        label = f"{species.lower()}_h"

    for key in ("m_h", "alpha_m", "c_m", "n_m"):
        if not getattr(result, f"flag_{key}_frozen"):
            break
    else:
        raise ValueError(f"Can't figure out which label to use")
    
    value = getattr(result, key)
    e_value = getattr(result, f"e_{key}")

    if not ABUNDANCE_RELATIVE_TO_H[species] and value is not None:
        # [X/M] = [X/H] - [M/H]
        # [X/H] = [X/M] + [M/H]                
        value += existing_kwds["m_h_atm"]
        e_value = np.sqrt(e_value**2 + existing_kwds["e_m_h_atm"]**2)
        
    new_kwds = {
        f"{label}_task_pk": result.task_pk,
        f"{label}_rchi2": result.rchi2,
        f"{label}": value,
        f"e_{label}": e_value,
        f"raw_{label}": value,
        f"raw_e_{label}": e_value,
        f"{label}_flags": result.ferre_flags,
        
    }
    if hasattr(result, f"{key}_flags"):
        new_kwds[f"{label}_flags"] = getattr(result, f"{key}_flags")
        
    return new_kwds
'''