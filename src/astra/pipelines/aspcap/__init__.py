import os
import numpy as np
import concurrent.futures
import subprocess
import re
import heapq
import thread
from multiprocessing import Pipe, Lock
from datetime import datetime
from tempfile import mkdtemp
from typing import Optional, Iterable, List, Tuple, Callable, Union
from peewee import JOIN, fn
from tqdm import tqdm
from time import time, sleep
from threading import Timer

from astra import __version__, task
from astra.utils import log, expand_path, list_to_dict
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
from astra.models.aspcap import ASPCAP, FerreCoarse, FerreStellarParameters, FerreChemicalAbundances, Source
from astra.models.spectrum import Spectrum, SpectrumMixin
from astra.pipelines.ferre.pre_process import pre_process_ferre
from astra.pipelines.ferre.post_process import post_process_ferre
from astra.pipelines.ferre.utils import  parse_header_path
from astra.pipelines.aspcap.initial import get_initial_guesses
from astra.pipelines.aspcap.coarse import coarse_stellar_parameters, post_coarse_stellar_parameters, plan_coarse_stellar_parameters, penalize_coarse_stellar_parameter_result
from astra.pipelines.aspcap.continuum import MedianFilter

#from astra.pipelines.aspcap.stellar_parameters import stellar_parameters, post_stellar_parameters
#from astra.pipelines.aspcap.abundances import abundances, get_species, post_abundances
#from astra.pipelines.aspcap.utils import ABUNDANCE_RELATIVE_TO_H


import signal
from threading import Event, Lock, Thread
from subprocess import Popen, PIPE, STDOUT


@task
def aspcap(
    spectra: Iterable[ApogeeCoaddedSpectrumInApStar], 
    initial_guess_callable: Optional[Callable] = None,
    header_paths: Optional[Union[List[str], Tuple[str], str]] = "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list",
    weight_path: Optional[str] = "$MWM_ASTRA/pipelines/aspcap/masks/global.mask",
    element_weight_paths: str = "$MWM_ASTRA/pipelines/aspcap/masks/elements.list",
    parent_dir: Optional[str] = None, 
    max_processes: Optional[int] = 16,
    max_threads: Optional[int] = 128,
    max_concurrent_loading: Optional[int] = 4,
    soft_thread_ratio: Optional[float] = 2,
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

    plans, spectra_with_no_initial_guess = plan_coarse_stellar_parameters(
        spectra=spectra, 
        header_paths=header_paths, 
        initial_guess_callable=initial_guess_callable,
        weight_path=weight_path
    )

    for spectrum in spectra_with_no_initial_guess:
        yield ASPCAP.from_spectrum(spectrum, flag_no_suitable_initial_guess=True)

    coarse_results = _aspcap_stage("coarse", plans, parent_dir, max_processes, max_threads, max_concurrent_loading, soft_thread_ratio)

    STAGE = "params"
    # Plan stellar parameter stage.
    coarse_results_by_spectrum = {}
    for kwds in coarse_results:
        this = FerreCoarse(**kwds)
        # TODO: Make the penalized rchi2 a property of the FerreCoarse class.
        this.penalized_rchi2 = penalize_coarse_stellar_parameter_result(this)

        best = None
        try:
            existing = coarse_results_by_spectrum[this.spectrum_pk]
        except KeyError:
            best = this
        else:
            if this.penalized_rchi2 < existing.penalized_rchi2:
                best = this
            elif this.penalized_rchi2 > existing.penalized_rchi2:
                best = existing
            elif this.penalized_rchi2 == existing.penalized_rchi2:
                best = existing
                best.flag_multiple_equally_good_coarse_results = True            
        finally:
            coarse_results_by_spectrum[this.spectrum_pk] = best

    spectra_by_pk = {s.spectrum_pk: s for s in spectra}

    pre_continuum = MedianFilter()

    futures = []
    with concurrent.futures.ProcessPoolExecutor(128) as executor:
        for r in tqdm(coarse_results_by_spectrum.values(), desc="Distributing work"):
            spectrum = spectra_by_pk[r.spectrum_pk]
            futures.append(executor.submit(_pre_compute_continuum, r, spectrum, pre_continuum))

    pre_computed_continuum = {}
    with tqdm(total=len(futures), desc="Pre-computing continuum") as pb:
        for future in concurrent.futures.as_completed(futures):
            spectrum_pk, continuum = future.result()
            pre_computed_continuum[spectrum_pk] = continuum
            pb.update()        

    group_task_kwds = {}
    for r in tqdm(coarse_results_by_spectrum.values(), desc="Grouping results"):
        group_task_kwds.setdefault(r.header_path, [])
        spectrum = spectra_by_pk[r.spectrum_pk]

        group_task_kwds[r.header_path].append(
            dict(
                spectra=spectrum,
                pre_computed_continuum=pre_computed_continuum[r.spectrum_pk],
                initial_teff=r.teff,
                initial_logg=r.logg,
                initial_m_h=r.m_h,
                initial_log10_v_sini=r.log10_v_sini,
                initial_log10_v_micro=r.log10_v_micro,
                initial_alpha_m=r.alpha_m,
                initial_c_m=r.c_m,
                initial_n_m=r.n_m,
                initial_flags=r.initial_flags,                
                upstream_pk=r.task_pk,
            )
        )

    stellar_parameter_plans = []
    for header_path in group_task_kwds.keys():
        short_grid_name = parse_header_path(header_path)["short_grid_name"]
        group_task_kwds[header_path] = list_to_dict(group_task_kwds[header_path])
        group_task_kwds[header_path].update(
            header_path=header_path,
            weight_path=weight_path,
            relative_dir=f"{STAGE}/{short_grid_name}",
            **kwargs
        )
        stellar_parameter_plans.append(group_task_kwds[header_path])        

    stellar_parameter_results = _aspcap_stage("params", stellar_parameter_plans, parent_dir, max_processes, max_threads, max_concurrent_loading, soft_thread_ratio)
    
    # yeet back some ASPCAP results
    

    raise a

    """
    for kwds in post_process_ferre(pwd):
        result = FerreCoarse(**kwds)
        penalize_coarse_stellar_parameter_result(result)
        yield result
    """

# TODO: remove from coarse.py

def _pre_compute_continuum(coarse_result, spectrum, pre_continuum):
    try:
        # Apply continuum normalization.
        pre_computed_continuum = pre_continuum.fit(spectrum, coarse_result)
    except:
        log.exception(f"Exception when computing continuum for spectrum {spectrum} from coarse result {coarse_result}:")
        return (spectrum.spectrum_pk, None)
    else:
        return (spectrum.spectrum_pk, pre_computed_continuum)




def _aspcap_stage(stage, plans, parent_dir, max_processes, max_threads, max_concurrent_loading, soft_thread_ratio):
    # FERRE can be limited by at least three mechanisms:
    # 1. Too many threads requested (CPU limited).
    # 2. Too many processes started (RAM limited).
    # 3. Too many grids load at once (disk I/O limited).
    parent, child = Pipe()
    current_processes, current_threads, currently_loading = (0, 0, 0)
    pre_processed_futures, ferre_futures, post_processed_futures = ([], [], [])
    n_executions, n_executions_total, timings, t_full = (0, len(plans), {}, -time())

    # TODO: the soft thread ratio is to account for the fact that it takes time to load the grid, and we can load the grid while
    #       we are still thread limited. By the time the grid is loaded, we won't be thread limited anymore.
    #       It's a massive hack that we should revisit.
    at_capacity = lambda p, t, c: (
        p >= max_processes,
        t >= (soft_thread_ratio * max_threads),
        c >= max_concurrent_loading
    )

    max_workers = max(max_processes, max_threads)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

        for plan in plans:
            # Remove any output files files in the expected directory.
            pwd = os.path.join(parent_dir, plan.pop("relative_dir"))
            os.system(f"rm -f {pwd}/*.output {pwd}/stdout {pwd}/stderr")
            (
                executor
                .submit(pre_process_ferre, pwd=pwd, **plan)
                .add_done_callback(lambda future: pre_processed_futures.insert(0, future))
            )

        with tqdm(total=sum(len(plan["spectra"]) for plan in plans)) as pb:
            
            def check_capacity(current_processes, current_threads, currently_loading):
                worker_limit, thread_limit, loading_limit = at_capacity(current_processes, current_threads, currently_loading)
                pb.set_description(
                    f"ASPCAP {stage} ("
                    f"thread {current_threads}/{max_threads}{'*' if thread_limit else ''}; "
                    f"proc {current_processes}/{max_processes}{'*' if worker_limit else ''}; "
                    f"load {currently_loading}/{max_concurrent_loading}{'*' if loading_limit else ''}; "
                    f"job {n_executions}/{n_executions_total})"
                )

                while parent.poll():
                    child_directory, n = parent.recv()
                    if n == 0:
                        currently_loading -= 1
                    current_threads -= n
                    pb.update(n)
            
                try:
                    ferre_future = next(concurrent.futures.as_completed(ferre_futures, timeout=0))
                except TimeoutError:
                    None
                else:
                    (completed_directory, return_code, t_overhead, t_elapsed) = ferre_future.result()
                    if return_code not in (0, 1):
                        log.exception(f"FERRE failed on {completed_directory}: {return_code}")
                    timings[completed_directory] = (t_overhead, t_elapsed)
                    post_processed_futures.append(
                        executor.submit(
                            post_process_ferre,
                            completed_directory
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
                    directory, n_obj, skipped = future.result()

                    # Spectra might be skipped because the file could not be found, or if there were too many bad pixels.
                    for spectrum, kwds in skipped:
                        pb.update(1)
                        #yield ASPCAP.from_spectrum(spectrum, **kwds)

                    while any(at_capacity(current_processes, current_threads, currently_loading)):
                        current_processes, current_threads, currently_loading = check_capacity(current_processes, current_threads, currently_loading)
                    ferre_futures.append(executor.submit(ferre, directory, n_obj, child))
                    n_executions, currently_loading, current_threads, current_processes = (n_executions + 1, currently_loading + 1, current_threads + n_obj, current_processes + 1)

            # All submitted. Now wait for them to finish.
            while current_processes:
                current_processes, current_threads, currently_loading = check_capacity(current_processes, current_threads, currently_loading)

        t_full += time()
        print(f"FUll time: {t_full}")

                
        parent.close()
        child.close()
        print("closed parent/child")
    
        results = []
        for future in concurrent.futures.as_completed(post_processed_futures):
            for result in future.result():
                # Assign timings to the results.
                try:
                    t_overhead, t_elapsed_all = timings[result["pwd"]]
                    t_elapsed = t_elapsed_all[result["ferre_name"]]
                except:
                    t_elapsed = t_overhead = np.nan
                finally:
                    result["t_overhead"] = t_overhead
                    result["t_elapsed"] = t_elapsed
                results.append(result)

        print("closing ferre executor")
        executor.shutdown(wait=False, cancel_futures=True)
        print("closed ferre executor")
    
    return results


regex_next_object = re.compile(r"next object #\s+(\d+)")
regex_completed = re.compile(r"\s+(\d+)\s(\d+_[\d\w_]+)") # assumes input ids start with an integer and underscore
 


def ferre(directory, n_obj, pipe, timeout_line=10, timeout_grid_load=60):
    rid = os.getpid()
    #print(f"FERRE starting {rid} {directory}")
    try:
        stdout, n_complete = ([], 0)
        t_start, t_overhead, t_elapsed = (time(), None, {})

        process = subprocess.Popen(["ferre.x"], cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        def callback(*args):
            print(f"callback hit: {args}")
            raise KeyboardInterrupt

        #watchdog = WatchdogTimer(timeout_line, callback=callback, daemon=True)
        #watchdog.start()

        for line in iter(process.stdout.readline, ""):
            #except KeyboardInterrupt:
            #    # watchdog hit.
            #    print(f"{rid} {directory} watchdog. time_elapsed: {time() - t_start:.0f} s, n_complete: {n_complete}/{n_obj} (t_overhead: {t_overhead})")
            #else:
            #with watchdog.blocked:
            # if an object is completed, note the completion time and send back some information to the parent.
            if match := regex_next_object.search(line):
                t_elapsed[int(match.group(1))] = -time()
                if t_overhead is None:
                    t_overhead = time() - t_start
                    pipe.send((directory, 0))
            
            if match := regex_completed.search(line):
                t_elapsed[match.group(2)] = t_elapsed.pop(int(match.group(1))) + time()
                n_complete += 1
                pipe.send((directory, 1))
            stdout.append(line)
        #finally:
        #    watchdog.restart()
            
        """
        for line in iter(process.stdout.readline, ""):

            # if an object is completed, note the completion time and send back some information to the parent.
            if match := regex_next_object.search(line):
                t_elapsed[int(match.group(1))] = -time()
                if t_overhead is None:
                    t_overhead = time() - t_start
                    child.send((directory, 0))
            
            if match := regex_completed.search(line):
                t_elapsed[match.group(2)] = t_elapsed.pop(int(match.group(1))) + time()
                n_complete += 1
                child.send((directory, 1))
            stdout.append(line)

            print(f"{rid} still waiting in {directory} after {timeout_line}")
            # Check which objects we are waiting on, and how long we have been waiting.
            t_since_start = time() - t_start
            is_grid_loaded = t_overhead is not None
            if (t_overhead is None) and (time() - t_start) > timeout_grid_load:
                print(f"{rid} timeout on grid load: {t_since_start:.0f} s so far")
            
            currently_running = {k: (time() + v) for k, v in t_elapsed.items() if v < 0}
            if len(currently_running) > 0:
                print(f"{rid} currently running {len(currently_running)} objects:")
                for k, v in currently_running.items():
                    print(f"    {rid}  {k}: {v:.0f} s")
        """


        #finally:
        #    timeout.cancel()

        # In case there were some we did not pick up the timings for, or we somehow sent false-positives.                      
        if n_complete != n_obj:
            child.send((directory, n_obj - n_complete))

        with open(os.path.join(directory, "stdout"), "w") as fp:
            fp.write("".join(stdout))

        process.stdout.close()
        return_code = int(process.wait())
        stderr = process.stderr.read()
        process.stderr.close()
        try:
            process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            None

        if return_code not in (0, 1): # FERRE returns return code 1 even if everything is OK.
            log.error(f"FERRE returned code {return_code} in {directory}:")
            log.error(stderr)
            with open(os.path.join(directory, "stderr"), "w") as fp:
                fp.write(stderr)
    except:
        print(f"FAILED ON {directory}")
        raise
    return (directory, return_code, t_overhead, t_elapsed)




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