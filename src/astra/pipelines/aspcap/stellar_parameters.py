import os
import numpy as np
import concurrent.futures
from typing import Iterable, Optional
from astra.utils import log, list_to_dict
from astra.models.aspcap import FerreCoarse
from astra.models.aspcap import ASPCAP
from astra.pipelines.aspcap.coarse import penalize_coarse_stellar_parameter_result
from astra.pipelines.aspcap.debugger import debugger
from astra.pipelines.ferre.utils import parse_header_path
from astra.pipelines.aspcap.continuum import MedianFilter

STAGE = "params"

def _pre_compute_continuum(coarse_result, spectrum, pre_continuum):
    try:
        # Apply continuum normalization.
        pre_computed_continuum = pre_continuum.fit(spectrum, coarse_result)
    except:
        log.exception(f"Exception when computing continuum for spectrum {spectrum} from coarse result {coarse_result}:")
        return (spectrum.spectrum_pk, 1)
    else:
        return (spectrum.spectrum_pk, pre_computed_continuum)


def plan_stellar_parameters_stage(spectra, parent_dir, coarse_results, weight_path, pre_continuum=MedianFilter, **kwargs):

    debugger(f"plan_stellar_parameters_stage: entry n_coarse_results={len(coarse_results)} n_spectra={len(spectra)} pre_continuum={pre_continuum}")

    best_coarse_results = {}
    # Total FERRE time across every coarse grid tried for a spectrum, accumulated
    # separately from the best-result selection. Doing it inline only credited the two
    # most recent grids, and only for spectra that had more than one coarse result at
    # all -- a spectrum fit in a single grid never got a time.
    total_t_elapsed_coarse = {}
    for kwds in coarse_results:
        this = FerreCoarse(**kwds)
        # TODO: Make the penalized rchi2 a property of the FerreCoarse class.
        this.penalized_rchi2 = penalize_coarse_stellar_parameter_result(this)

        # `t_elapsed` is None when no timing was recovered for this execution. Test for
        # None explicitly: a truthiness test reads a legitimate 0.0 as missing.
        t_elapsed = np.nan if this.t_elapsed is None else this.t_elapsed
        total_t_elapsed_coarse[this.spectrum_pk] = (
            total_t_elapsed_coarse.get(this.spectrum_pk, 0.0) + t_elapsed
        )

        best = None
        try:
            existing = best_coarse_results[this.spectrum_pk]
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

            if best is None:
                log.error(f"Error for {kwds} - best is None. {existing} {existing.penalized_rchi2} {this} {this.penalized_rchi2}")

            if not np.isfinite(this.penalized_rchi2):
                best.flag_affected_by_timeout = True

        finally:
            best_coarse_results[this.spectrum_pk] = best

    # Credit the whole coarse-stage cost to whichever result was chosen.
    for spectrum_pk, best in best_coarse_results.items():
        if best is not None:
            best.ferre_time_coarse = total_t_elapsed_coarse[spectrum_pk]

    debugger(f"plan_stellar_parameters_stage: built best_coarse_results n={len(best_coarse_results)}")

    spectra_dict = { s.spectrum_pk: s for s in spectra }

    #no_good_result = set(spectra_dict.keys()).difference(best_coarse_results.keys())
    #coarse_failures = [ASPCAP.from_spectrum(spectra_dict[spectrum_pk], flag_no_good_coarse_result=True) for spectrum_pk in no_good_result]

    if pre_continuum is None:
        pre_computed_continuum = { s.spectrum_pk: 1 for s in spectra }
    else:
        fun = pre_continuum()
        debugger(f"plan_stellar_parameters_stage: instantiated pre_continuum; submitting {len(best_coarse_results)} continuum futures with cpu_count={os.cpu_count()}")

        futures = []
        with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
            for r in best_coarse_results.values():
                spectrum = spectra_dict[r.spectrum_pk]
                futures.append(executor.submit(_pre_compute_continuum, r, spectrum, fun))
            debugger(f"plan_stellar_parameters_stage: all {len(futures)} continuum futures submitted; waiting for executor shutdown")

        debugger(f"plan_stellar_parameters_stage: ThreadPoolExecutor closed; collecting results")
        pre_computed_continuum = {}
        #with tqdm(total=len(futures), desc="Pre-computing continuum") as pb:
        for future in concurrent.futures.as_completed(futures):
            spectrum_pk, continuum = future.result()
            pre_computed_continuum[spectrum_pk] = continuum
        debugger(f"plan_stellar_parameters_stage: pre_computed_continuum collected n={len(pre_computed_continuum)}")

    # Plan the next stage
    debugger(f"plan_stellar_parameters_stage: building group_task_kwds")
    group_task_kwds = {}
    for r in best_coarse_results.values():
        group_task_kwds.setdefault(r.header_path, [])
        spectrum = spectra_dict[r.spectrum_pk]

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
            pwd=f"{parent_dir}/{STAGE}/{short_grid_name}",
            **kwargs
        )
        stellar_parameter_plans.append([group_task_kwds[header_path]])

    debugger(f"plan_stellar_parameters_stage: returning {len(stellar_parameter_plans)} plans, {len(best_coarse_results)} best coarse results")
    return (stellar_parameter_plans, best_coarse_results)
