import os
import numpy as np
import concurrent.futures
from typing import Iterable, Optional
from astra.utils import log, list_to_dict
from astra.models.aspcap import FerreCoarse
from astra.pipelines.aspcap.coarse import penalize_coarse_stellar_parameter_result
from astra.pipelines.ferre.utils import parse_header_path
from astra.pipelines.aspcap.continuum import MedianFilter

STAGE = "params"

def _pre_compute_continuum(coarse_result, spectrum, pre_continuum):
    try:
        # Apply continuum normalization.
        pre_computed_continuum = pre_continuum.fit(spectrum, coarse_result)
    except:
        log.exception(f"Exception when computing continuum for spectrum {spectrum} from coarse result {coarse_result}:")
        return (spectrum.spectrum_pk, None)
    else:
        return (spectrum.spectrum_pk, pre_computed_continuum)


def plan_stellar_parameters_stage(spectra, parent_dir, coarse_results, weight_path, pre_continuum=MedianFilter, **kwargs):
        
    best_coarse_results = {}
    for kwds in coarse_results:
        this = FerreCoarse(**kwds)
        # TODO: Make the penalized rchi2 a property of the FerreCoarse class.
        this.penalized_rchi2 = penalize_coarse_stellar_parameter_result(this)

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
            else:
                best.ferre_time_coarse = this.t_elapsed + existing.t_elapsed
            
        finally:            
            best_coarse_results[this.spectrum_pk] = best


    spectra_dict = { s.spectrum_pk: s for s in spectra }

    if pre_continuum is None:
        pre_computed_continuum = { s.spectrum_pk: 1 for s in spectra }
    else:            
        fun = pre_continuum()

        futures = []
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as executor:
            for r in best_coarse_results.values():
                spectrum = spectra_dict[r.spectrum_pk]
                futures.append(executor.submit(_pre_compute_continuum, r, spectrum, fun))

        pre_computed_continuum = {}
        #with tqdm(total=len(futures), desc="Pre-computing continuum") as pb:
        for future in concurrent.futures.as_completed(futures):
            spectrum_pk, continuum = future.result()
            pre_computed_continuum[spectrum_pk] = continuum

    # Plan the next stage
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

    return (stellar_parameter_plans, best_coarse_results)