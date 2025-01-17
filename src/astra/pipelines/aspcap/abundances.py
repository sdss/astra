import os
import numpy as np
from typing import Iterable, Optional
from astra.utils import expand_path, log, list_to_dict
from astra.pipelines.ferre.utils import (get_apogee_pixel_mask, parse_ferre_spectrum_name, read_ferre_headers, parse_header_path)
from astra.pipelines.aspcap.utils import get_abundance_keywords

STAGE = "abundances"

# NOTE to Future Andy:
# ferre often (but not always; ie nthreads>1) needs the NOBJ keyword to be set for doing abundances in list (-l) mode 
 
def plan_abundances_stage(
    spectra,
    parent_dir: str,
    stellar_parameter_results, 
    element_weight_paths: str,
    use_ferre_list_mode: Optional[bool] = False,
    continuum_order: Optional[int] = -1,
    continuum_flag: Optional[int] = 0,
    continuum_observations_flag: Optional[int] = 0,
    **kwargs,
):
    """
    Plan abundance executions with FERRE for some given spectra.
    
    In the abundances stage we keep the continuum fixed to what was found from the stellar parameter stage. That's why the
    defaults are set for `continuum_order`, `continuum_flag`, and `continuum_observations_flag`.
    """

    with open(expand_path(element_weight_paths), "r") as fp:
        weight_paths = list(map(str.strip, fp.readlines()))
 
    # Load abundance keywords on demand.
    ferre_headers, abundance_keywords = ({}, {})
    lookup_spectrum_by_primary_key = { s.spectrum_pk: s for s in spectra }
    
    mask = get_apogee_pixel_mask()
    continuum_cache, continuum_cache_names = ({}, {})

    group_task_kwds, pre_computed_continuum = ({}, {})
    for result in stellar_parameter_results:

        if result["short_grid_name"].find("combo5_BA") > 0:
            # Not doing abundances for BA_lsfcombo5 grids
            continue

        group_task_kwds.setdefault(result["header_path"], [])
        if result["header_path"] not in abundance_keywords:
            abundance_keywords[result["header_path"]] = {}
            try:
                headers, *segment_headers = ferre_headers[result["header_path"]]
            except KeyError:
                headers, *segment_headers = ferre_headers[result["header_path"]] = read_ferre_headers(result["header_path"])

            for weight_path in weight_paths:
                species = get_species(weight_path)
                frozen_parameters, ferre_kwds = get_abundance_keywords(species, headers["LABEL"])
                abundance_keywords[result["header_path"]][species] = (weight_path, frozen_parameters, ferre_kwds)
        
        spectrum = lookup_spectrum_by_primary_key[result["spectrum_pk"]]

        prefix = f"{result['pwd']}"#/params/{result['short_grid_name']}"

        try:
            continuum_cache[prefix]
        except:
            P = 7514
            rectified_model_flux = np.atleast_2d(np.loadtxt(f"{prefix}/rectified_model_flux.output", usecols=range(1, 1+P)))
            model_flux = np.atleast_2d(np.loadtxt(f"{prefix}/model_flux.output", usecols=range(1, 1+P)))
            rectified_flux = np.atleast_2d(np.loadtxt(f"{prefix}/rectified_flux.output", usecols=range(1, 1+P)))
            ferre_flux = np.atleast_2d(np.loadtxt(f"{prefix}/flux.input", usecols=range(P)))

            continuum = (rectified_model_flux/model_flux) / (rectified_flux/ferre_flux)
            continuum_cache[prefix] = np.nan * np.ones((continuum.shape[0], 8575))
            continuum_cache[prefix][:, mask] = continuum

            # Check names
            # TODO: This is a sanity check. if it is expensive, we can remove it later.
            continuum_cache_names[prefix] = [
                np.atleast_1d(np.loadtxt(f"{prefix}/model_flux.output", usecols=(0, ), dtype=str)),
                np.atleast_1d(np.loadtxt(f"{prefix}/rectified_flux.output", usecols=(0, ), dtype=str)),
                np.atleast_1d(np.loadtxt(f"{prefix}/rectified_model_flux.output", usecols=(0, ), dtype=str)),
            ]    

        finally:
            pre_computed_continuum[result["spectrum_pk"]] = continuum_cache[prefix][int(result["ferre_index"])]
            for each in continuum_cache_names[prefix]:
                meta = parse_ferre_spectrum_name(each[int(result["ferre_index"])])
                assert int(meta["source_pk"]) == int(result["source_pk"])
                assert int(meta["spectrum_pk"]) == int(result["spectrum_pk"])
                assert int(meta["index"]) == int(result["ferre_index"])
        
        group_task_kwds[result["header_path"]].append(
            dict(
                spectra=spectrum,
                pre_computed_continuum=pre_computed_continuum[result["spectrum_pk"]],
                initial_teff=result["teff"],
                initial_logg=result["logg"],
                initial_m_h=result["m_h"],
                initial_log10_v_sini=result.get("log10_v_sini", np.nan),
                initial_log10_v_micro=result.get("log10_v_micro", np.nan),
                initial_alpha_m=result.get("alpha_m", np.nan),
                initial_c_m=result.get("c_m", np.nan),
                initial_n_m=result.get("n_m", np.nan),
            )
        )

    """
    The file structure for the abundances stage is:
    
    PARENT/abundances/
    PARENT/abundances/Mg_d/
    PARENT/abundances/Mg_d/input.nml
    PARENT/abundances/Mg_d/flux.input
    PARENT/abundances/Mg_d/e_flux.input
    PARENT/abundances/Mg_d/Al/input.nml
    PARENT/abundances/Mg_d/Al/parameters.input
    PARENT/abundances/Mg_d/Al/parameters.output

    # These two should be the same if we are not doing normalisation
    PARENT/abundances/Mg_d/Al/model_flux.output
    PARENT/abundances/Mg_d/Al/rectified_model_flux.output
    
    """
    extra_kwds = dict(
        continuum_order=continuum_order,
        continuum_flag=continuum_flag,
        continuum_observations_flag=continuum_observations_flag,
    )
    extra_kwds.update(kwargs)

    plans = []
    #spectra_with_no_stellar_parameters = set(spectra)
    for header_path in group_task_kwds.keys():

        grid_kwds = list_to_dict(group_task_kwds[header_path])
        short_grid_name = parse_header_path(header_path)["short_grid_name"]

        for i, (species, details) in enumerate(abundance_keywords[header_path].items()):
            weight_path, frozen_parameters, ferre_kwds = details
            kwds = grid_kwds.copy()
            kwds.update(
                pwd=os.path.join(parent_dir, STAGE, short_grid_name, species),
                header_path=header_path,
                weight_path=weight_path,
                frozen_parameters=frozen_parameters,
                ferre_kwds=ferre_kwds,
                # In the chemical abundances stage we avoid repeating the flux/e_flux files everywhere
                reference_pixel_arrays_for_abundance_run=True,
                # Only write the flux arrays to the parent folder on the first run of this header path
                write_input_pixel_arrays=(i == 0)
            )
            kwds.update(extra_kwds)
            
            if all(frozen_parameters.get(ln, False) for ln in ferre_headers[kwds["header_path"]][0]["LABEL"]):
                log.warning(f"Ignoring {species} species on grid {short_grid_name} because all parameters are frozen")
                continue
            plans.append(kwds)
        

    # Group together as necessary
    if use_ferre_list_mode:
        grouped = {}
        for plan in plans:
            short_grid_name = parse_header_path(plan["header_path"])["short_grid_name"]

            grouped.setdefault(short_grid_name, [])
            grouped[short_grid_name].append(plan)

            #spectra_with_no_stellar_parameters -= set(grid_kwds["spectra"])

        grouped_plans = list(grouped.values())        
        #spectra_with_no_stellar_parameters = tuple(spectra_with_no_stellar_parameters)
        #return (plans, spectra_with_no_stellar_parameters)
        return grouped_plans
    
    else:
        # In each directory, create symbolic links to the flux/e_flux arrays
        return [[plan] for plan in plans]


def get_species(weight_path):
    return os.path.basename(weight_path)[:-5]
