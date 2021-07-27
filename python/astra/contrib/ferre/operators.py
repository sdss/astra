import numpy as np
import os
import json
import pickle
from ast import literal_eval
from typing import Dict, Optional

from airflow.exceptions import AirflowSkipException
from airflow.models.baseoperator import BaseOperator
from airflow.hooks.subprocess import SubprocessHook
from airflow.models import BaseOperator
from airflow.utils.operator_helpers import context_to_airflow_vars
from sdss_access import SDSSPath

from sqlalchemy import or_, and_

from astropy.time import Time
from astropy.io.fits import getheader
from astropy.table import Table

from astra.tools.spectrum import Spectrum1D
from astra.tools.spectrum.writers import write_astra_source_data_product
from astra.utils import log, get_base_output_path
from astra.database import astradb, apogee_drpdb, catalogdb, session
from astra.database.utils import (
    create_task_output,
    serialize_pks_to_path,
    deserialize_pks,
    get_or_create_task_instance, 
    get_sdss4_apstar_kwds,
    get_sdss5_apstar_kwds,
    update_task_instance_parameters
)

from astra.contrib.ferre.core import ferre
from astra.contrib.ferre import utils
from astra.contrib.ferre.continuum import median_filtered_correction

def estimate_stellar_parameters(
        pks,
        header_path,
        spectrum_kwds=None,
        **kwargs
    ):
    """    
    Estimate the stellar parameters with the task instances given.

    :param pks:
        The primary keys of the task instances to estimate stellar parameters for.

    :param header_path:
        The path of the FERRE header file.

    :param spectrum_kwds: [optional]
        An optional dictionary of keyword arguments to supply to `astra.tools.spectrum.Spectrum1D`
        when loading spectra. For example, `spectrum_kwds=dict(data_slice=slice(0, 1))` will
        only return the first (stacked?) spectrum for analysis.

    :param \**kwargs:
        Keyword arguments to provide directly to `astra.contrib.ferre.ferre`.
    """
    print(f"In stellar parameters with pks {pks} ({type(pks)})")
    
    # Get the task instances.
    log.debug(f"In stellar parameters with pks {pks} ({type(pks)})")
    instances = get_instances_with_this_header_path(pks, header_path)

    # Load the spectra.
    spectrum_kwds = (spectrum_kwds or dict())
    wavelength, flux, sigma, spectrum_meta, Ns = load_spectra(instances, **spectrum_kwds)
    
    # Create names for easy debugging in FERRE outputs.
    names = create_names(instances, Ns, "{star_index}_{telescope}_{obj}_{spectrum_index}")

    # Load the initial parameters from the task instances.
    initial_parameters = create_initial_parameters(instances, Ns)
    
    # Run FERRE.
    param, param_err, output_meta = ferre(
        wavelength=wavelength,
        flux=flux,
        sigma=sigma,
        header_path=header_path,
        names=names,
        initial_parameters=initial_parameters,
        **kwargs
    )
    
    # Group results by instance.
    results = group_results_by_instance(param, param_err, output_meta, spectrum_meta, Ns)

    assert len(results) == len(instances)

    base_output_path = get_base_output_path()
    
    for instance, (result, data) in zip(instances, results):
        if result is None: continue

        # Write FERRE outputs to database.
        create_task_output(
            instance,
            astradb.Ferre,
            **result
        )

        # Update the original task instance to include keywords that went to FERRE.
        update_task_instance_parameters(instance, **output_meta["ferre_control_parameters"])
        
        # Write FERRE outputs to disk.
        output_path = os.path.join(
            base_output_path,
            "ferre",
            "scratch",
            f"{instance.pk}",
            "outputs.pkl"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fp:
            pickle.dump((result, data), fp)

        log.debug(f"Wrote additional outputs to {output_path}")


    # Return primary keys that we worked on, so these can be passed downstream.
    return [instance.pk for instance in instances]


def estimate_chemical_abundances(
        pks,
        header_path,
        element,
        spectrum_kwds=None,
        continuum_path_format=None,
        **kwargs
    ):
    """    
    Estimate the abundance of a chemical element with the task instances given.

    :param pks:
        The primary keys of the task instances to estimate stellar parameters for.

    :param header_path:
        The path of the FERRE header file.

    :param element:
        The name of the element to measure (e.g., 'Al'). This is used to govern the `TTIE`
        keywords for FERRE.

    :param spectrum_kwds: [optional]
        An optional dictionary of keyword arguments to supply to `astra.tools.spectrum.Spectrum1D`
        when loading spectra. For example, `spectrum_kwds=dict(data_slice=slice(0, 1))` will
        only return the first (stacked?) spectrum for analysis.

    :param continuum_path_format: [optional]
        A string representing the path to continuum files to use for the observations
        before supplying the data to FERRE. This is useful for making median-filtered
        corrections on the data before FERRE handles it. This should have a format like:

        '{telescope}/{obj}-continuum-{pk}.pkl'

        where the values will be formatted by Python in the usual way. The parameters
        available for string formatting include:

        - `pk`: the primary key of this task instance
        - `release`: the name of the SDSS data release
        - `filetype`: the SDSS data model name
        
        and all other keywords associated with that `filetype`. 

    :param \**kwargs:
        Keyword arguments to provide directly to `astra.contrib.ferre.ferre`.
    """

    # Get FERRE keywords for this element.
    headers, *segment_headers = utils.read_ferre_headers(utils.expand_path(header_path))
    frozen_parameters, ferre_kwds = utils.get_abundance_keywords(element, headers["LABEL"])

    # Overwrite frozen_parameters and ferre_kwds
    kwds = kwargs.copy()
    
    log.debug(f"Printing out everything")
    for k, v in kwds.items():
        log.debug(f"    {k}: {v} {type(v)}")
    
    kwds.setdefault("ferre_kwds", {})
    kwds["ferre_kwds"].update(**ferre_kwds)
    kwds["frozen_parameters"] = frozen_parameters

    log.debug(f"AND primary keys are {type(pks)} {pks}")

    return estimate_stellar_parameters(
        pks, 
        header_path, 
        spectrum_kwds=spectrum_kwds,
        continuum_path_format=continuum_path_format,
        **kwds
    )


def write_astra_source_data_products(pks):
    """
    Write the results of stellar parameters and chemical abundances to AstraSource
    data model products on disk.

    :param pks:
        the primary keys of previous task instances that have determined stellar
        parameters and chemical abundances
    """

    pks = deserialize_pks(pks, flatten=True)

    log.debug(f"Writing AstraSource objects with pks {pks}")

    base_output_path = get_base_output_path()
    # TODO: Make a single source of truth for this path reference
    get_output_path = lambda pk: os.path.join(base_output_path, "ferre", "scratch", f"{pk:.0f}", "outputs.pkl")

    trees = {}

    for pk in pks:

        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk)
        instance = q.one_or_none()

        #if "stellar_parameters" not in instance.task_id:
        #    log.debug(f"Skipping instance {instance} because need to decide what to store about chemical abundances")
        #    continue

        release = instance.parameters["release"]
        try:
            tree = trees[release]                
        except KeyError:
            tree = trees[release] = SDSSPath(release=release)

        path = tree.full(**instance.parameters)

        spectrum = Spectrum1D.read(path)
        
        # Get the FERRE outputs.
        for associated_pk in instance.output.associated_ti_pks:
            
            q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == associated_pk)
            associated_instance = q.one_or_none()

            if "stellar_parameters" in associated_instance.task_id:
                log.debug(f"Taking {associated_pk} as the associated instance {associated_instance} for stellar parameters")
                with open(get_output_path(associated_pk), "rb") as fp:
                    result, data = pickle.load(fp)
                break

            else:
                log.debug(f"Skipping associated instance {associated_instance} because {associated_instance.task_id}")

            
        # TODO: Check whether the median filtered correction is being applied here or not.
        continuum = data["continuum"]
        normalized_flux = data["flux"] / continuum
        normalized_ivar = (data["sigma"] / continuum)**-2


        N = len(instance.output.snr)
        result_rows = {}
        for k, v in instance.output.__dict__.items():
            if k == "_sa_instance_state": continue
            if k == "associated_ti_pks":
                v = " ".join(map(str, v))
            
            if isinstance(v, list):
                result_rows[k] = v
            else:
                if v is None:
                    v = ""
                result_rows[k] = [v] * N
        
        try:
            results_table = Table(data=result_rows)
            
        except:
            log.exception(f"Unable to create results table.")
            log.warning(f"Outputting row details:")
            for key, values in result_rows.items():
                log.warning(f"\t{key} ({type(key)}): {type(values)} ({len(values)})")
                
            log.warning(f"Data values:")
            for key, values in result_rows.items():
                log.warning(f"\t{key}: {values}")

            raise

        # Decide on where this will be stored.
        output_path = os.path.join(base_output_path, "aspcap", f"{instance.parameters['obj']}-{instance.pk}.fits")

        kwds = dict(
            output_path=output_path,
            spectrum=spectrum,
            normalized_flux=normalized_flux,
            normalized_ivar=normalized_ivar,
            continuum=continuum,
            model_flux=data["normalized_model_flux"],
            model_ivar=np.zeros_like(data["normalized_model_flux"]),
            results_table=results_table,
            instance=instance,
        )
        
        write_astra_source_data_product(**kwds)

        log.debug(f"Written AstraSource object to {output_path}")

    return None
    

def write_summary_database_outputs(pks, task_id="aspcap"):
    """
    Write the results of stellar parameters and chemical abundances to a unifed
    ASPCAP table in the database.

    :param pks:
        the primary keys of previous task instances that have determined stellar
        parameters and chemical abundances
    """

    log.debug(f"Input to pks is {pks}")
    
    pks = deserialize_pks(pks, flatten=True)
    log.debug(f"Writing summary database outputs with input primary keys {pks}")

    q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk.in_(pks))

    # We need to group and order the primary keys together.
    trees = {}
    grouped_pks = {}
    for instance in q.all():
        release = instance.parameters["release"]
        try:
            tree = trees[release]                
        except KeyError:
            tree = trees[release] = SDSSPath(release=release)

        keys = ["release", "filetype"]
        keys.extend(tree.lookup_keys(instance.parameters["filetype"]))

        uid = "_".join([instance.parameters[key] for key in keys])

        grouped_pks.setdefault(uid, [])
        log.debug(f"Created grouped set of primary keys with uid {uid}")

        # Put the stellar parameter instance first.    
        if "stellar_parameters" in instance.task_id:
            log.debug(f"Adding task instance {instance} as stellar parameter instance {uid}")
            grouped_pks[uid].insert(0, instance.pk)
        else:
            log.debug(f"Adding task instance {instance} as elemental abundance instance {uid}")
            grouped_pks[uid].append(instance.pk)
    

    instances = []    
    for uid, (sp_pk, *element_pks) in grouped_pks.items():

        log.debug(f"Grouped together primary keys {sp_pk} and {element_pks}")

        sp_instance = session.query(astradb.TaskInstance)\
                             .filter(astradb.TaskInstance.pk == sp_pk).one_or_none()
        el_instances = session.query(astradb.TaskInstance)\
                              .filter(astradb.TaskInstance.pk.in_(element_pks)).all()

        # Get the keys that are common to all instances, which will include the release, filetype,
        # and associated keys.
        keep_keys = []
        for key, value in sp_instance.parameters.items():
            for instance in el_instances:
                if instance.parameters[key] != value:
                    break
            else:
                keep_keys.append(key)
        
        log.debug(f"Keeping keys {keep_keys}")

        parameters = { key: instance.parameters[key] for key in keep_keys }
        
        log.debug(f"Passing parameters {parameters}")

        instance = get_or_create_task_instance(
            task_id=task_id,
            dag_id=sp_instance.dag_id,
            run_id=sp_instance.run_id,
            parameters=parameters,
        )

        # Create a partial results table.
        keys = ["snr"]
        label_names = ("teff", "logg", "metals", "log10vdop", "o_mg_si_s_ca_ti", "lgvsini", "c", "n")
        for key in label_names:
            keys.extend([key, f"u_{key}"])
        
        results = dict([(key, getattr(sp_instance.output, key)) for key in keys])

        # Now update with elemental abundance instances.
        for el_instance in el_instances:
            # TODO: Should not rely on the task ID to infer the element abundance.
            #       Should we use the input_weights_path, or provide 'element' as a parameter?
            element = el_instance.task_id.split(".")[1].lower()
            
            # Check what is not frozen.
            thawed_label_names = []
            ignore = ("lgvsini", ) # Ignore situations where lgvsini was missing from grid and it screws up the task
            for key in label_names:
                if key not in ignore and not getattr(el_instance.output, f"frozen_{key}"):
                    thawed_label_names.append(key)

            if len(thawed_label_names) > 1:
                log.warning(f"Multiple thawed label names for {element} {el_instance}: {thawed_label_names}")

            values = np.hstack([getattr(el_instance.output, ln) for ln in thawed_label_names]).tolist()
            u_values = np.hstack([getattr(el_instance.output, f"u_{ln}") for ln in thawed_label_names]).tolist()

            results.update({
                f"{element}_h": values,
                f"u_{element}_h": u_values,
            })

        # Include associated primary keys so we can reference back to original parameters, etc.
        results["associated_ti_pks"] = [sp_pk, *element_pks]

        log.debug(f"Results entry: {results}")

        # Create an entry in the output interface table.
        # (We will update this later with any elemental abundance results).
        # TODO: Should we link back to the original FERRE primary keys?
        output = create_task_output(
            instance,
            astradb.Aspcap,
            **results
        )
        log.debug(f"Created output {output} for instance {instance}")
        instances.append(instance)
    
    return [instance.pk for instance in instances]


def get_instances_with_this_header_path(pks, header_path):
    """
    Get the task instances that have a primary key in the list of primary keys given,
    and have a matching header path.
    
    :param pks:
        A list of primary keys of task instances.
    
    :param header_path:
        The header path of the FERRE grid file.
    
    :returns:
        A list of `astra.database.astradb.TaskInstance` objects.
    """
    pks = deserialize_pks(pks, flatten=True)
    log.debug(f"in get_instances_with_this_header_path with {pks} ({type(pks)})")
    q = session.query(astradb.TaskInstance).join(astradb.TaskInstanceParameter).join(astradb.Parameter)
    q = q.filter(and_(
        astradb.TaskInstance.pk.in_(pks),
        astradb.Parameter.parameter_name == "header_path",
        astradb.Parameter.parameter_value == header_path
    ))
    
    instances = q.all()
    log.info(f"There are {len(instances)} relevant primary keys")
    return instances


def group_results_by_instance(
        param, 
        param_err, 
        output_meta, 
        spectrum_meta,
        Ns
    ):
    """
    Group FERRE results together into a list of dictionaries where the size of the outputs
    matches the size of the input spectra loaded for each task instance.
    
    :param param:
        The array of output parameters from FERRE.

    :param param_err:
        The estimated errors on the output parameters from FERRE.
    
    :param output_meta:
        A metadata dictionary output by FERRE.
    
    :param spectrum_meta:
        A list of dictionaries of spectrum metadata for each instance.
    
    :param Ns:
        A list of integers indicating the number of spectra that were loaded with each
        instance (e.g., `sum(Ns)` should equal `param.shape[0]`).
    
    :returns:
        A list of dictionaries that contain results for each instance.
    """
    
    common_results = dict(frozen_parameters=output_meta["frozen_parameters"])
    
    si, results = (0, [])
    parameter_names = tuple(map(utils.sanitise_parameter_name, output_meta["parameter_names"]))
    for i, N in enumerate(Ns):
        if N == 0: 
            results.append((None, None))
            continue
        
        sliced = slice(si, si + N)

        result = dict(
            snr=spectrum_meta[i]["snr"],
            log_snr_sq=output_meta["log_snr_sq"][sliced],
            log_chisq_fit=output_meta["log_chisq_fit"][sliced],
        )

        data = {}
        for key in ("wavelength", "flux", "sigma", "normalized_model_flux", "continuum"):
            data[key] = output_meta[key][sliced]

        # Same for all results in this group, but we include it for convenience.
        # TODO: Consider sending back something else instead of mask array.
        data["mask"] = output_meta["mask"]

        for j, parameter_name in enumerate(parameter_names):
            result[f"{parameter_name}"] = param[sliced][:, j]
            result[f"u_{parameter_name}"] = param_err[sliced][:, j]
            result[f"initial_{parameter_name}"] = output_meta["initial_parameters"][sliced][:, j]
            result[f"frozen_{parameter_name}"] = output_meta["frozen_parameters"][parameter_name]

        results.append((result, data))
    
    return results



def create_initial_parameters(instances, Ns):
    """
    Create a list of initial parameters for spectra loaded from task instances.

    :param instances:
        A list of task instances where spectra were loaded. This should
        be length `N` long.
    
    :param Ns:
        A list containing the number of spectra loaded from each task instance.
        This should have the same length as `instances`.
    
    :returns:
        A dictionary of initial values.
    """
    initial_parameters = {}
    for i, (instance, N) in enumerate(zip(instances, Ns)):
        if i == 0:
            for key in instance.parameters.keys():
                if key.startswith("initial_"):
                    initial_parameters[utils.desanitise_parameter_name(key[8:])] = []
        
        for key, value in instance.parameters.items():
            if key.startswith("initial_"):
                ferre_label = utils.desanitise_parameter_name(key[8:])
                value = json.loads(value)
                if value is None:
                    value = [np.nan] * N
                elif isinstance(value, (float, int)):
                    value = [value] * N
                elif isinstance(value, (list, tuple)):
                    assert len(value) == N

                initial_parameters[ferre_label].extend(value)

    return initial_parameters


def create_names(instances, Ns, str_format):
    """
    Create a list of names for spectra loaded from task instances.

    :param instances:
        A list of task instances where spectra were loaded. This should
        be length `N` long.
    
    :param Ns:
        A list containing the number of spectra loaded from each task instance.
        This should have the same length as `instances`.
    
    :param str_format:
        A string formatting for the names. The available keywords include
        all parameters associated with the task, as well as the `star_index`
        and the `spectrum_index`.
    
    :returns:
        A list with length `sum(Ns)` that contains the given names for all 
        the spectra loaded.
    """
    names = []
    for star_index, (instance, N) in enumerate(zip(instances, Ns)):
        kwds = instance.parameters.copy()
        kwds.update(star_index=star_index)
        for index in range(N):
            kwds["spectrum_index"] = index
            names.append(str_format.format(**kwds))
    return names


# TODO: This is a sufficiently common function that it should probably go elsewhere.
# TODO: We should also consider returning metadata.
def load_spectra(instances, **kwargs):
    """
    Load spectra from the given task instances.

    :param instances:
        A list of task instances to load associated spectra for.
    
    All other kwargs are passed directly to `astra.tools.spectrum.Spectrum1D`.

    :returns:
        A four length tuple that contains:
        
        1. A wavelength array of shape (N, P), where N is the number of spectra
           (not instances!) and P is the number of pixels.
        2. A flux array of shape (N, P).
        3. A flux uncertainty array of shape (N, P).
        4. A `T` length array, where `T` is the number of task instances provided,
           where each value in the array indicates the number of spectra loaded
           from that task instance.
    """

    # Load the observations from the database instances.
    trees = {}
    wavelength, flux, sigma, meta = ([], [], [], [])
    
    Ns = np.zeros(len(instances), dtype=int)
    data_slice = kwargs.get("data_slice", slice(None)) # for extracting snr info

    base_output_path = get_base_output_path()
    # TODO: Make a single source of truth for this path reference
    get_output_path = lambda pk: os.path.join(base_output_path, "ferre", "scratch", f"{pk:.0f}", "outputs.pkl")

    for i, instance in enumerate(instances):
        release = instance.parameters["release"]
        try:
            tree = trees[release]                
        except KeyError:
            tree = trees[release] = SDSSPath(release=release)
        
        path = tree.full(**instance.parameters)

        try:
            # TODO: Profile this.
            spectrum = Spectrum1D.read(path, **kwargs)
            N, P = spectrum.flux.shape

            flux_ = spectrum.flux.value
            sigma_ = spectrum.uncertainty.array**-0.5            

            prev_pk = instance.parameters.get("use_median_filter_continuum_correction_from_pk", None)
            if prev_pk is not None:
                prev_path = get_output_path(int(prev_pk))
                log.debug(
                    f"In instance {instance.pk} we are applying median filtered continuum correction "\
                    f"from previous instance {prev_pk} stored at {prev_path}"
                )
                with open(prev_path, "rb") as fp:
                    result, data = pickle.load(fp)

                flux_ /= data["median_filtered_continuum"]
                sigma_ /= data["median_filtered_continuum"]
            else:
                log.debug(f"Not doing any median_filter_continuum_correction on instance {instance}")

            flux.append(flux_)
            sigma.append(sigma_)
            wavelength.append(
                np.tile(spectrum.wavelength.value, N).reshape((N, -1))
            )
            meta.append(dict(snr=spectrum.meta["snr"][data_slice]))
        except:
            log.exception(f"Exception in trying to load data product associated with {instance} from {path} with {kwargs}")
            meta.append({})
            raise 
            

        else:
            Ns[i] = N
    
    wavelength, flux, sigma = tuple(map(np.vstack, (wavelength, flux, sigma)))
    
    return (wavelength, flux, sigma, meta, Ns)


def yield_initial_guess_from_doppler_headers(data_model_kwds):
    """
    Get initial guesses for the sources provided.
    """

    # Get the observation parameters from the upstream task.
    # TODO: Include photometry code into this task, because we need the telescope and mean fiber
    #       to compare against grids.

    trees = {}

    for kwds in data_model_kwds:

        tree = trees.get(kwds["release"], None)
        if tree is None:
            tree = trees[kwds["release"]] = SDSSPath(release=kwds["release"])
        
        try:
            path = tree.full(**kwds)

            header = getheader(path)

            teff = utils.safe_read_header(header, ("RV_TEFF", "RVTEFF"))
            logg = utils.safe_read_header(header, ("RV_LOGG", "RVLOGG"))
            fe_h = utils.safe_read_header(header, ("RV_FEH", "RVFEH"))

            # Get information relevant for matching initial guess and grids.
            initial_guess = dict(
                telescope=kwds["telescope"], # important for LSF information
                mean_fiber=header["MEANFIB"], # important for LSF information
                teff=teff,
                logg=logg,
                metals=fe_h,
            )

        except Exception as exception:
            log.exception(f"Exception: {exception}")
            continue

        else:
            yield (kwds, initial_guess)


def median_filter_continuum(pks, **kwargs):
    """
    Estimate the continuum using a median filtered correction, based on the existing result.

    :param pks:
        the primary keys of the task instances to estimate continuum for
    
    :param \**kwargs:
        all keyword arguments will go directly to `astra.contrib.ferre.continuum.median_filtered_correction`
    """

    pks = deserialize_pks(pks, flatten=True)

    # Get the path
    base_output_path = get_base_output_path()
    # TODO: Make a single source of truth for this path reference
    get_output_path = lambda pk: os.path.join(base_output_path, "ferre", "scratch", f"{pk:.0f}", "outputs.pkl")

    all_n_pixels = {}

    for pk in pks:
        log.debug(f"Running median filtered continuum for primary key {pk}")

        with open(get_output_path(pk), "rb") as fp:
            result, data = pickle.load(fp)
        
        # Need the header path.
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk).one_or_none()
        header_path = q.parameters["header_path"]

        try:
            n_pixels = all_n_pixels[header_path]
        except KeyError:
            n_pixels = all_n_pixels[header_path] = [header["NPIX"] for header in utils.read_ferre_headers(utils.expand_path(header_path))][1:]

        indices = 1 + np.cumsum(data["mask"]).searchsorted(np.cumsum(n_pixels))
        # These indices will be for each chip, but will need to be left-trimmed.
        segment_indices = np.sort(np.hstack([
            0,
            np.repeat(indices[:-1], 2),
            data["mask"].size
        ])).reshape((-1, 2))
        
        # Left-trim the indices.
        for i, (start, end) in enumerate(segment_indices):
            segment_indices[i, 0] += data["mask"][start:].searchsorted(True)
    
        continuum = median_filtered_correction(
            wavelength=data["wavelength"],
            # TODO: Check this median filtered correction.
            normalised_observed_flux=data["flux"] / data["continuum"],
            normalised_observed_flux_err=data["sigma"] / data["continuum"],
            normalised_model_flux=data["normalized_model_flux"],
            segment_indices=segment_indices,
            **kwargs
        )

        # Save it back to the file.
        data["median_filtered_continuum"] = continuum
        
        with open(get_output_path(pk), "wb") as fp:
            pickle.dump((result, data), fp)

        log.info(f"Wrote median filtered continuum output back to {get_output_path(pk)}")
        '''
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        axes[0].plot(data["wavelength"][0], data["normalized_model_flux"][0], c="r")
        axes[0].plot(data["wavelength"][0], data["flux"][0] / data["continuum"][0], c="k")
        axes[1].plot(data["wavelength"][0], continuum, c="b")
        axes[2].plot(data["wavelength"][0], data["normalized_model_flux"][0], c="r")
        axes[2].plot(data["wavelength"][0], (data["flux"][0] / data["continuum"][0]) / continuum, c="k")
        fig.tight_layout()
        fig.savefig("/uufs/chpc.utah.edu/common/home/u6020307/astra/tmp.png", dpi=300)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data["wavelength"][0], continuum, c='b')
        fig.savefig("/uufs/chpc.utah.edu/common/home/u6020307/astra/tmp2.png", dpi=300)
        '''

    log.debug(f"Done")
    return [] # Do not return 'None' here or it wont deserialize downstream.



def _create_partial_ferre_task_instances_from_observations(
        dag_id, 
        run_id,
        task_id_function,
        data_model_kwds,
        ferre_header_paths,
        **kwargs
    ):

    # Get grid information.
    grid_info = utils.parse_grid_information(ferre_header_paths)

    # Get initial guesses.
    initial_guesses = yield_initial_guess_from_doppler_headers(data_model_kwds)
    
    # Match observations, initial guesses, and FERRE header files.
    instance_meta = []
    for kwds, initial_guess in initial_guesses:
        for header_path, ferre_headers in utils.yield_suitable_grids(grid_info, **initial_guess):
            instance_meta.append(dict(
                header_path=header_path,
                # Add the initial guess information.
                initial_teff=np.round(initial_guess["teff"], 0),
                initial_logg=np.round(initial_guess["logg"], 2),
                initial_metals=np.round(initial_guess["metals"], 2),
                initial_log10vdop=np.round(utils.approximate_log10_microturbulence(initial_guess["logg"]), 2),
                initial_o_mg_si_s_ca_ti=0.0,
                # lgvsini = 0 is not always in the bounds of the grid (e.g., p_apstdM_180901_lsfa_l33_012_075.hdr has lgvsini limits (0.18, 2.28))
                # but by default we will clip initial values to be within the bounds of the grid
                initial_lgvsini=0.0, 
                initial_c=0.0,
                initial_n=0.0,
                # Add the data model keywords.
                **kwds
            ))

    # Create task instances.
    log.debug(f"Task ID function is {task_id_function}")
    pks = []
    for meta in instance_meta:
        task_id = task_id_function(**meta)
        instance = get_or_create_task_instance(
            dag_id=dag_id,
            run_id=run_id,
            task_id=task_id,
            parameters=meta
        )
        log.info(f"Created or retrieved task instance {instance} for {dag_id} {task_id} with {meta}")
        pks.append(instance.pk)
    
    # Return the primary keys, which will be passed on to the next task.
    return serialize_pks_to_path(pks, **kwargs)
    

def create_task_instances_for_sdss4_apstars(dag_id, task_id_function, ferre_header_paths, **kwargs):
    """
    Query the database for SDSS-4 ApStar observations and create partial task instances with
    their observation keywords, initial guesses, and other FERRE requirements.

    This function will return primary key values in the `astra.ti` table.
    
    :param dag_id:
        The identifier of the Apache Airflow directed acyclic graph.
    
    :param task_id_function:
        A callable that takes in metadata keywords of the task instance and returns the name of the
        task identifier.
    
    :param ferre_header_paths:
        A list of FERRE header paths to consider.
    """

    data_model_kwds = get_sdss4_apstar_kwds()
    
    # Create the task instances.
    return _create_partial_ferre_task_instances_from_observations(
        dag_id=dag_id,
        task_id_function=task_id_function,
        data_model_kwds=data_model_kwds,
        ferre_header_paths=ferre_header_paths,
        **kwargs
    )

def create_task_instances_for_sdss5_apstars(
        mjd,
        dag_id,
        run_id,
        task_id_function,
        ferre_header_paths,
        limit=None,
        **kwargs
    ):
    """
    Query the database for SDSS-V ApStar observations taken on the date start, and create
    partial task instances with their observation keywords, initial guesses, and other FERRE
    requirements.

    This function will return primary key values in the `astra.ti` table.

    :param MJD:
        The Modified Julian Date of the ApStar observations.
    
    :param dag_id:
        The identifier of the Apache Airflow directed acyclic graph.
    
    :param run_id:
        The identifier of the Apache Airflow run.
    
    :param task_id_function:
        A callable that takes in metadata keywords of the task instance and returns the name of the
        task identifier.
    
    :param ferre_header_paths:
        A list of FERRE header paths to consider.
    """

    '''
    # TODO: Here we are assuming a "@daily" interval schedule. We should consider how to incorporate
    #       the schedule interval.
    mjd = Time(ds).mjd
    print(f"Cheating and taking the most recent MJD.")
    q = session.query(apogee_drpdb.Star.mjdend).order_by(apogee_drpdb.Star.mjdend.desc())
    mjd, = q.limit(1).one_or_none()
    '''

    # Get the identifiers for the APOGEE observations taken on this MJD.
    data_model_kwds = get_sdss5_apstar_kwds(mjd, limit=limit)

    log.info(f"There are {len(data_model_kwds)} keywords for MJD {mjd}")

    # Create the task instances.
    return _create_partial_ferre_task_instances_from_observations(
        dag_id=dag_id,
        run_id=run_id,
        task_id_function=task_id_function,
        data_model_kwds=data_model_kwds,
        ferre_header_paths=ferre_header_paths,
        **kwargs
    )


def create_task_instances_for_next_iteration(
        pks, 
        task_id_function, 
        use_median_filter_continuum_correction=False,
        **kwargs
    ):
    """
    Create task instances for a subsequent iteration of FERRE execution, based on
    some FERRE task instances that have already been executed. An example might be
    running FERRE with some dimensions fixed to get a poor estimate of parameters,
    and then running FERRE again without those parameters fixed. This function 
    could be used to create the task instances for the second FERRE execution.

    :param pks:
        The primary keys of the existing task instances.

    :param task_id_function:
        A callable function that returns the task ID to use, given the parameters.
    
    :param use_median_filter_continuum_correction: [optional]
        On the new tasks, use the median filter continuum correction from the previous 
        results.
    """

    log.debug(f"Deserializing pks {pks} ({type(pks)}")
    print(f"PRINT deserializing pks {pks} ({type(pks)})")

    pks = deserialize_pks(pks, flatten=True)
    
    # If we are given the task ID of the immediate downstream task, then just use that.
    log.debug(f"Creating task instances for next iteration given primary keys {pks} and task_id_function {task_id_function} (type: {type(task_id_function)})")
    if isinstance(task_id_function, str):
        try:
            task_id_function = literal_eval(task_id_function)
        except:
            log.debug(f"Using task_id_function as given")
            raise 
        
        else:
            if isinstance(task_id_function, (set, )) and len(task_id_function) == 1:
                value, = list(task_id_function)
                task_id_function = lambda **_: value
                log.debug(f"Using task identifier '{value}' for all task instances")
            else:
                raise ValueError(f"not sure what to do with de-serialized task instance ID function {task_id_function} (type: {type(task_id_function)})")

    # Get the existing task instances.
    q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk.in_(pks))

    first_or_none = lambda item: None if item is None else item[0]

    # For each one, create a new task instance but set the initial_teff/ etc
    # as the output from the previous task instance.
    keys = [
        ("initial_teff", lambda i: first_or_none(i.output.teff)),
        ("initial_logg", lambda i: first_or_none(i.output.logg)),
        ("initial_metals", lambda i: first_or_none(i.output.metals)),
        ("initial_log10vdop", lambda i: first_or_none(i.output.log10vdop)),
        ("initial_o_mg_si_s_ca_ti", lambda i: first_or_none(i.output.o_mg_si_s_ca_ti)),
        ("initial_lgvsini", lambda i: first_or_none(i.output.lgvsini)),
        ("initial_c", lambda i: first_or_none(i.output.c)),
        ("initial_n", lambda i: first_or_none(i.output.n)),
    ]
    
    trees = {}

    new_pks = []
    for instance in q.all():

        # Initial parameters.
        parameters = { k: f(instance) for k, f in keys }

        # Data keywords
        release = instance.parameters["release"]
        filetype = instance.parameters["filetype"]
        header_path = instance.parameters["header_path"]
        parameters.update(
            release=release,
            filetype=filetype,
            header_path=header_path
        )

        # Use any continuum correction?
        if use_median_filter_continuum_correction:
            # If it was already supplied in the previous instance, just propagate that.
            k = "use_median_filter_continuum_correction_from_pk" 
            if k in instance.parameters:
                log.debug(f"Propagating {k} from previous instance {instance}: {instance.parameters[k]}")
                parameters[k] = instance.parameters[k]
            else:
                log.debug(f"Setting {k} as {instance.pk} from instance {instance}")
                parameters[k] = instance.pk

        tree = trees.get(release, None)
        if tree is None:
            tree = trees[release] = SDSSPath(release=release)
        
        for key in tree.lookup_keys(filetype):
            parameters[key] = instance.parameters[key]

        new_instance = get_or_create_task_instance(
            dag_id=instance.dag_id,
            task_id=task_id_function(**parameters),
            run_id=instance.run_id,
            parameters=parameters
        )
        new_pks.append(new_instance.pk)
        log.debug(f"Retrieved or created new instance {new_instance} with header path {header_path}")
    
    return serialize_pks_to_path(new_pks, **kwargs)



def choose_which_ferre_tasks_to_execute(pks, task_id_function, **kwargs):
    """
    A function to be used with BranchPythonOperator that selects the FERRE tasks that
    need executing, based on the FERRE header paths in partial task instances.
    
    :param pks:
        The primary keys of the possible FERRE tasks.
    
    :param task_id_function:
        A callable that takes in metadata keywords of the task instance and returns the name of the
        task identifier.
    """
    # Get primary keys from immediate upstream task.
    pks = deserialize_pks(pks, flatten=True)

    # Get the header paths for those task instances.
    q = session.query(astradb.Parameter.parameter_value)\
                .join(astradb.TaskInstanceParameter)\
                .join(astradb.TaskInstance)\
                .distinct(astradb.Parameter.parameter_value)\
                .filter(astradb.TaskInstance.pk.in_(pks))\
                .filter(astradb.Parameter.parameter_name == "header_path")

    rows = q.all()

    task_ids = [task_id_function(header_path=header_path) for header_path, in rows]
    log.debug(f"Execute the following task IDs: {task_ids}")
    return task_ids


def get_best_initial_guess(pks, **kwargs):
    """
    When there are numerous FERRE tasks that are upstream, this
    function will return the primary keys of the task instances that gave
    the best result on a per-observation basis.
    """

    # Get the PKs from upstream.
    pks = deserialize_pks(pks, flatten=True)

    log.debug(f"Getting best initial guess among primary keys {pks}")

    # Need to uniquely identify observations.
    trees = {}
    best_tasks = {}
    for i, pk in enumerate(pks):
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk==pk)
        instance = q.one_or_none()

        if instance.output is None:
            log.warning(f"No output found for task instance {instance}")
            continue

        p = instance.parameters
        try:
            tree = trees[p["release"]]                
        except KeyError:
            tree = trees[p["release"]] = SDSSPath(release=p["release"])
        
        key = "_".join([
            p['release'],
            p['filetype'],
            *[p[k] for k in tree.lookup_keys(p['filetype'])]
        ])
        
        best_tasks.setdefault(key, (np.inf, None))
        
        log_chisq_fit, *_ = instance.output.log_chisq_fit
        previous_teff, *_ = instance.output.teff

        # Note: If FERRE totally fails then it will assign -999 values to the log_chisq_fit. So we have to
        #       check that the log_chisq_fit is actually sensible!
        #       (Or we should only query task instances where the output is sensible!)
        if log_chisq_fit < 0:
            log.debug(f"Skipping result for {instance} {instance.output} as log_chisq_fit = {log_chisq_fit}")
            continue
            
        parsed_header = utils.parse_header_path(p["header_path"])
    
        # Penalise chi-sq in the same way they did for DR16.
        # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
        if parsed_header["spectral_type"] == "GK" and previous_teff < 3985:
            # \chi^2 *= 10
            log_chisq_fit += np.log(10)

        # Is this the best so far?
        if log_chisq_fit < best_tasks[key][0]:
            log.debug(f"Assigning this output to best task as {log_chisq_fit} < {best_tasks[key][0]}: {pk}")
            best_tasks[key] = (log_chisq_fit, pk)
    
    for key, (log_chisq_fit, pk) in best_tasks.items():
        if pk is None:
            log.warning(f"No good task found for key {key}: ({log_chisq_fit}, {pk})")
        else:
            log.info(f"Best task for key {key} with log \chi^2 of {log_chisq_fit:.2f} is primary key {pk}")

    if best_tasks:
        return [pk for (log_chisq_fit, pk) in best_tasks.values() if pk is not None]
    else:
        raise AirflowSkipException(f"no task outputs found from {len(pks)} primary keys")

