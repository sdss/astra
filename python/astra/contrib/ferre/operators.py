import numpy as np
import os
import json
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

from astra.tools.spectrum import Spectrum1D

#from astra.contrib.ferre.utils import (
#    parse_header_path, parse_grid_information, safe_read_header, approximate_log10_microturbulence, yield_suitable_grids,
#    write_data_file, format_ferre_input_parameters, read_output_parameter_file,
#    sanitise_parameter_name
#)
from astra.utils import log, flatten
from astra.database import astradb, apogee_drpdb, catalogdb, session
from astra.database.utils import (
    create_task_output,
    deserialize_pks,
    get_or_create_task_instance, 
    get_sdss4_apstar_kwds,
    get_sdss5_apstar_kwds,
)

from astra.contrib.ferre.core import ferre
from astra.contrib.ferre import utils





def estimate_stellar_parameters(
        pks,
        header_path,
        spectrum_kwds=None,
        continuum_path_format=None,
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

    # Get the task instances.
    instances = get_instances_with_this_header_path(pks, header_path)

    # Load the spectra.
    spectrum_kwds = (spectrum_kwds or dict())
    wavelength, flux, sigma, Ns = load_spectra(instances, **spectrum_kwds)
    
    # Apply any continuum corrections.
    if continuum_path_format is not None:
        raise NotImplementedError

    # Create names for easy debugging in FERRE outputs.
    names = create_names(instances, Ns, "{star_index}_{telescope}_{obj}_{spectrum_index}")

    # Load the initial parameters from the task instances.
    initial_parameters = create_initial_parameters(instances, Ns)

    # Run FERRE.
    param, param_err, meta = ferre(
        wavelength=wavelength,
        flux=flux,
        sigma=sigma,
        header_path=header_path,
        names=names,
        initial_parameters=initial_parameters,
        **kwargs
    )
    
    # Group results by instance.
    results = group_results_by_instance(param, param_err, meta, Ns)

    # Write FERRE outputs to database and to disk
    for instance, result in zip(instances, results):
        if result is None: continue
        create_task_output(
            instance,
            astradb.Ferre,
            **result
        )

    return None



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
    kwds.setdefault(ferre_kwds, {})
    kwds["ferre_kwds"].update(ferre_kwds)
    kwds["frozen_parameters"] = frozen_parameters

    return estimate_stellar_parameters(
        pks, 
        header_path, 
        spectrum_kwds=spectrum_kwds,
        continuum_path_format=continuum_path_format,
        **kwds
    )


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
    pks = deserialize_pks(pks)
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
        meta, 
        Ns
    ):
    """
    Group FERRE results together into a list of dictionaries where the size of the outputs
    matches the size of the input spectra loaded for each task instance.
    
    :param param:
        The array of output parameters from FERRE.

    :param param_err:
        The estimated errors on the output parameters from FERRE.
    
    :param meta:
        A metadata dictionary output by FERRE.
    
    :param Ns:
        A list of integers indicating the number of spectra that were loaded with each
        instance (e.g., `sum(Ns)` should equal `param.shape[0]`).
    
    :returns:
        A list of dictionaries that contain results for each instance.
    """
    
    common_results = dict(frozen_parameters=meta["frozen_parameters"])
    
    si, results = (0, [])
    parameter_names = tuple(map(utils.sanitise_parameter_name, meta["parameter_names"]))
    for i, N in enumerate(Ns):
        if N == 0: 
            results.append(None)
            continue
        
        sliced = slice(si, si + N)

        result = dict(
            log_snr_sq=meta["log_snr_sq"][sliced],
            log_chisq_fit=meta["log_chisq_fit"][sliced]
        )
        for j, parameter_name in enumerate(parameter_names):
            result[f"{parameter_name}"] = param[sliced][:, j]
            result[f"u_{parameter_name}"] = param_err[sliced][:, j]
            result[f"initial_{parameter_name}"] = meta["initial_parameters"][sliced][:, j]

        results.append(result)
    
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

                try:
                    initial_parameters[ferre_label].extend(value)
                except:
                    print(f"Setting {ferre_label} with {value}")
                    raise


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
    wavelength, flux, sigma = ([], [], [])
    
    Ns = np.zeros(len(instances), dtype=int)

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
            flux.append(spectrum.flux.value)
            sigma.append(spectrum.uncertainty.array**-0.5)
            wavelength.append(
                np.tile(spectrum.wavelength.value, N).reshape((N, -1))
            )
        except:
            log.exception(f"Exception in trying to load data product associated with {instance} from {path} with {kwargs}")
            raise 

        else:
            Ns[i] = N
    
    wavelength, flux, sigma = tuple(map(np.vstack, (wavelength, flux, sigma)))
    
    return (wavelength, flux, sigma, Ns)


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



def _create_partial_ferre_task_instances_from_observations(
        dag_id, 
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
    pks = []
    for meta in instance_meta:
        task_id = task_id_function(**meta)
        instance = get_or_create_task_instance(
            dag_id=dag_id,
            task_id=task_id,
            parameters=meta
        )
        log.info(f"Created or retrieved task instance {instance} for {dag_id} {task_id} with {meta}")
        pks.append(instance.pk)
    
    # Return the primary keys, which will be passed on to the next task.
    return pks
    

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
        task_id_function=task_id_function,
        data_model_kwds=data_model_kwds,
        ferre_header_paths=ferre_header_paths,
        **kwargs
    )


def create_task_instances_for_next_iteration(pks, task_id_function, full_output=False):
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
    """

    pks = flatten(deserialize_pks(pks))

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

    new_instances = []
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

        tree = trees.get(release, None)
        if tree is None:
            tree = trees[release] = SDSSPath(release=release)
        
        for key in tree.lookup_keys(filetype):
            parameters[key] = instance.parameters[key]

        new_instances.append(get_or_create_task_instance(
            dag_id=instance.dag_id,
            task_id=task_id_function(**parameters),
            parameters=parameters
        ))
    
    if full_output:
        return new_instances

    return [instance.pk for instance in new_instances]





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
    pks = flatten(deserialize_pks(pks))

    # Get the header paths for those task instances.
    q = session.query(astradb.Parameter.parameter_value)\
                .join(astradb.TaskInstanceParameter)\
                .join(astradb.TaskInstance)\
                .distinct(astradb.Parameter.parameter_value)\
                .filter(astradb.TaskInstance.pk.in_(pks))\
                .filter(astradb.Parameter.parameter_name == "header_path")

    rows = q.all()

    return [task_id_function(header_path=header_path) for header_path, in rows]


def get_best_initial_guess(pks, **kwargs):
    """
    When there are numerous FERRE tasks that are upstream, this
    function will return the primary keys of the task instances that gave
    the best result on a per-observation basis.
    """

    # Get the PKs from upstream.
    pks = deserialize_pks(pks)

    # Need to uniquely identify observations.
    trees = {}
    best_tasks = {}
    for pk in pks:
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk==pk)
        instance = q.one_or_none()

        if instance.output is None:
            log.warn(f"No output found for task instance {instance}")
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

        parsed_header = utils.parse_header_path(p["header_path"])
    
        # Penalise chi-sq in the same way they did for DR16.
        # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
        if parsed_header["spectral_type"] == "GK" and previous_teff < 3985:
            # \chi^2 *= 10
            log_chisq_fit += np.log(10)

        # Is this the best so far?
        if log_chisq_fit < best_tasks[key][0]:
            best_tasks[key] = (log_chisq_fit, pk)
    
    if best_tasks:
        return [pk for (log_chisq_fit, pk) in best_tasks.values()]
    else:
        raise ValueError(f"no task outputs found from {len(pks)} primary keys")



class FerreOperator2222(BaseOperator):

    template_fields = ("pks", )
    template_fields_renderers = {
        "pks": "py"
    }

    def __init__(
        self,
        pks,
        *,
        header_path: str,
        frozen_parameters: Optional[Dict] = None,
        interpolation_order: int = 3,
        init_flag: int = 1,
        init_algorithm_flag: int = 1,
        error_algorithm_flag: int = 0,
        continuum_flag: Optional[int] = None,
        continuum_order: Optional[int] = None,
        continuum_reject: Optional[float] = None,
        continuum_observations_flag: int = 0,
        optimization_algorithm_flag: int = 3,
        wavelength_interpolation_flag: int = 0,
        lsf_shape_flag: int = 0,
        use_direct_access: bool = False,
        n_threads: int = 1,
        input_weights_path: Optional[str] = None,
        input_lsf_path: Optional[str] = None,
        ferre_kwds: Optional[Dict[str, str]] = None,
        slurm_kwds: Optional[Dict] = None,
        analyze_individual_visits: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pks = pks
        self.header_path = header_path
        self.frozen_parameters = frozen_parameters
        self.interpolation_order = interpolation_order
        self.init_flag = init_flag
        self.init_algorithm_flag = init_algorithm_flag
        self.error_algorithm_flag = error_algorithm_flag
        self.continuum_flag = continuum_flag
        self.continuum_order = continuum_order
        self.continuum_reject = continuum_reject
        self.continuum_observations_flag = continuum_observations_flag
        self.optimization_algorithm_flag = optimization_algorithm_flag
        self.wavelength_interpolation_flag = wavelength_interpolation_flag
        self.lsf_shape_flag = lsf_shape_flag
        self.use_direct_access = use_direct_access
        self.n_threads = n_threads
        self.input_weights_path = input_weights_path
        self.input_lsf_path = input_lsf_path
        self.ferre_kwds = ferre_kwds
        self.slurm_kwds = slurm_kwds
        self.analyze_individual_visits = analyze_individual_visits
        return None
    

    def get_partial_task_instances_in_database(self, context):
        """
        Look upstream in the DAG and get the primary keys of the task instances 
        that were created in the database and to be executed by FERRE.

        :param context:
            The context of the task instance in the DAG.
        """
        task, ti, params = (context["task"], context["ti"], context["params"])

        pks = deserialize_pks(self.pks)

        log.info(f"In {task} {ti} and there are {len(pks)} upstream primary keys")
        
        q = session.query(astradb.TaskInstance)\
                   .join(astradb.TaskInstanceParameter)\
                   .join(astradb.Parameter)
        q = q.filter(astradb.TaskInstance.pk.in_(pks))
        q = q.filter(and_(
            astradb.Parameter.parameter_name == "header_path",
            astradb.Parameter.parameter_value == self.header_path
        ))
        
        r = q.all()
        log.info(f"In {task} {ti} and there are now {len(r)} relevant primary keys")

        return r
        
    def prepare_observations(
            self,
            instances,
            parameter_names,
            grid_mid_point,
            analyze_individual_visits,
            wavelength_interpolation_flag
        ):
        
        # Load the observations from the database instances.
        trees = {}
        wavelength, flux, sigma, initial_parameters, names, snrs = ([], [], [], [], [], [])
        
        Ns = np.zeros(len(instances), dtype=int)

        data_slice = slice(None) if analyze_individual_visits else slice(0, 1)

        for i, instance in enumerate(instances):
            release = instance.parameters["release"]
            try:
                tree = trees[release]                
            except KeyError:
                tree = trees[release] = SDSSPath(release=release)
            
            path = tree.full(**instance.parameters)

            try:
                # TODO: Profile this.
                spectrum = Spectrum1D.read(path, data_slice=data_slice)

                N, P = spectrum.flux.shape
                flux.append(spectrum.flux.value)
                sigma.append(spectrum.uncertainty.array**-0.5)
                if wavelength_interpolation_flag > 0:
                    wavelength.append(
                        np.tile(spectrum.wavelength.value, N).reshape((N, -1))
                    )

                point = np.tile(grid_mid_point, N).reshape((N, -1))

                for j, parameter_name in enumerate(parameter_names):
                    v = instance.parameters.get(f"initial_{parameter_name.lower().replace(' ', '_')}", None)
                    if v is not None:
                        point[:, j] = float(v)

                names.extend([f"{i}_{instance.parameters['telescope']}_{instance.parameters['obj']}_{j}" for j in range(N)])
                initial_parameters.append(point)

                snrs.append(spectrum.meta["snr"][data_slice])

            except:
                log.exception(f"Exception in trying to load data product associated with {instance}")
                raise 

            else:
                Ns[i] = N
        
        if wavelength_interpolation_flag > 0:
            wavelength = np.vstack(wavelength)
        else:
            wavelength = spectrum.wavelength.value.copy()

        snrs = np.vstack(snrs).flatten()
        flux = np.vstack(flux)
        sigma = np.vstack(sigma)
        initial_parameters = np.vstack(initial_parameters)

        bad = ~np.isfinite(flux) + ~np.isfinite(sigma) + (sigma == 0)
        flux[bad] = 1.0
        sigma[bad] = 1e6

        return (wavelength, flux, sigma, initial_parameters, names, snrs, Ns)

    

    def execute(self, context, instances=None):
        
        # The upstream task has created an instance in the database for this FERRE task, 
        # which includes the observation source parameters, etc.
        if instances is None:
            instances = self.get_partial_task_instances_in_database(context)

        log.info(f"Got instances {instances}")

        # Create all the files.
        ferre = Ferre(
            os.path.expandvars(self.header_path),
            frozen_parameters=self.frozen_parameters,
            interpolation_order=self.interpolation_order,
            init_flag=self.init_flag,
            init_algorithm_flag=self.init_algorithm_flag,
            error_algorithm_flag=self.error_algorithm_flag,
            continuum_flag=self.continuum_flag,
            continuum_order=self.continuum_order,
            continuum_reject=self.continuum_reject,
            continuum_observations_flag=self.continuum_observations_flag,
            optimization_algorithm_flag=self.optimization_algorithm_flag,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag,
            lsf_shape_flag=self.lsf_shape_flag,
            n_threads=self.n_threads,
            use_direct_access=self.use_direct_access,
            input_weights_path=self.input_weights_path,
            input_lsf_path=self.input_lsf_path,
            ferre_kwds=self.ferre_kwds
        )

        # Create some directory.
        log.info(f"Preparing to run FERRE in {ferre.directory}")

        ferre._setup()
        ferre._write_ferre_input_file()

        wavelength, flux, sigma, initial_parameters, names, snr, Ns = self.prepare_observations(
            instances=instances,
            parameter_names=ferre.parameter_names,
            grid_mid_point=ferre.grid_mid_point,
            analyze_individual_visits=self.analyze_individual_visits,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag
        )

        ferre.write_inputs(wavelength, flux, sigma, initial_parameters, names)
        
        # Execute FERRE.
        # TODO: Slurm it and/or batch it
        ferre._execute(total=int(sum(Ns)))

        # Parse outputs.
        output_names, p_opt, p_opt_err, meta = ferre.parse_outputs(full_output=True)

        # TODO: Update tasks to include other relevant FERRE keywords.

        # Write FERRE outputs to database.
        _parameter_names = []
        common_kwds = {}
        frozen_parameters = self.frozen_parameters or dict()
        for parameter_name in ferre.parameter_names:
            sanitised_parameter_name = sanitise_parameter_name(parameter_name)
            common_kwds[f"frozen_{sanitised_parameter_name}"] = frozen_parameters.get(parameter_name, False)
            _parameter_names.append(sanitised_parameter_name)

        # TODO: Assuming FERRE outputs are ordered the same as inputs.
        si = 0
        for i, (instance, N) in enumerate(zip(instances, Ns)):
            if N == 0: continue

            if N > 1:
                print(f"MULTIPLE OBSERVATIONS {instance} {N}")

            sliced = slice(si, si + N)

            results = dict(
                snr=list(snr[sliced]),
                log_snr_sq=list(meta["log_snr_sq"][sliced]),
                log_chisq_fit=list(meta["log_chisq_fit"][sliced])
            )
            # Initial values.
            for j, pn in enumerate(_parameter_names):
                results[f"initial_{pn}"] = list(initial_parameters[sliced][:, j]) # Initial.
                results[pn] = list(p_opt[sliced][:, j]) # Optimized
                results[f"u_{pn}"] = list(p_opt_err[sliced][:, j]) # errors.

            results.update(common_kwds) # Add frozen parameters.

            # Create result.
            _instance, output = create_task_output(
                instance,
                astradb.Ferre,
                **results
            )
            si += N

            log.info(f"Created output {output} for instance {_instance}")
        
        # Teardown
        ferre.teardown()

        return [instance.pk for instance in instances]



