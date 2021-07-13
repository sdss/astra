from airflow.models.baseoperator import BaseOperator

import os
from typing import Dict, Optional


from airflow.exceptions import AirflowSkipException
from airflow.hooks.subprocess import SubprocessHook
from airflow.models import BaseOperator
from airflow.utils.operator_helpers import context_to_airflow_vars
from sdss_access import SDSSPath

from sqlalchemy import or_, and_


from astropy.time import Time
from astropy.io.fits import getheader

from astra.contrib.ferre.core import Ferre
from astra.contrib.ferre.utils import (
    parse_grid_information, safe_read_header, approximate_log10_microturbulence, yield_suitable_grids
)
from astra.utils import log
from astra.database import astradb, apogee_drpdb, catalogdb, session
from astra.database.utils import (
    get_or_create_task_instance, 
    get_sdss4_apstar_kwds,
    get_sdss5_apstar_kwds,
)



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

            teff = safe_read_header(header, ("RV_TEFF", "RVTEFF"))
            logg = safe_read_header(header, ("RV_LOGG", "RVLOGG"))
            fe_h = safe_read_header(header, ("RV_FEH", "RVFEH"))

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
    grid_info = parse_grid_information(ferre_header_paths)

    # Get initial guesses.
    initial_guesses = yield_initial_guess_from_doppler_headers(data_model_kwds)
    
    # Match observations, initial guesses, and FERRE header files.
    instance_meta = []
    for kwds, initial_guess in initial_guesses:
        for header_path, ferre_headers in yield_suitable_grids(grid_info, **initial_guess):
            instance_meta.append(dict(
                header_path=header_path,
                # Add the initial guess information.
                initial_teff=np.round(initial_guess["teff"], 0),
                initial_logg=np.round(initial_guess["logg"], 2),
                initial_metals=np.round(initial_guess["metals"], 2),
                initial_log10vdop=np.round(approximate_log10_microturbulence(initial_guess["logg"]), 2),
                initial_o_mg_si_s_ca_ti=0.0,
                initial_lgvsini=0.0,
                initial_c=0.0,
                initial_n=0.0,
                # Add the data model keywords.
                **kwds
            ))

    # Create task instances.
    pks = []
    for meta in instance_meta:
        # The task ID is temporary, and will be over-written when the next operator uses it.
        instance = get_or_create_task_instance(
            dag_id,
            task_id_function(**meta),
            **meta
        )
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

def create_task_instances_for_sdss5_apstars(mjd, dag_id, task_id_function, ferre_header_paths, **kwargs):
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
    ''''

    # Get the identifiers for the APOGEE observations taken on this MJD.
    data_model_kwds = get_sdss5_apstar_kwds(ds)
    
    # Create the task instances.
    return _create_partial_ferre_task_instances_from_observations(
        dag_id=dag_id,
        task_id_function=task_id_function,
        data_model_kwds=data_model_kwds,
        ferre_header_paths=ferre_header_paths,
        **kwargs
    )


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
    pks = deserialize_pks(pks)

    # Get the header paths for those task instances.
    q = session.query(astradb.Parameter.parameter_value)\
                .join(astradb.TaskInstanceParameter)\
                .join(astradb.TaskInstance)\
                .distinct(astradb.Parameter.parameter_value)\
                .filter(astradb.TaskInstance.pk.in_(pks))\
                .filter(astradb.Parameter.parameter_name == "header_path")

    rows = q.all()
    print(f"Retrieved {rows}")
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
    best_results = {}
    for pk in pks:
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk==pk)
        instance = q.one_or_none()

        p = instance.parameters
        try:
            tree = trees[p["release"]]                
        except KeyError:
            tree = trees[p["release"]] = SDSSPath(release=p["release"])
        
        key = "_".join([
            p['release'],
            p['filetype'],
            *[getattr(p, k) for k in tree.lookup_keys(p['filetype'])]
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
    
    return [pk for (log_chisq_fit, pk) in best_tasks.values()]



class FerreOperator(BaseOperator):

    def __init__(
        self,
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
        slurm_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
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
        self.slurm_kwargs = slurm_kwargs
        return None
    

    def get_partial_task_instances_in_database(self, context):
        """
        Look upstream in the DAG and get the primary keys of the task instances 
        that were created in the database and to be executed by FERRE.

        :param context:
            The context of the task instance in the DAG.
        """
        task, ti, params = (context["task"], context["ti"], context["params"])

        pks = ti.xcom_pull(task_ids="create_partial_ferre_task_instances")

        q = session.query(astradb.TaskInstance)\
                   .join(astradb.TaskInstanceParameter)\
                   .join(astradb.Parameter)
        q = q.filter(astradb.TaskInstance.pk.in_(pks))
        q = q.filter(and_(
            astradb.Parameter.parameter_name == "header_path",
            astradb.Parameter.parameter_value == self.header_path
        ))
        
        return q.all()
        

    def execute(self, context, instances=None):
        
        # The upstream task has created an instance in the database for this FERRE task, 
        # which includes the observation source parameters, etc.
        if instances is None:
            instances = self.get_partial_task_instances_in_database()

        log.info(f"Got instances {instances}")

        # Create all the files.
        ferre = Ferre(
            self.header_path,
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

        # Load the observations from the database instances.
        trees = {}
        wavelength, flux, sigma, initial_parameters, names = ([], [], [], [], [])
        
        Ns = np.zeros(len(instances))

        for i, instance in enumerate(instances):
            release = instance.parameters["release"]
            try:
                tree = trees[release]                
            except KeyError:
                tree = trees[release] = SDSSPath(release=release)
            
            path = tree.full(
                instance.parameters["data_model_name"],
                **instance.parameters
            )

            try:
                # TODO: Profile this.
                spectrum = Spectrum1D.read(path)

                N, P = spectrum.flux.shape
                flux.append(spectrum.flux.value)
                sigma.append(spectrum.uncertainty.array**-0.5)
                if self.wavelength_interpolation_flag > 0:
                    wavelength.append(
                        np.tile(spectrum.wavelength.value, N).reshape((N, -1))
                    )

    
                point = np.tile(ferre.grid_mid_point, N)
                for j, parameter_name in enumerate(ferre.parameter_names):
                    v = instance.parameters.get(f"initial_{parameter_name.lower().replace(' ', '_')}", None)
                    if v is not None:
                        point[:, j] = float(v)

                names.extend([f"{i}_{instance.parameters['obj']}_{j}" for j in range(N)])
                initial_parameters.append(point)

            except:
                log.exception(f"Exception in trying to load data product associated with {instance}")
            
            else:
                Ns[i] = N
    
    
        flux = np.vstack(flux)
        sigma = np.vstack(sigma)
        wavelength = np.vstack(wavelength)
        initial_parameters = np.vstack(initial_parameters)

        bad = ~np.isfinite(flux) + ~np.isfinite(sigma) + (sigma == 0)
        flux[bad] = 1.0
        sigma[bad] = 1e6

        # TODO: Assuming all have the same wavelength grid.
        mask = ferre.wavelength_mask(spectrum.wavelength.value)

        # Write wavelengths
        log.debug(f"Writing input files for FERRE in {ferre.directory}")
        if self.wavelength_interpolation_flag > 0:
            utils.write_data_file(
                wavelengths[:, mask],
                os.path.join(ferre.directory, ferre.kwds["input_wavelength_path"])
            )

        # Write flux.
        log.debug(f"Writing input fluxes in {ferre.directory}")
        utils.write_data_file(
            flux[:, mask],
            os.path.join(ferre.directory, ferre.kwds["input_flux_path"])
        )

        # Write uncertainties.
        log.debug(f"Writing input uncertainties in {ferre.directory}")
        utils.write_data_file(
            sigma[:, mask],
            os.path.join(ferre.directory, ferre.kwds["input_uncertainties_path"])
        )

        # Write initial values.
        with open(os.path.join(ferre.directory, ferre.kwds["input_parameter_path"]), "w") as fp:
            for name, point in zip(names, initial_parameters):
                fp.write(utils.format_ferre_input_parameters(*point, star_name=star_name))

        # Execute FERRE.

        # Take ownership of the instance names.
        # Write FERRE outputs to database.



        # Teardown
        #ferre.teardown()



