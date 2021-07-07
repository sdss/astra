from airflow.models.baseoperator import BaseOperator

import os
from typing import Dict, Optional


from airflow.exceptions import AirflowSkipException
from airflow.hooks.subprocess import SubprocessHook
from airflow.models import BaseOperator
from airflow.utils.operator_helpers import context_to_airflow_vars
from sdss_access.path import SDSSPath

from astropy.time import Time
from astropy.io.fits import getheader

from astra.contrib.ferre.core import Ferre
from astra.contrib.ferre.utils import (
    parse_grid_information, safe_read_header, approximate_log10_microturbulence, yield_suitable_grids
)
from astra.utils import log
from astra.database import astradb, apogee_drpdb, catalogdb, session
from astra.database.utils import create_task_instance_in_database


def get_sdss5_apstar_daily_observations(mjd, min_ngoodrvs=1):
    
    release, data_model_name = ("sdss5", "apStar")
    columns = (
        apogee_drpdb.Star.apred_vers.label("apred"), # TODO: Raise with Nidever
        apogee_drpdb.Star.healpix,
        apogee_drpdb.Star.telescope,
        apogee_drpdb.Star.apogee_id.label("obj"), # TODO: Raise with Nidever
    )

    q = session.query(*columns).distinct(*columns)
    q = q.filter(apogee_drpdb.Star.mjdend == mjd)\
         .filter(apogee_drpdb.Star.ngoodrvs >= min_ngoodrvs)

    rows = q.all()
    keys = [column.name for column in columns]

    data_model_kwds = [dict(zip(keys, values)) for values in rows]

    return (release, data_model_name, data_model_kwds)


def get_sdss4_apstar_observations(**kwargs):

    # We need; 'apstar', 'apred', 'obj', 'telescope', 'field', 'prefix'
    release, data_model_name = ("DR16", "apStar")
    columns = (
        catalogdb.SDSSDR16ApogeeStar.apogee_id.label("obj"),
        catalogdb.SDSSDR16ApogeeStar.field,
        catalogdb.SDSSDR16ApogeeStar.telescope,
        catalogdb.SDSSDR16ApogeeStar.apstar_version.label("apstar"),
        catalogdb.SDSSDR16ApogeeStar.file, # for prefix and apred
    )
    q = session.query(*columns).distinct(*columns)
    # TODO: remove this limit. just for testing.
    q = q.limit(100)
    if kwargs:
        q = q.filter(**kwargs)
    
    data_model_kwds = []
    for obj, field, telescope, apstar, filename in q.all():

        prefix = filename[:2]
        apred = filename.split("-")[1]

        data_model_kwds.append(dict(
            obj=obj,
            field=field,
            telescope=telescope,
            apstar=apstar,
            prefix=prefix,
            apred=apred
        ))

    return (release, data_model_name, data_model_kwds)



def yield_initial_guess_from_doppler_headers(
        release, 
        data_model_name, 
        all_data_model_kwds, 
        public=False
    ):
    """
    Get initial guesses for the sources provided.
    """

    # Get the observation parameters from the upstream task.
    # TODO: Include photometry code into this task, because we need the telescope and mean fiber
    #       to compare against grids.

    tree = SDSSPath(release=release, public=public)

    for data_model_kwds in all_data_model_kwds:
        try:
            path = tree.full(data_model_name, **data_model_kwds)

            header = getheader(path)

            teff = safe_read_header(header, ("RV_TEFF", "RVTEFF"))
            logg = safe_read_header(header, ("RV_LOGG", "RVLOGG"))
            fe_h = safe_read_header(header, ("RV_FEH", "RVFEH"))

            # Get information relevant for matching initial guess and grids.
            initial_guess = dict(
                telescope=data_model_kwds["telescope"], # important for LSF information
                mean_fiber=header["MEANFIB"], # important for LSF information
                teff=teff,
                logg=logg,
                metals=fe_h,
            )

        except Exception as exception:
            log.exception(f"Exception: {exception}")
            continue

        else:
            yield (data_model_kwds, initial_guess)



def _create_partial_ferre_task_instances_from_observations(
        release,
        data_model_name,
        data_model_kwds,
        task,
        params,
        **kwargs
    ):

    # Get grid information.
    grid_info = parse_grid_information(params["ferre_header_paths"])

    # Get initial guesses.
    initial_guesses = yield_initial_guess_from_doppler_headers(
        release, 
        data_model_name, 
        all_data_model_kwds
    )

    # Match observations, initial guesses, and FERRE header files.
    instance_meta = []
    for data_model_kwds, initial_guess in initial_guesses:
        for header_path, ferre_headers in yield_suitable_grids(grid_info, **initial_guess):
            instance_meta.append(dict(
                data_model_name=data_model_name,
                header_path=header_path,
                # Add the initial guess information.
                initial_teff=initial_guess["teff"],
                initial_logg=initial_guess["logg"],
                initial_metals=initial_guess["metals"],
                initial_log10vdop=approximate_log10_microturbulence(initial_guess["logg"]),
                initial_o_mg_si_s_ca_ti=0.0,
                initial_lgvsini=0.0,
                initial_c=0.0,
                initial_n=0.0,
                # Add the data model keywords.
                **data_model_kwds
            ))

    # Create task instances.
    pks = []
    for meta in instance_meta:
        # The task ID is temporary, and will be over-written when the next operator uses it.
        instance = create_task_instance_in_database(task.task_id, **meta)
        pks.append(instance.pk)
    
    # Return the primary keys, which will be passed on to the next task.
    return pks
    

def create_partial_ferre_task_instances_from_sdss4_apstar_observations(ds, task, params, **kwargs):
    """
    Query the database for SDSS-4 ApStar observations and create partial task instances with
    their observation keywords, initial guesses, and other FERRE requirements.

    This function will return primary key values in the `astra.ti` table.

    :param ds:
        Date start of the task.
    
    :param task:
        The Apache Airflow task.
    
    :params:
        The parameters given to this task in Apache Airflow.
    """
    release, data_model_name, all_data_model_kwds = get_sdss4_apstar_observations()

    # Create the task instances.
    return _create_partial_ferre_task_instances_from_observations(
        release, 
        data_model_name,
        data_model_kwds, 
        task,
        params,
        **kwargs
    )

def create_partial_ferre_task_instances_from_sdss5_apstar_observations(ds, task, params, **kwargs):
    """
    Query the database for SDSS-V ApStar observations taken on the date start, and create
    partial task instances with their observation keywords, initial guesses, and other FERRE
    requirements.

    This function will return primary key values in the `astra.ti` table.

    :param ds:
        Date start of the task.
    
    :param task:
        The Apache Airflow task.
    
    :params:
        The parameters given to this task in Apache Airflow.
    """

    # TODO: Here we are assuming a "@daily" interval schedule. We should consider how to incorporate
    #       the schedule interval.
    mjd = Time(ds).mjd

    # Get the identifiers for the APOGEE observations taken on this MJD.
    release, data_model_name, all_data_model_kwds = get_sdss5_apstar_daily_observations(mjd)

    # Create the task instances.
    return _create_partial_ferre_task_instances_from_observations(
        release, 
        data_model_name,
        data_model_kwds, 
        task,
        params,
        **kwargs
    )


def choose_which_ferre_tasks_to_execute(func_task_name_from_path, task, **kwargs):
    """
    A function to be used with BranchPythonOperator that selects the FERRE tasks that
    need executing, based on the FERRE header paths in partial task instances.
    
    :param func_task_name_from_path:
        A function that takes a header path and returns the name of the task.
    
    :param task:
        The Apache Airflow task.
    """
    # Get primary keys from immediate upstream task.
    pks = ti.xcom_pull(task_ids=task.upstream_list[0].task_id)

    # Get the header paths for those task instances.
    q = session.query(astradb.Parameter.parameter_value).distinct(astradb.Parameter.parameter_value)
    q = q.filter(or_(*(astradb.TaskInstance.pk == pk for pk in pks)))
    q = q.filter(astradb.Parameter.parameter_name == "header_path")
    
    rows = q.all()
    print(f"Retrieved {rows}")

    return [func_task_name_from_path(header_path) for header_path, in rows]





class FerreOperator(BaseOperator):

    def __init__(
        self,
        *,
        header_path: str,
        frozen_parameters: Optional[Dict[str, str]] = None,
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
        return None
    

    def get_partial_task_instance_in_database(self, context):
        task, ti, params = (context["task"], context["ti"], context["params"])

        pks = ti.xcom_pull(task_ids="create_partial_ferre_task_instances")

        q = session.query(astradb.TaskInstance).filter(or_(*(
            astradb.TaskInstance.pk == pk for pk in pks
        )))

        for instance in q.all():
            if instance.parameters["header_path"] == self.header_path:
                return instance
        else:
            raise ValueError(f"no partial instance found matching header path {self.header_path} from pks ({pks})")

        


    def execute(self, context):
        
        # The upstream task has created an instance in the database for this FERRE task, 
        # which includes the observation source parameters, etc.

        instance = self.get_partial_task_instance_in_database()

        print(f"Got instance {instance}")

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
        print(f"Preparing to run FERRE in {ferre.directory}")

        ferre._setup()
        ferre._write_ferre_input_file()

        # Load the observations.
        tree = SDSSPath(release=params["release"], public=params["public"])

        wavelength = []
        flux = []
        uncertainties = []
        
        for (data_model_name, data_model_kwds, initial_guess) in inputs:

            path = tree.full(data_model_name, **data_model_kwds)

            #spectrum = Spectrum1D.read(path)




        # Teardown
        ferre.teardown()
        
        
        '''

        log.debug(f"Preparing inputs for FERRE in {self.directory}")
        flux = np.vstack([spectrum.flux.value for spectrum in spectra])
        uncertainties = np.vstack([spectrum.uncertainty.array**-0.5 for spectrum in spectra])
        Ns = np.array([spectrum.flux.shape[0] for spectrum in spectra])
        wavelengths = np.vstack([
            np.tile(spectrum.wavelength.value, n).reshape((n, -1)) \
            for (spectrum, n) in zip(spectra, Ns)
        ])
        
        N, P = flux.shape
        assert flux.shape == uncertainties.shape
        
        log.debug(f"Parsing initial parameters in {self.directory}")
        parsed_initial_parameters = self.parse_initial_parameters(initial_parameters, Ns)

        # Make sure we are not sending nans etc.
        bad = ~np.isfinite(flux) \
            + ~np.isfinite(uncertainties) \
            + (uncertainties == 0)
        flux[bad] = 1.0
        uncertainties[bad] = 1e6

        # We only send specific set of pixels to FERRE.
        mask = self.wavelength_mask(wavelengths[0])

        # TODO: Should we be doing this if we are using wavelength_interpolation_flag > 0?
        #wavelengths = wavelengths[:, mask]
        #flux = flux[:, mask]
        #uncertainties = uncertainties[:, mask]

        # Write wavelengths?
        log.debug(f"Writing input files for FERRE in {self.directory}")
        if self.kwds["wavelength_interpolation_flag"] > 0:            
            utils.write_data_file(
                wavelengths[:, mask],
                os.path.join(self.directory, self.kwds["input_wavelength_path"])
            )

        # Write flux.
        log.debug(f"Writing input fluxes in {self.directory}")
        utils.write_data_file(
            flux[:, mask],
            os.path.join(self.directory, self.kwds["input_flux_path"])
        )

        # Write uncertainties.
        log.debug(f"Writing input uncertainties in {self.directory}")
        utils.write_data_file(
            uncertainties[:, mask],
            os.path.join(self.directory, self.kwds["input_uncertainties_path"])
        )

        # Write initial parameters to disk.
        log.debug(f"Writing input names and parameters in {self.directory}")
        if names is None:
            names = [f"idx_{i:.0f}" for i in range(len(parsed_initial_parameters))]
        
        else:
            # Expand out names if needed.
            if len(names) == len(spectra) and sum(Ns) > len(spectra):
                expanded_names = []
                for i, (name, n) in enumerate(zip(names, Ns)):
                    for j in range(n):
                        expanded_names.append(f"{name}_{j}")
                names = expanded_names

        with open(os.path.join(self.directory, self.kwds["input_parameter_path"]), "w") as fp:            
            #for i, each in enumerate(parsed_initial_parameters):
            #    fp.write(utils.format_ferre_input_parameters(*each, star_name=f"idx_{i:.0f}"))
            for star_name, ip in zip(names, parsed_initial_parameters):
                fp.write(utils.format_ferre_input_parameters(*ip, star_name=star_name))

        '''

