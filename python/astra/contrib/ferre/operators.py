import os
import json
import numpy as np
import pickle
from airflow.operators.python import BranchPythonOperator
from sdss_access import SDSSPath
from ast import literal_eval

from astra.database import astradb, session
from astra.database.utils import create_task_output
from astra.contrib.ferre import utils
from astra.contrib.ferre.core import (prepare_ferre, parse_ferre_outputs)
from astra.operators.sdss_data_product import DataProductOperator
from astra.utils import flatten, get_base_output_path, log


class FerreOperator(DataProductOperator):

    """
    An operator to execute FERRE on some data products.

    :param header_path:
        The path of the FERRE header file.
        
    :param frozen_parameters: [optional]
        A dictionary with parameter names (as per the header file) as keys, and either
        a boolean flag or a float as value. If boolean `True` is given for a parameter,
        then the value will be fixed at the initial value per spectrum. If a float is
        given then this value will supercede all initial values given, fixing the
        dimension for all input spectra regardless of the initial value.

    :param interpolation_order: [optional]
        Order of interpolation to use (default: 1, as per FERRE).
        This corresponds to the FERRE keyword `inter`.

        0. nearest neighbour
        1. linear
        2. quadratic Bezier
        3. cubic Bezier
        4. cubic splines

    :param input_weights_path: [optional]
        The location of a weight (or mask) file to apply to the pixels. This corresponds
        to the FERRE keyword `filterfile`.
    
    :para input_lsf_shape_path: [optional]
        The location of a file containing describing the line spread function to apply to
        the observations. This keyword is ignored if `lsf_shape_flag` is anything but 0.
        This corresponds to the FERRE keyword `lsffile`.
    
    :param lsf_shape_flag: [optional]
        A flag indicating what line spread convolution to perform. This should be one of:

        0. no LSF convolution (default)
        1. 1D (independent of wavelength), one and the same for all spectra
        2. 2D (a function of wavelength), one and the same for all
        3. 1D and Gaussian  (i.e. described by a single parameter, its width), one for all objects
        4. 2D and Gaussian, one for all objects
        11. 1D and particular for each spectrum
        12. 2D and particular for each spectrum
        13. 1D Gaussian, but particular for each spectrum
        14. 2D Gaussian and particular for each object.

        If `lsf_shape_flag` is anything but 0, then an `input_lsf_path` keyword argument
        will also be required, pointing to the location of the LSF file.

    :param error_algorithm_flag: [optional]
        Choice of algorithm to compute error bars (default: 1, as per FERRE).
        This corresponds to the FERRE keyword `errbar`.

        0. To adopt the distance from the solution at which $\chi^2$ = min($\chi^2$) + 1
        1. To invert the curvature matrix
        2. Perform numerical experiments injecting noise into the data

    :param wavelength_interpolation_flag: [optional]
        Flag to indicate what to do about wavelength interpolation (default: 0).
        This is not usually needed as the FERRE grids are computed on the resampled
        APOGEE grid. This corresponds to the FERRE keyword `winter`.

        0. No interpolation.
        1. Interpolate observations.
        2. The FERRE documentation says 'Interpolate fluxes', but it is not clear to the
           writer how that is any different from Option 1.

    :param optimization_algorithm_flag: [optional]
        Integer flag to indicate which optimization algorithm to use:

        1. Nelder-Mead
        2. Boender-Timmer-Rinnoy Kan
        3. Powell's truncated Newton method
        4. Nash's truncated Newton method

    :param continuum_flag: [optional]
        Choice of algorithm to use for continuum fitting (default: 1).
        This corresponds to the FERRE keyword `cont`, and is related to the
        FERRE keywords `ncont` and `rejectcont`.

        If `None` is supplied then no continuum keywords will be given to FERRE.

        1. Polynomial fitting using an iterative sigma clipping algrithm (set by
           `continuum_order` and `continuum_reject` keywords).
        2. Segmented normalization, where the data are split into `continuum_segment`
           segments, and the values in each are divided by their mean values.
        3. The input data are divided by a running mean computed with a window of
           `continuum_segment` pixels.
    
    :param continuum_order: [optional]
        The order of polynomial fitting to use, if `continuum_flag` is 1.
        This corresponds to the FERRE keyword `ncont`, if `continuum_flag` is 1.
        If `continuum_flag` is not 1, this keyword argument is ignored.
    
    :param continuum_segment: [optional]
        Either the number of segments to split the data into for performing normalization,
        (e.g., when `continuum_flag` = 2), or the window size to use when `continuum_flag`
        = 3. This corresponds to the FERRE keyword `ncont` if `continuum_flag` is 2 or 3.
        If `continuum_flag` is not 2 or 3, this keyword argument is ignored.

    :param continuum_reject: [optional]
        When using polynomial fitting with an iterative sigma clipping algorithm
        (`continuum_flag` = 1), this sets the relative error where data points will be
        excluded. Any data points with relative errors larger than `continuum_reject`
        will be excluded. This corresponds to the FERRE keyword `rejectcont`.
        If `continuum_flag` is not 1, this keyword argument is ignored.
    
    :param continuum_observations_flag: [optional]
        This corresponds to the FERRE keyword `obscont`. Nothing is written down in the
        FERRE documentation about this keyword.
    
    :param full_covariance: [optional]
        Return the full covariance matrix from FERRE (default: True). 
        This corresponds to the FERRE keyword `covprint`.
    
    :param pca_project: [optional]
        Use Principal Component Analysis to compress the spectra (default: False).
        This corresponds to the FERRE keyword `pcaproject`.
    
    :param pca_chi: [optional]
        Use Principal Component Analysis to compress the spectra when calculating the
        $\chi^2$ statistic. This corresponds to the FERRE keyword `pcachi`.
    
    :param n_threads: [optional]
        The number of threads to use for FERRE. This corresponds to the FERRE keyword
        `nthreads`.

    :param f_access: [optional]
        If `False`, load the entire grid into memory. If `True`, run the interpolation 
        without loading the entire grid into memory -- this is useful for small numbers 
        of interpolation. If `None` (default), automatically determine which is faster.
        This corresponds to the FERRE keyword `f_access`.
    
    :param f_format: [optional]
        File format of the FERRE grid: 0 (ASCII) or 1 (UNF format, default).
        This corresponds to the FERRE keyword `f_format`.

    :param ferre_kwargs: [optional]
        A dictionary of options to apply directly to FERRE, which will over-ride other
        settings supplied here, so use with caution.    
    """

    def __init__(
        self,
        header_path,
        frozen_parameters=None,
        interpolation_order=3,
        input_weights_path=None,
        input_lsf_shape_path=None,
        lsf_shape_flag=0,
        error_algorithm_flag=1,
        wavelength_interpolation_flag=0,
        optimization_algorithm_flag=3,
        continuum_flag=1,
        continuum_order=4,
        continuum_segment=None,
        continuum_reject=0.3,
        continuum_observations_flag=1,
        full_covariance=False,
        pca_project=False,
        pca_chi=False,
        n_threads=32,
        f_access=None,
        f_format=1,
        ferre_kwargs=None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)        
        self.header_path = header_path
        self.frozen_parameters = frozen_parameters
        self.interpolation_order = interpolation_order
        self.input_weights_path = input_weights_path
        self.input_lsf_shape_path = input_lsf_shape_path
        self.lsf_shape_flag = lsf_shape_flag
        self.error_algorithm_flag = error_algorithm_flag
        self.wavelength_interpolation_flag = wavelength_interpolation_flag
        self.optimization_algorithm_flag = optimization_algorithm_flag
        self.continuum_flag = continuum_flag
        self.continuum_order = continuum_order
        self.continuum_segment = continuum_segment
        self.continuum_reject = continuum_reject
        self.continuum_observations_flag = continuum_observations_flag
        self.full_covariance = full_covariance
        self.pca_project = pca_project
        self.pca_chi = pca_chi
        self.n_threads = n_threads
        self.f_access = f_access
        self.f_format = f_format
        self.ferre_kwargs = ferre_kwargs
    

    def data_model_identifiers(self, context):
        """
        Yield data model identifiers from upstream that match this operator's
        header path.
        """

        pks, task = ([], context["task"])
        while True:
            for upstream_task in task.upstream_list:
                log.debug(f"Considering {upstream_task}")
                if isinstance(upstream_task, BranchPythonOperator):
                    # Jump over branch operators
                    log.debug(f"Jumping over BranchPythonOperator {upstream_task}")
                    task = upstream_task
                    break

                log.debug(f"Using upstream results from {upstream_task}")
                pks.extend(context["ti"].xcom_pull(task_ids=upstream_task.task_id))
            else:
                break
        
        pks = flatten(pks)
        if not pks:
            raise RuntimeError(f"No upstream primary keys identified.")
            
        log.debug(f"From pks: {pks}")
        log.debug(f"That also match {self.header_path}")
        
        # Restrict to primary keys that have the same header path.
        q = session.query(astradb.TaskInstanceParameter.ti_pk)\
                   .distinct(astradb.TaskInstanceParameter.ti_pk)\
                   .join(astradb.Parameter, 
                         astradb.TaskInstanceParameter.parameter_pk == astradb.Parameter.pk)\
                   .filter(astradb.Parameter.parameter_name == "header_path")\
                   .filter(astradb.Parameter.parameter_value == self.header_path)\
                   .filter(astradb.TaskInstanceParameter.ti_pk.in_(pks))
        
        log.debug(f"Restricting to primary keys:")

        first_or_none = lambda item: None if item is None else item[0]
        callables = [
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
        for pk, in q.all():
            q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk)
            instance = q.one_or_none()
            log.debug(f"{instance} with {instance.output}")

            release = instance.parameters["release"]
            filetype = instance.parameters["filetype"]

            parameters = dict(
                release=release,
                filetype=filetype
            )

            tree = trees.get(release, None)
            if tree is None:
                tree = trees[release] = SDSSPath(release=release)
            
            for key in tree.lookup_keys(filetype):
                parameters[key] = instance.parameters[key]

            # What other information should we pass on?
            if instance.output is None:
                # Only pass on the data model identifiers, and any initial values.
                # Let everything else be specified in this operator
                for key, callable in callables:
                    parameters[key] = instance.parameters[key]

            else:
                # There is an upstream FerreOperator.
                log.debug(f"Taking previous result in {pk} as initial result here")

                # Take final teff/logg/etc as the initial values for this task.
                # TODO: Query whether we should be taking first or none, because if
                #       we are running all visits we may want to use individual visit
                #       results from the previous iteration
                for key, callable in callables:
                    parameters[key] = callable(instance)
                
            # Store upstream primary key as a parameter, too.
            # We could decide not to do this, but it makes it much easier to find
            # upstream tasks.
            parameters.setdefault("upstream_pk", [])
            if "upstream_pk" in instance.parameters:
                try:
                    upstream_pk = literal_eval(instance.parameters["upstream_pk"])
                    parameters["upstream_pk"].extend(upstream_pk)
                except:
                    log.exception(f"Cannot add upstream primary keys from {instance}: {instance.parameters['upstream_pk']}")
                
            parameters["upstream_pk"].append(pk)

            yield parameters                


    def execute(self, context):
        """
        Execute the operator.

        :param context:
            The Airflow DAG context.
        """

        # Load spectra.
        instances, Ns = ([], [])
        wavelength, flux, sigma, spectrum_meta = ([], [], [], [])
        for instance, path, spectrum in self.prepare_data():
            if spectrum is None: continue

            N, P = spectrum.flux.shape
            wavelength.append(np.tile(spectrum.wavelength.value, N).reshape((N, -1)))
            flux.append(spectrum.flux.value)
            sigma.append(spectrum.uncertainty.array**-0.5)
            spectrum_meta.append(dict(snr=spectrum.meta["snr"]))

            Ns.append(N)
            instances.append(instance)

        Ns = np.array(Ns, dtype=int)
        wavelength, flux, sigma = tuple(map(np.vstack, (wavelength, flux, sigma)))

        # Create names for easy debugging in FERRE outputs.
        names = create_names(instances, Ns, "{star_index}_{telescope}_{obj}_{spectrum_index}")

        # Load initial parameters, taking account 
        initial_parameters = create_initial_parameters(instances, Ns)

        # Directory.
        directory = os.path.join(
            get_base_output_path(),
            "ferre",
            "tasks",
            f"{context['ds']}-{context['dag'].dag_id}-{context['task'].task_id}-{context['run_id']}"
        )
        os.makedirs(directory, exist_ok=True)
        log.info(f"Working directory for task is {directory}")

        # Prepare FERRE.
        args = prepare_ferre(
            directory,
            dict(
                wavelength=wavelength,
                flux=flux,
                sigma=sigma,
                header_path=self.header_path,
                names=names,
                initial_parameters=initial_parameters,
                frozen_parameters=self.frozen_parameters,  
                interpolation_order=self.interpolation_order, 
                input_weights_path=self.input_weights_path,
                input_lsf_shape_path=self.input_lsf_shape_path,
                lsf_shape_flag=self.lsf_shape_flag,
                error_algorithm_flag=self.error_algorithm_flag,
                wavelength_interpolation_flag=self.wavelength_interpolation_flag,
                optimization_algorithm_flag=self.optimization_algorithm_flag,
                continuum_flag=self.continuum_flag,
                continuum_order=self.continuum_order,
                continuum_segment=self.continuum_segment,
                continuum_reject=self.continuum_reject,
                continuum_observations_flag=self.continuum_observations_flag,
                full_covariance=self.full_covariance,
                pca_project=self.pca_project,
                pca_chi=self.pca_chi,
                n_threads=self.n_threads,
                f_access=self.f_access,
                f_format=self.f_format,
                ferre_kwargs=self.ferre_kwargs
            )
        )

        # Execute, either by slurm or whatever.
        log.debug(f"FERRE ready to roll in {directory}")
        assert self.slurm_kwargs
        self.execute_by_slurm(
            context,
            bash_command="/uufs/chpc.utah.edu/common/home/sdss09/software/apogee/Linux/apogee/trunk/bin/ferre.x",
            directory=directory,
        )
        # Unbelievably, FERRE sends a '1' exit code every time it is executed. Even if it succeeds.
        # TODO: Ask Carlos or Jon to remove this insanity.
        
        # Parse outputs.
        # TODO: clean up this function
        param, param_err, output_meta = parse_ferre_outputs(directory, self.header_path, *args)

        results = group_results_by_instance(param, param_err, output_meta, spectrum_meta, Ns)

        for instance, (result, data) in zip(instances, results):
            if result is None: continue

            create_task_output(
                instance,
                astradb.Ferre,
                **result
            )

            log.debug(f"{instance}")
            log.debug(f"{result}")
            log.debug(f"{data}")

            # TODO: Write a data model product for this intermediate output!
            output_path = utils.output_data_product_path(instance.pk)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as fp:
                pickle.dump((result, data), fp)
            
            log.info(f"Wrote outputs of task instance {instance} to {output_path}")

        # Always return the primary keys that were worked on!
        return self.pks


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
    
    results = []
    common_results = dict(frozen_parameters=output_meta["frozen_parameters"])    
    parameter_names = tuple(map(utils.sanitise_parameter_name, output_meta["parameter_names"]))

    log.debug(f"Ns: {Ns}")

    si = 0
    for i, N in enumerate(Ns):
        if N == 0:
            results.append((None, None))
            continue
        
        sliced = slice(si, si + N)

        result = dict(
            snr=spectrum_meta[i]["snr"],
            log_snr_sq=output_meta["log_snr_sq"][sliced],
            log_chisq_fit=output_meta["log_chisq_fit"][sliced],
            bitmask_flag=output_meta["bitmask_flag"][sliced]
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

        si += N
        
    return results