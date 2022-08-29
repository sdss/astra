from mimetypes import knownfiles
import os
import numpy as np
import subprocess
import sys
import pickle
from tempfile import mkdtemp

from astra import log, __version__
from astra.base import ExecutableTask, Parameter, TupleParameter, DictParameter
from astra.tools.spectrum import Spectrum1D
from astra.contrib.ferre import bitmask, utils
from astra.utils import flatten, executable, expand_path, nested_list
from astra.database.astradb import (
    database,
    DataProduct,
    TaskOutputDataProducts,
    Output,
    TaskOutput,
    FerreOutput,
)
from astra.operators.sdss import get_apvisit_metadata


class Ferre(ExecutableTask):

    header_path = Parameter("header_path", bundled=True)
    initial_parameters = DictParameter("initial_parameters", default=None)
    frozen_parameters = DictParameter("frozen_parameters", default=None, bundled=True)
    interpolation_order = Parameter("interpolation_order", default=3, bundled=True)
    weight_path = Parameter("weight_path", default=None, bundled=True)
    lsf_shape_path = Parameter("lsf_shape_path", default=None, bundled=True)
    lsf_shape_flag = Parameter("lsf_shape_flag", default=0, bundled=True)
    error_algorithm_flag = Parameter("error_algorithm_flag", default=1, bundled=True)
    wavelength_interpolation_flag = Parameter(
        "wavelength_interpolation_flag", default=0, bundled=True
    )
    optimization_algorithm_flag = Parameter(
        "optimization_algorithm_flag", default=3, bundled=True
    )
    continuum_flag = Parameter("continuum_flag", default=1, bundled=True)
    continuum_order = Parameter("continuum_order", default=4, bundled=True)
    continuum_segment = Parameter("continuum_segment", default=None, bundled=True)
    continuum_reject = Parameter("continuum_reject", default=0.3, bundled=True)
    continuum_observations_flag = Parameter(
        "continuum_observations_flag", default=1, bundled=True
    )
    full_covariance = Parameter("full_covariance", default=False, bundled=True)
    pca_project = Parameter("pca_project", default=False, bundled=True)
    pca_chi = Parameter("pca_chi", default=False, bundled=True)
    f_access = Parameter("f_access", default=None, bundled=True)
    f_format = Parameter("f_format", default=1, bundled=True)
    ferre_kwds = DictParameter("ferre_kwds", default=None, bundled=True)
    parent_dir = Parameter("parent_dir", default=None, bundled=True)
    n_threads = Parameter("n_threads", default=1, bundled=True)

    # For normalization to be made before the FERRE run.
    normalization_method = Parameter(
        "normalization_method", default=None, bundled=False
    )
    normalization_kwds = DictParameter(
        "normalization_kwds", default=None, bundled=False
    )

    # For deciding what rows to use from each data product.
    slice_args = TupleParameter("slice_args", default=None, bundled=False)

    # FERRE will sometimes hang forever if there is a spike in the data (e.g., a skyline) that
    # is not represented by the uncertainty array (e.g., it looks 'real').

    # An example of this on Utah is under ~/ferre-death-examples/spike/
    # To self-preserve FERRE, we do a little adjustment to the uncertainty array.
    spike_threshold_to_inflate_uncertainty = Parameter(default=5, bundled=True)

    # Maximum timeout in seconds for FERRE
    timeout = Parameter(default=12 * 60 * 60, bundled=True)

    @classmethod
    def estimate_relative_cost_factors(cls, parameters):
        """
        Return a three-length array containing the relative cost per:
            - task,
            - data product, and
            - size of the data product.

        Here 'relative cost' is relative to other tasks of this type. For example, if one parameter
        makes the cost of this task twice as long per data product, this method will take that
        into account. That makes Slurm scheduling more efficient.
        """
        # The cost scales significantly with the number of dimensions being solved for.
        headers, *segment_headers = utils.read_ferre_headers(
            expand_path(parameters["header_path"])
        )
        D = int(headers["N_OF_DIM"] - len(parameters.get("frozen_parameters", {})))

        # some rough scaling from experiments on 20220412
        scales = {6: 1, 7: 2, 8: 10}
        scale = scales.get(D, 1)

        factor_task, factor_data_product, factor_data_product_size = (0, 0, 0)

        # Now we just need to figure out where this scaling should apply.
        # If we are slicing the data products then the scaling should go as number of data products.

        # If we are not slicing the data products then the scaling should go as the size of the
        # data products
        if parameters.get("slice_args", None) is None:
            factor_data_product_size = scale
        else:
            # Estimate the number slicing each time.
            N = np.ptp(np.array(parameters["slice_args"]).flatten())
            factor_data_product = N * scale

        return np.array([factor_task, factor_data_product, factor_data_product_size])

    @classmethod
    def to_name(cls, i, j, k, data_product, snr, **kwargs):
        obj = data_product.kwargs.get("obj", "NOOBJ")
        return f"{i:.0f}_{j:.0f}_{k:.0f}_{snr:.1f}_{obj}"

    @classmethod
    def from_name(cls, name):
        i, j, k, snr, *obj = name.split("_")
        return dict(i=int(i), j=int(j), k=int(k), snr=float(snr), obj="_".join(obj))

    def pre_execute(self):

        # Check if the pre-execution has already happened somewhere else.
        if "pre_execute" in self.context:
            return None

        # Create a temporary directory.
        if self.parent_dir is not None:
            parent_dir = expand_path(self.parent_dir)
            bundle = self.context.get("bundle", None)
            if bundle is None:
                tasks = self.context.get("tasks", None)
                if tasks is not None:
                    if len(tasks) == 1:
                        descr = f"task_{tasks[0].id}"
                    else:
                        if len(tasks) == 2:
                            first_task, last_task = tasks
                        else:
                            first_task, *_, last_task = tasks
                        descr = f"tasks/task_{first_task.id}_to_{last_task.id}"
                    dir = os.path.join(parent_dir, descr)
                else:
                    dir = mkdtemp(dir=parent_dir)
            else:
                dir = os.path.join(parent_dir, f"bundles/{bundle.id % 100}/{bundle.id}")
        else:
            dir = mkdtemp()

        os.makedirs(dir, exist_ok=True)

        log.info(f"Created directory for FERRE: {dir}")

        # Validate the control file keywords.
        (
            control_kwds,
            headers,
            segment_headers,
            frozen_parameters,
        ) = utils.validate_ferre_control_keywords(
            header_path=self.header_path,
            frozen_parameters=self.frozen_parameters,
            interpolation_order=self.interpolation_order,
            weight_path=self.weight_path,
            lsf_shape_path=self.lsf_shape_path,
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
        )

        # Write the control file.
        with open(os.path.join(dir, "input.nml"), "w") as fp:
            fp.write(utils.format_ferre_control_keywords(control_kwds))

        # Read in the input data products.
        wl, flux, sigma = ([], [], [])
        names, initial_parameters_as_dicts = ([], [])
        indices = []
        spectrum_metas = []
        for i, (task, data_products, parameters) in enumerate(self.iterable()):
            for j, data_product in enumerate(flatten(data_products)):
                spectrum = Spectrum1D.read(data_product.path)

                # Get relevant spectrum metadata
                spectrum_meta = get_apvisit_metadata(data_product)

                # Apply any slicing, if requested.
                if parameters["slice_args"] is not None:
                    # TODO: Refactor this and put somewhere common.
                    slices = tuple([slice(*args) for args in parameters["slice_args"]])
                    spectrum_meta = spectrum_meta[
                        slices[0]
                    ]  # TODO: allow for more than 1 slice?
                    spectrum._data = spectrum._data[slices]
                    spectrum._uncertainty.array = spectrum._uncertainty.array[slices]
                    for key in ("bitmask", "snr"):
                        try:
                            spectrum.meta[key] = np.array(spectrum.meta[key])[slices]
                        except:
                            log.exception(
                                f"Unable to slice '{key}' metadata with {parameters['slice_args']} on {task} {data_product}"
                            )

                # Apply any general normalization method.
                if parameters["normalization_method"] is not None:
                    _class = executable(parameters["normalization_method"])
                    rectifier = _class(
                        spectrum, **(parameters["normalization_kwds"] or dict())
                    )

                    # Normalization methods for FERRE cannot be applied within the log-likelihood
                    # function, because we'd have to have it executed *within* FERRE.
                    if len(rectifier.parameter_names) > 0:
                        raise TypeError(
                            f"Normalization method {parameters['normalization_method']} on {self} cannot be applied within the log-likelihood function for FERRE."
                        )
                    spectrum = rectifier()

                N, P = spectrum.flux.shape
                initial_parameters = parameters["initial_parameters"]
                # Allow initital parameters to be a dict (applied to all spectra) or a list of dicts (one per spectra)
                log.debug(
                    f"There are {N} spectra in {task} {data_product} and initial params is {len(initial_parameters)} long"
                )
                log.debug(f"And {set(map(type, initial_parameters))}")

                if len(initial_parameters) == N and all(
                    isinstance(_, dict) for _ in initial_parameters
                ):
                    log.debug(
                        f"Allowing different initial parameters for each {N} spectra on task {task}"
                    )
                    initial_parameters_as_dicts.extend(initial_parameters)
                else:
                    if N > 1:
                        log.debug(
                            f"Using same initial parameters {initial_parameters} for all {N} spectra on task {task}"
                        )
                    initial_parameters_as_dicts.extend([initial_parameters] * N)

                if N != len(spectrum_meta):
                    log.warning(
                        f"Number of spectra does not match expected from visit metadata: {N} != {len(spectrum_meta)}"
                    )

                for k in range(N):
                    indices.append((i, j, k))
                    names.append(
                        self.to_name(
                            i=i,
                            j=j,
                            k=k,
                            data_product=data_product,
                            snr=spectrum.meta["snr"][k],
                        )
                    )
                    wl.append(spectrum.wavelength)
                    flux.append(spectrum.flux.value[k])
                    sigma.append(spectrum.uncertainty.array[k] ** -0.5)

                spectrum_metas.extend(spectrum_meta)

        indices, wl, flux, sigma = (
            np.array(indices),
            np.array(wl),
            np.array(flux),
            np.array(sigma),
        )

        # Convert list of dicts of initial parameters to array.
        initial_parameters = utils.validate_initial_and_frozen_parameters(
            headers,
            initial_parameters_as_dicts,
            frozen_parameters,
            clip_initial_parameters_to_boundary_edges=True,
            clip_epsilon_percent=1,
        )

        with open(os.path.join(dir, control_kwds["pfile"]), "w") as fp:
            for name, point in zip(names, initial_parameters):
                fp.write(utils.format_ferre_input_parameters(*point, name=name))

        # Construct mask to match FERRE model grid.
        model_wavelengths = tuple(map(utils.wavelength_array, segment_headers))
        mask = np.zeros(wl.shape[1], dtype=bool)
        for model_wavelength in model_wavelengths:
            # TODO: Building wavelength mask off just the first wavelength array.
            #       We are assuming all have the same wavelength array.
            s_index, e_index = wl[0].searchsorted(model_wavelength[[0, -1]])
            mask[s_index : e_index + 1] = True

        # Sometimes FERRE will run forever
        self.spike_threshold_to_inflate_uncertainty = 5
        if self.spike_threshold_to_inflate_uncertainty > 0:
            flux_median = np.median(flux[:, mask], axis=1).reshape((-1, 1))
            flux_stddev = np.std(flux[:, mask], axis=1).reshape((-1, 1))
            sigma_median = np.median(sigma[:, mask], axis=1).reshape((-1, 1))

            delta = (flux - flux_median) / flux_stddev
            is_spike = (delta > self.spike_threshold_to_inflate_uncertainty) * (
                sigma < (self.spike_threshold_to_inflate_uncertainty * sigma_median)
            )

            if np.any(is_spike):
                fraction = np.sum(is_spike[:, mask]) / is_spike[:, mask].size
                log.warning(
                    f"Inflating uncertainties for {np.sum(is_spike)} pixels ({100 * fraction:.2f}%) that were identified as spikes."
                )
                for i in range(is_spike.shape[0]):
                    n = np.sum(is_spike[i, mask])
                    if n > 0:
                        log.debug(f"  {n} pixels on spectrum index {i}")
                sigma[is_spike] = 1e10

        # Write data arrays.
        savetxt_kwds = dict(fmt="%.4e", footer="\n")
        np.savetxt(
            os.path.join(dir, control_kwds["ffile"]), flux[:, mask], **savetxt_kwds
        )
        np.savetxt(
            os.path.join(dir, control_kwds["erfile"]), sigma[:, mask], **savetxt_kwds
        )

        # Write metadata file to pick up later.
        with open(os.path.join(dir, "spectrum_meta.pkl"), "wb") as fp:
            pickle.dump(spectrum_metas, fp)

        context = dict(dir=dir)
        return context

    def execute(self):
        """Execute FERRE"""

        try:
            dir = self.context["pre_execute"]["dir"]
        except:
            raise ValueError(
                f"No directory prepared by pre-execute in self.context['pre_execute']['dir'] "
            )

        stdout_path = os.path.join(dir, "stdout")
        stderr_path = os.path.join(dir, "stderr")

        try:
            with open(stdout_path, "w") as stdout:
                with open(stderr_path, "w") as stderr:
                    process = subprocess.run(
                        ["ferre.x"],
                        cwd=dir,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=24 * 60 * 60,  # self.timeout
                    )
        except:
            log.exception(f"Exception when calling FERRE in {dir}:")
            raise

        else:
            with open(stdout_path, "r") as fp:
                stdout = fp.read()
            with open(stderr_path, "r") as fp:
                stderr = fp.read()

        """
        # Issues with processes hanging forever, which might be related to pipe buffers being full.

        try:
            process = subprocess.Popen(
                ["ferre.x"],
                cwd=dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                encoding="utf-8",
                close_fds="posix" in sys.builtin_module_names
            )
        except subprocess.CalledProcessError:
            log.exception(f"Exception when calling FERRE in {dir}")
            raise
        else:
            try:
                stdout, stderr = process.communicate()
            except subprocess.TimeoutExpired:
                raise
            else:
                log.info(f"FERRE stdout:\n{stdout}")
                log.error(f"FERRE stderr:\n{stderr}")
        """

        n_done, n_error, control_kwds = utils.parse_ferre_output(dir, stdout, stderr)
        log.info(f"FERRE finished with {n_done} successful and {n_error} errors.")

        # Write stdout and stderr
        with open(os.path.join(dir, "stdout"), "w") as fp:
            fp.write(stdout)
        with open(os.path.join(dir, "stderr"), "w") as fp:
            fp.write(stderr)

        # We actually have timings per-spectrum but we aggregate this to per-task.
        # We might want to store the per-data-product and per-spectrum timing elsewhere.
        try:
            # Update internal timings with those from FERRE.
            timings = utils.get_processing_times(stdout)
            """
            names = np.loadtxt(os.path.join(dir, control_kwds["PFILE"]), usecols=0, dtype=str)
            time_execute_per_task = np.zeros(len(self.context["tasks"]))
            for name, t in zip(names, timings["time_per_spectrum"]):
                time_execute_per_task[self.from_name(name)["i"]] += t
            
            # And store the FERRE load time as the bundle overhead.
            # TODO: The only other way around this is to somehow take the number of threads used
            #       into account when timing a task, but that becomes pretty tricky for all tasks.
            self.context["timing"]["time_execute_per_task"] = list(time_execute_per_task)
            self.context["timing"]["time_execute_bundle_overhead"] = timings["time_load"]
            """
        except:
            log.exception(
                f"Exception when trying to update internal task timings from FERRE."
            )
        else:
            log.debug(f"Timing information from FERRE stdout:")
            for key, value in timings.items():
                log.debug(f"\t{key}: {value}")

        # Parse the outputs from the FERRE run.
        path = os.path.join(dir, control_kwds["OPFILE"])
        try:
            names, params, param_errs, meta = utils.read_output_parameter_file(
                path,
                n_dimensions=control_kwds["NDIM"],
                full_covariance=control_kwds["COVPRINT"],
            )
        except:
            log.exception(f"Exception when parsing FERRE output parameter file {path}")
            raise

        # Parse flux outputs.
        try:
            path = os.path.join(dir, control_kwds["FFILE"])
            flux = np.atleast_2d(np.loadtxt(path))
        except:
            log.exception(f"Failed to load input flux from {path}")

        try:
            path = os.path.join(dir, control_kwds["OFFILE"])
            model_flux = np.atleast_2d(np.loadtxt(path))
        except:
            log.exception(f"Failed to load model flux from {path}")
            raise

        try:
            path = os.path.join(dir, control_kwds["ERFILE"])
            flux_sigma = np.atleast_2d(np.loadtxt(path))
        except:
            log.exception(f"Failed to load flux sigma from {path}")
            raise

        if "SFFILE" in control_kwds:
            try:
                path = os.path.join(dir, control_kwds["SFFILE"])
                normalized_flux = np.atleast_2d(np.loadtxt(path))
            except:
                log.exception(f"Failed to load normalized flux from {path}")
                raise
            else:
                continuum = flux / normalized_flux
        else:
            continuum = np.ones_like(flux)
            normalized_flux = flux

        headers, *segment_headers = utils.read_ferre_headers(
            utils.expand_path(self.header_path)
        )
        parameter_names = utils.sanitise(headers["LABEL"])

        wavelength = np.hstack(tuple(map(utils.wavelength_array, segment_headers)))

        # Flag things.
        param_bitmask = bitmask.ParamBitMask()
        param_bitmask_flags = np.zeros(params.shape, dtype=np.int64)

        bad_lower = headers["LLIMITS"] + headers["STEPS"] / 8
        bad_upper = headers["ULIMITS"] - headers["STEPS"] / 8
        param_bitmask_flags[
            (params < bad_lower) | (params > bad_upper)
        ] |= param_bitmask.get_value("GRIDEDGE_BAD")

        warn_lower = headers["LLIMITS"] + headers["STEPS"]
        warn_upper = headers["ULIMITS"] - headers["STEPS"]
        param_bitmask_flags[
            (params < warn_lower) | (params > warn_upper)
        ] |= param_bitmask.get_value("GRIDEDGE_WARN")
        param_bitmask_flags[
            (params == -999) | (param_errs < -0.01)
        ] |= param_bitmask.get_value("FERRE_FAIL")

        # Check for any erroneous outputs
        if np.any(param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")):
            v = param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")
            idx = np.where(
                np.any(
                    param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL"), axis=1
                )
            )
            log.warning(f"FERRE returned all erroneous values for an entry: {idx} {v}")

        with open(os.path.join(dir, "spectrum_meta.pkl"), "rb") as fp:
            spectrum_metas = pickle.load(fp)

        results_dict = {}
        ijks = []
        for z, (name, param, param_err, bitmask_flag, spectrum_meta) in enumerate(
            zip(names, params, param_errs, param_bitmask_flags, spectrum_metas)
        ):
            parsed = self.from_name(name)
            result = dict(
                log_chisq_fit=meta["log_chisq_fit"][z],
                log_snr_sq=meta["log_snr_sq"][z],
                frac_phot_data_points=meta["frac_phot_data_points"][z],
                snr=parsed["snr"],
            )

            result["meta"] = spectrum_meta

            try:
                result.update(
                    ferre_time_elapsed=timings["time_per_spectrum"][z],
                    ferre_time_load=timings["time_load"],
                    ferre_n_obj=timings["n_obj"],
                    ferre_n_threads=timings["n_threads"],
                )
            except:
                log.exception(
                    f"Exception while trying to include FERRE timing information in the database for {self}"
                )

            result.update(dict(zip(parameter_names, param)))
            result.update(dict(zip([f"u_{pn}" for pn in parameter_names], param_err)))
            result.update(
                dict(zip([f"bitmask_{pn}" for pn in parameter_names], bitmask_flag))
            )

            # Add spectra.
            result["data"] = dict(
                wavelength=wavelength,
                flux=flux[z],
                flux_sigma=flux_sigma[z],
                model_flux=model_flux[z],
                continuum=continuum[z],
                normalized_flux=normalized_flux[z],
            )

            i, j, k = (int(parsed[_]) for _ in "ijk")
            ijks.append((i, j, k))
            results_dict.setdefault((i, j), [])
            results_dict[(i, j)].append(result)

        # List-ify.
        results = nested_list(ijks)
        for (i, j), value in results_dict.items():
            for k, result in enumerate(value):
                results[i][j][k] = result
        return results

    def post_execute(self):
        """
        Post-execute hook after FERRE is complete.

        Read in the output files, create rows in the database, and produce output data products.
        """

        results = self.context["execute"]

        # Create outputs in the database.
        with database.atomic() as txn:
            for (task, data_products, _), task_results in zip(self.iterable(), results):
                for (data_product, data_product_results) in zip(
                    flatten(data_products), task_results
                ):

                    for result in data_product_results:
                        output = Output.create()

                        # Create a data product.
                        # TODO: This is a temporary hack until we have a data model in place.
                        path = expand_path(
                            f"$MWM_ASTRA/{__version__}/ferre/tasks/{task.id % 100}/{task.id}/output_{output.id}.pkl"
                        )
                        os.makedirs(os.path.dirname(path), exist_ok=True)

                        with open(path, "wb") as fp:
                            pickle.dump(result, fp)

                        output_data_product = DataProduct.create(
                            release=data_product.release,
                            filetype="full",
                            kwargs=dict(full=path),
                        )
                        TaskOutputDataProducts.create(
                            task=task, data_product=output_data_product
                        )

                        # Spectra don't belong in the database.
                        result.pop("data")

                        TaskOutput.create(task=task, output=output)
                        FerreOutput.create(task=task, output=output, **result)

                        log.info(
                            f"Created output {output} for task {task} and data product {data_product}"
                        )
                        log.info(
                            f"New output data product: {output_data_product} at {path}"
                        )

        return None


"""
def create_task_output(task, model, **kwargs):
    output = Output.create()
    task_output = TaskOutput.create(task=task, output=output)
    result = model.create(
        task=task,
        output=output,
        **kwargs
    )
    return (output, task_output, result)
"""
