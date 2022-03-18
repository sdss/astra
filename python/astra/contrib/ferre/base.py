from mimetypes import knownfiles
import os
import numpy as np
import subprocess
import sys
import pickle
from tempfile import mkdtemp

from astra import log, __version__
from astra.base import ExecutableTask, Parameter
from astra.tools.spectrum import Spectrum1D
from astra.contrib.ferre import (bitmask, utils)
from astra.utils import flatten, executable, expand_path, nested_list
from astra.database.astradb import (database, DataProduct, TaskOutputDataProducts, Output, TaskOutput, FerreOutput)

class Ferre(ExecutableTask):

    header_path = Parameter("header_path", bundled=True)
    initial_parameters = Parameter("initial_parameters", default=None)
    frozen_parameters = Parameter("frozen_parameters", default=None, bundled=True)
    interpolation_order = Parameter("interpolation_order", default=3, bundled=True)
    weight_path = Parameter("weight_path", default=None, bundled=True)
    lsf_shape_path = Parameter("lsf_shape_path", default=None, bundled=True)
    lsf_shape_flag = Parameter("lsf_shape_flag", default=0, bundled=True)
    error_algorithm_flag = Parameter("error_algorithm_flag", default=1, bundled=True)
    wavelength_interpolation_flag = Parameter("wavelength_interpolation_flag", default=0, bundled=True)
    optimization_algorithm_flag = Parameter("optimization_algorithm_flag", default=3, bundled=True)
    continuum_flag = Parameter("continuum_flag", default=1, bundled=True)
    continuum_order = Parameter("continuum_order", default=4, bundled=True)
    continuum_segment = Parameter("continuum_segment", default=None, bundled=True)
    continuum_reject = Parameter("continuum_reject", default=0.3, bundled=True)
    continuum_observations_flag = Parameter("continuum_observations_flag", default=1, bundled=True)
    full_covariance = Parameter("full_covariance", default=False, bundled=True)
    pca_project = Parameter("pca_project", default=False, bundled=True)
    pca_chi = Parameter("pca_chi", default=False, bundled=True)
    f_access = Parameter("f_access", default=None, bundled=True)
    f_format = Parameter("f_format", default=1, bundled=True)
    ferre_kwds = Parameter("ferre_kwds", default=None, bundled=True)
    parent_dir = Parameter("parent_dir", default=None, bundled=True)
    n_threads = Parameter("n_threads", default=1, bundled=True)

    # For normalization to be made before the FERRE run.
    normalization_method = Parameter("normalization_method", default=None, bundled=False)
    normalization_kwds = Parameter("normalization_kwds", default=None, bundled=False)

    # For deciding what rows to use from each data product.
    slice_args = Parameter("slice_args", default=None, bundled=False)


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
        if "dir" in self.context:
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
                        descr = f"task_{first_task.id}_to_{last_task.id}"
                    dir = os.path.join(parent_dir, descr)
                else:
                    dir = mkdtemp(dir=parent_dir)
            else:
                dir = os.path.join(parent_dir, f"bundle_{bundle.id}")
        else:
            dir = mkdtemp()

        os.makedirs(dir, exist_ok=True)

        log.info(f"Created directory for FERRE: {dir}")

        # Validate the control file keywords.
        control_kwds, headers, segment_headers, frozen_parameters = utils.validate_ferre_control_keywords(
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
        for i, (task, data_products, parameters) in enumerate(self.iterable()):
            for j, data_product in enumerate(flatten(data_products)):
                spectrum = Spectrum1D.read(data_product.path)

                # Apply any slicing, if requested.
                if parameters["slice_args"] is not None:
                    # TODO: Refactor this and put somewhere common.                    
                    slices = tuple([slice(*args) for args in parameters["slice_args"]])
                    spectrum._data = spectrum._data[slices]
                    spectrum._uncertainty.array = spectrum._uncertainty.array[slices]
                    for key in ("bitmask", "snr"):
                        try:
                            spectrum.meta[key] = np.array(spectrum.meta[key])[slices]
                        except:
                            log.exception(f"Unable to slice '{key}' metadata with {parameters['slice_args']} on {task} {data_product}")

                # Apply any general normalization method.
                if parameters["normalization_method"] is not None:
                    _class = executable(parameters["normalization_method"])
                    rectifier = _class(spectrum, **(parameters["normalization_kwds"] or dict()))

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
                log.debug(f"There are {N} spectra in {task} {data_product} and initial params is {len(initial_parameters)} long")
                log.debug(f"And {set(map(type, initial_parameters))}")

                if len(initial_parameters) == N and all(isinstance(_, dict) for _ in initial_parameters):
                    log.debug(f"Allowing different initial parameters for each {N} spectra on task {task}")
                    initial_parameters_as_dicts.extend(initial_parameters)
                else:
                    if N > 1:
                        log.debug(f"Using same initial parameters {initial_parameters} for all {N} spectra on task {task}")
                    initial_parameters_as_dicts.extend([initial_parameters] * N)

                for k in range(N):
                    indices.append((i, j, k))
                    names.append(self.to_name(i=i, j=j, k=k, data_product=data_product, snr=spectrum.meta["snr"][k]))
                    wl.append(spectrum.wavelength)
                    flux.append(spectrum.flux.value[k])
                    sigma.append(spectrum.uncertainty.array[k]**-0.5)

        indices, wl, flux, sigma = (np.array(indices), np.array(wl), np.array(flux), np.array(sigma))
        
        # Convert list of dicts of initial parameters to array.
        initial_parameters = utils.validate_initial_and_frozen_parameters(
            headers,
            initial_parameters_as_dicts,
            frozen_parameters,
            clip_initial_parameters_to_boundary_edges=True,
            clip_epsilon_percent=1
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
            mask[s_index:e_index + 1] = True

        # Write data arrays.
        savetxt_kwds = dict(fmt="%.4e", footer="\n")
        np.savetxt(os.path.join(dir, control_kwds["ffile"]), flux[:, mask], **savetxt_kwds)
        np.savetxt(os.path.join(dir, control_kwds["erfile"]), sigma[:, mask], **savetxt_kwds)

        context = dict(dir=dir)
        return context


    def execute(self):
        """ Execute FERRE """
        
        dir = self.context["dir"]

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
            
        n_done, n_error, control_kwds = utils.parse_ferre_output(dir, stdout, stderr)
        log.info(f"FERRE finished with {n_done} successful and {n_error} errors.")

        # We actually have timings per-spectrum but we aggregate this to per-task.
        # We might want to store the per-data-product and per-spectrum timing elsewhere.
        try:
            # Update internal timings with those from FERRE.
            timings = utils.get_processing_times(stdout)

            names = np.loadtxt(os.path.join(dir, control_kwds["PFILE"]), usecols=0, dtype=str)
            time_execute_task = np.zeros(len(self.context["tasks"]))
            for name, t in zip(names, timings["time_per_ordered_spectrum"]):
                time_execute_task[self.from_name(name)["i"]] += t
            self._timing["time_execute_task"] = time_execute_task
        except:
            log.warning(f"Exception when trying to update internal task timings from FERRE.")

        # Parse the outputs from the FERRE run.
        path = os.path.join(dir, control_kwds["OPFILE"])
        try:
            names, params, param_errs, meta = utils.read_output_parameter_file(
                path,
                n_dimensions=control_kwds["NDIM"],
                full_covariance=control_kwds["COVPRINT"]
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
        
        headers, *segment_headers = utils.read_ferre_headers(utils.expand_path(self.header_path))
        parameter_names = utils.sanitise(headers["LABEL"])

        wavelength = np.hstack(tuple(map(utils.wavelength_array, segment_headers)))            

        # Flag things.
        param_bitmask = bitmask.ParamBitMask()
        param_bitmask_flags = np.zeros(params.shape, dtype=np.int64)
        
        bad_lower = (headers["LLIMITS"] + headers["STEPS"]/8)
        bad_upper = (headers["ULIMITS"] - headers["STEPS"]/8)
        param_bitmask_flags[(params < bad_lower) | (params > bad_upper)] |= param_bitmask.get_value("GRIDEDGE_BAD")

        warn_lower = (headers["LLIMITS"] + headers["STEPS"])
        warn_upper = (headers["ULIMITS"] - headers["STEPS"])
        param_bitmask_flags[(params < warn_lower) | (params > warn_upper)] |= param_bitmask.get_value("GRIDEDGE_WARN")
        param_bitmask_flags[(params == -999) | (param_errs < -0.01)] |= param_bitmask.get_value("FERRE_FAIL")

        # Check for any erroneous outputs
        if np.any(param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")):
            v = param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL")
            idx = np.where(np.any(param_bitmask_flags & param_bitmask.get_value("FERRE_FAIL"), axis=1))
            log.warning(f"FERRE returned all erroneous values for an entry: {idx} {v}")

        results_dict = {}
        ijks = []
        for z, (name, param, param_err, bitmask_flag) in enumerate(zip(names, params, param_errs, param_bitmask_flags)):
            parsed = self.from_name(name)
            result = dict(
                log_chisq_fit=meta["log_chisq_fit"][z],
                log_snr_sq=meta["log_snr_sq"][z],
                frac_phot_data_points=meta["frac_phot_data_points"][z],
                snr=parsed["snr"],
            )
            result.update(dict(zip(parameter_names, param)))
            result.update(dict(zip([f"u_{pn}" for pn in parameter_names], param_err)))
            result.update(dict(zip([f"bitmask_{pn}" for pn in parameter_names], bitmask_flag)))
            
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

        # Create outputs in the database.
        with database.atomic() as txn:
            for (task, data_products, _), task_results in zip(self.iterable(), self.result):
                for (data_product, data_product_results) in zip(flatten(data_products), task_results):

                    for result in data_product_results:
                        output = Output.create()

                        # Create a data product.
                        # TODO: This is a temporary hack until we have a data model in place.
                        path = expand_path(f"$MWM_ASTRA/{__version__}/ferre/task_{task.id}/output_{output.id}.pkl")
                        os.makedirs(os.path.dirname(path), exist_ok=True)

                        with open(path, "wb") as fp:
                            pickle.dump(result, fp)

                        output_data_product = DataProduct.create(
                            release=data_product.release,
                            filetype="full",
                            kwargs=dict(full=path)
                        )
                        TaskOutputDataProducts.create(
                            task=task,
                            data_product=output_data_product
                        )

                        # Spectra don't belong in the database.
                        result.pop("data")

                        TaskOutput.create(task=task, output=output)
                        FerreOutput.create(
                            task=task,
                            output=output,
                            **result
                        )

                        log.info(f"Created output {output} for task {task} and data product {data_product}")
                        log.info(f"New output data product: {output_data_product} at {path}")
                    
        return None

