
import numpy as np
import os
import json
import pickle
import shutil
import json
import hashlib
import multiprocessing as mp

from astropy.io.fits import getheader
from astropy.io import fits
from luigi import WrapperTask
from luigi.mock import MockTarget
from luigi.task import flatten
from sdss_access import SDSSPath
from shutil import copyfile
from sqlalchemy import inspect
from tqdm import tqdm

import astra
from astra.database import astradb
from astra.tasks import hashify
from astra.tasks.utils import batch_tasks_together
from astra.tasks.targets import (AstraSource, DatabaseTarget, LocalTarget)
from astra.tools.spectrum import Spectrum1D
from astra.utils import log, symlink_force, get_default

from astra.contrib.ferre_new.continuum import median_filtered_correction
from astra.contrib.ferre_new.tasks.mixin import (FerreMixin, SourceMixin)
from astra.contrib.ferre_new.tasks.ferre import FerreBase
from astra.contrib.ferre_new import utils



class ApStarMixinBase(object):

    """ A base mix-in class for SDSS-IV or SDSS-V ApStarFile objects. """
    
    def requires(self):
        """ Requirements for this task. """
        # If we require this check in batch mode then it means it will check all ApStar files multiple times.
        # This is bad practice, but we are going to do it anyways.
        if self.is_batch_mode:
            return []
        return dict(observation=self.clone(self.observation_task_factory))


    def read_input_observations(self, **kwargs):
        """ Read the input observations. """
        
        kwds = kwargs.copy()

        if self.spectrum_data_slice_args is not None:
            # Since ApStar files contain the combined spectrum and all individual visits, we are going
            # to supply a data_slice parameter to only return the first spectrum.
            #kwds.update(data_slice=(slice(0, 1), slice(None)))
            kwds.update(
                data_slice=slice(*self.spectrum_data_slice_args)
            )
        
        spectra = []
        for task in self.get_batch_tasks():
            spectra.append(Spectrum1D.read(task.input()["observation"].path, **kwds))
        return spectra
    
    
    def get_source_names(self, spectra):
        """ Return a list of source names for convenience in FERRE. """
        
        if self.is_batch_mode:
            args = (self.telescope, self.obj, spectra)
        else:
            args = ([self.telescope], [self.obj], spectra)

        names = []
        for i, (telescope, obj, spectrum) in enumerate(zip(*args)):
            for j in range(spectrum.flux.shape[0]):
                names.append(f"{i:.0f}_{j:.0f}_{telescope}_{obj}")
        return names


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        requirements = {
            "database": DatabaseTarget(astradb.Ferre, self),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements


class FerreGivenApStarFileBase(FerreBase):

    """ Execute FERRE given an ApStar file. """

    grid_header_path = astra.Parameter()
    spectrum_data_slice_args = astra.ListParameter(default=None)




class InitialEstimateOfStellarParametersGivenApStarFileBase(FerreMixin):
    
    grid_header_list_path = astra.Parameter(
        config_path=dict(section="FERRE", name="grid_header_list_path")
    )
    analyze_individual_visits = astra.BoolParameter(default=False)

    def requires(self):
        """ 
        The requirements of this task are initial estimates from running FERRE
        in potentially many grids per source.
        """

        try:
            return self._requirements

        except AttributeError:
            common_kwds = {
                "spectrum_data_slice_args": None if self.analyze_individual_visits else [0, 1] 
            }
            self._requirements = []
            total = self.get_batch_size()
            if total > 1:
                with tqdm(desc="Dispatching", total=total) as pbar:
                    for iteration, source_kwds, kwds in self.dispatcher():
                        pbar.update(iteration - pbar.n)
                        self._requirements.append(self.clone(
                            self.ferre_task_factory, 
                            **{**common_kwds, **kwds}
                        ))
            else:
                for iteration, source_kwds, kwds in self.dispatcher():
                    self._requirements.append(self.clone(
                        self.ferre_task_factory,
                        **{**common_kwds, **kwds}    
                    ))
                
            return self._requirements
            

    def dispatcher(self):
        """
        A generator that yields sources and FERRE grids that should be used for initial
        estimates of stellar parameters.
        """
        sources = self.get_batch_task_kwds(include_non_batch_keywords=False)
        yield from dispatch_apstars_for_analysis(
            sources,
            self.grid_header_list_path,
            release=self.release,
            public=self.public,
            mirror=self.mirror,
        )


    def run(self):
        """ Execute the task. """
        uid = lambda task: "_".join([f"{getattr(task, pn)}" for pn in self.batch_param_names()])

        best_tasks = {}
        for task, output in zip(self.requires(), self.input()):
            
            key = uid(task)
            best_tasks.setdefault(key, (np.inf, None))

            result = output["database"].read()
            
            log_chisq_fit, *_ = result.log_chisq_fit
            previous_teff, *_ = result.teff

            parsed_header = utils.parse_header_path(task.grid_header_path)

            # Penalise chi-sq in the same way they did for DR16.
            # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
            if parsed_header["spectral_type"] == "GK" and previous_teff < 3985:
                # \chi^2 *= 10
                log_chisq_fit += np.log(10)

            if log_chisq_fit < best_tasks[key][0]:
                best_tasks[key] = (log_chisq_fit, result)

        # We don't actually want to create a new database row here.
        # Instead we want to update the task state to point to this existing result.
        for task in self.get_batch_tasks():
            try:
                log_chisq_fit, result = best_tasks[uid(task)]
            except KeyError:
                log.exception(f"No FERRE runs found for {task}. Are the initial parameters within any grid?")
                raise
            
            else:
                # This will have the same effect as if we had written a new database row.
                task.query_state().update(dict(output_pk=result.output_pk))

        return None


    def output(self):
        """ Outputs of this task. """

        # I think the above hack has been fixed by just never sending these kinds of stars to the pipeline.
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        return dict(database=DatabaseTarget(astradb.Ferre, self))
        



class CreateMedianFilteredApStarFileBase(FerreMixin):

    median_filter_width = astra.IntParameter(default=151)
    bad_minimum_flux = astra.FloatParameter(default=0.01)
    non_finite_err_value = astra.FloatParameter(default=1e10)


    def requires(self):
        return {
            "observation": self.clone(self.observation_task_factory),
            "initial_estimate": self.clone(self.initial_estimate_task_factory)
        }


    def run(self):
        
        for task in self.get_batch_tasks():

            initial_estimate_result = task.input()["initial_estimate"]["database"].read()

            # Get the FERRE task so we can extract the grid header path.
            for task_state in initial_estimate_result.get_tasks():
                if task_state.task_id != task.requires()["initial_estimate"].task_id:
                    break
            initial_estimate = task_state.load_task()   
            
            # Re-normalize the spectrum using the previous estimate.
            image = fits.open(initial_estimate.output()["AstraSource"].path)

            # Get segments for each chip based on the model.
            n_pixels = [header["NPIX"] for header in utils.read_ferre_headers(initial_estimate.grid_header_path)][1:]

            with open(initial_estimate.input_wavelength_mask_path, "rb") as fp:
                mask = pickle.load(fp)

            indices = 1 + np.cumsum(mask).searchsorted(np.cumsum(n_pixels))
            # These indices will be for each chip, but will need to be left-trimmed.
            segment_indices = np.sort(np.hstack([
                0,
                np.repeat(indices[:-1], 2),
                mask.size
            ])).reshape((-1, 2))
            
            # Left-trim the indices.
            for i, (start, end) in enumerate(segment_indices):
                segment_indices[i, 0] += mask[start:].searchsorted(True)
            
            continuum = median_filtered_correction(
                wavelength=np.arange(image[1].data[0].size),
                normalised_observed_flux=image[1].data[0],
                normalised_observed_flux_err=image[2].data[0]**-0.5,
                normalised_model_flux=image[5].data[0],
                segment_indices=segment_indices,
                width=self.median_filter_width,
                bad_minimum_flux=self.bad_minimum_flux,
                non_finite_err_value=self.non_finite_err_value                
            )

            # Copy the original file to the output file, then change the flux.
            new_image = fits.open(task.input()["observation"].path)
            new_image[1].data /= continuum
            new_image[2].data /= continuum
            new_image.writeto(task.output().path, overwrite=True)
            
        return None


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        # TODO: To be defined by SDSS5/SDSS4 mixin
        new_path = AstraSource(self).path.replace("/AstraSource", "/ApStar")
        return LocalTarget(new_path)
        


class EstimateStellarParametersGivenApStarFileBase(FerreMixin):

    analyze_individual_visits = astra.BoolParameter(default=False)

    def requires(self):
        """
        The requirements of this task include a median-filtered ApStar file, and an initial
        estimate of the stellar parameters (based on a series of previous FERRE executions).
        """
        return {
            "observation": self.clone(self.observation_task_factory),
            "initial_estimate": self.clone(self.initial_estimate_task_factory)
        }

    
    def run(self):
        """ Execute this task. """

        execute_tasks = []
        
        for task in self.get_batch_tasks():
            # From the initial estimate we need the grid_header_path, and the previous stellar parameters
            # (which we will use for the initial guess here.)
            initial_estimate_result = task.input()["initial_estimate"]["database"].read()
            for task_state in initial_estimate_result.get_tasks():
                if task_state.task_id != task.requires()["initial_estimate"].task_id:
                    break
                
            grid_header_path = task_state.parameters["grid_header_path"]
            headers = utils.read_ferre_headers(grid_header_path)
            parameter_names = list(map(utils.sanitise_parameter_names, headers[0]["LABEL"]))
            
            kwds = dict(
                grid_header_path=grid_header_path,
                spectrum_data_slice_args=None if task.analyze_individual_visits else [0, 1] 
            )

            for parameter_name in parameter_names:
                # Take the first result (of perhaps many spectra; i.e. the stacked one) from initial estimate.
                kwds[f"initial_{parameter_name}"] = getattr(initial_estimate_result, parameter_name)[0]

            execute_tasks.append(task.clone(self.ferre_task_factory, **kwds))
            
        outputs = yield execute_tasks
        
        # Copy outputs from the executed tasks.
        for task, output in zip(self.get_batch_tasks(), outputs):
            for key, target in output.items():
                task.output()[key].copy_from(target)
        
        return None


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        requirements = {
            "database": DatabaseTarget(astradb.Ferre, self),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements



class EstimateChemicalAbundanceGivenApStarFileBase(FerreMixin):

    # Element is not a batch parameter: we run one element for many stars.
    element = astra.Parameter()
    analyze_individual_visits = astra.BoolParameter(default=False)

    def requires(self):
        """
        This task requires a median-filtered ApStar file, and the previously 
        determined stellar parameters.
        """
        return {
            "observation": self.clone(self.observation_task_factory),
            "stellar_parameters": self.clone(self.stellar_parameters_task_factory)
        }


    def get_executable_tasks(self):
        """ Return a list of FERRE tasks that need to be completed. """
        
        headers = {}
        executable_tasks = []
        
        for task in self.get_batch_tasks():

            # Get the previous stellar parameters, and the grid header path used.
            output = task.input()["stellar_parameters"]["database"].read()

            for task_state in output.get_tasks():
                if task_state.task_id != task.requires()["stellar_parameters"].task_id:
                    break
            
            grid_header_path = task_state.parameters["grid_header_path"]

            try:
                header = headers[grid_header_path]
            except KeyError:
                header = headers[grid_header_path] = utils.read_ferre_headers(grid_header_path)

            grid_label_names = header[0]["LABEL"]

            indv_labels, ferre_kwds = get_abundance_keywords(self.element, grid_label_names)

            # Since these are dynamic dependencies, we cannot build up the dependency graph at this time.
            # So we need to batch together our own tasks, and to only execute tasks that are incomplete.
            spectrum_data_slice_args = None if task.analyze_individual_visits else [0, 1]

            kwds = dict(
                ferre_kwds=ferre_kwds,
                grid_header_path=grid_header_path,
                # TODO: Need to put all the speclib contents in a nicer way together.
                input_weights_path=f"/uufs/chpc.utah.edu/common/home/u6020307/astra-component-data/FERRE/masks/{self.element}.mask",
                # Do we want to analyze individual visits or just the stacked spectrum?
                spectrum_data_slice_args=spectrum_data_slice_args,
                # Don't write source output files for chemical abundances.
                write_source_output=False,
            )

            # Set parameters.
            for grid_label_name in grid_label_names:
                parameter_name = utils.sanitise_parameter_names(grid_label_name)
                initial_parameter = getattr(output, parameter_name)
                # If we are doing chemical abundances only on the stacked spectrum, but we determined
                # stellar parameters for the individual visits, then we will just need to take the
                # first entry from the parameter array.
                if spectrum_data_slice_args is not None:
                    initial_parameter = initial_parameter[slice(*spectrum_data_slice_args)]

                kwds[f"initial_{parameter_name}"] = initial_parameter
                kwds[f"frozen_{parameter_name}"] = grid_label_name not in indv_labels
            
            executable_tasks.append(task.clone(task.ferre_task_factory, **kwds))

        return executable_tasks


    def run(self):
  
        # Since we are running things in a batch, it is possible that Star A and Star C can be run together in FERRE.
        # So we batch together what we can, and skip tasks that are already complete, since dynamically running tasks
        # at runtime using "yield <tasks>" does not check whether things are complete o not.
        executable_tasks = self.get_executable_tasks()
        execute_tasks = batch_tasks_together(executable_tasks, skip_complete=True)
        outputs = yield execute_tasks

        # Copy outputs from the executable tasks.
        for task, execute_task in zip(self.get_batch_tasks(), executable_tasks):
            for key, target in execute_task.output().items():
                task.output()[key].copy_from(target)

        return None


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return dict(database=DatabaseTarget(astradb.Ferre, self))
        

        


class EstimateChemicalAbundancesGivenApStarFileBase(FerreMixin):

    elements = astra.ListParameter(default=[
        "Al", "Ca", "Ce", "C", "CN", "Co", "Cr", "Cu", "Mg", "Mn", "Na", 
        "Nd", "Ni", "N",  "O", "P",  "Rb", "Si", "S",  "Ti", "V",  "Yb"
    ])
    analyze_individual_visits = astra.BoolParameter(default=False)


    def requires(self):
        """ This task requires the abundances to be measured of individual elements. """
        return dict([
            (element, self.clone(self.chemical_abundance_task_factory, element=element)) \
            for element in self.elements
        ])
    
    def run(self):
        
        ignore_names = ("pk", "output_pk")

        for task in self.get_batch_tasks():

            # Get the stellar parameters from the parent requirements.
            element, *_ = self.elements

            stellar_parameters_task = task.requires()[element].requires()["stellar_parameters"]
            stellar_parameters = stellar_parameters_task.output()["database"].read()

            sp_col_names = [col.name for col in stellar_parameters.__table__.columns]
            output_col_names = [col.name for col in task.output()["database"].model.__table__.columns]
            col_names = [col for col in sp_col_names if col in output_col_names and col not in ignore_names]

            # Create the result dictionary to write to database.
            result = { col_name: getattr(stellar_parameters, col_name) for col_name in col_names }
            
            for element, requirement in task.requires().items():

                output = requirement.output()["database"].read()

                # In most cases the chemical abundance is measured from the metals dimension, but
                # not always. Let's find the free dimension.
                # TODO: This could be an issue with CN. Maybe we should be more careful about this.
                thawed_key, *_ = [
                    col_name for col_name in sp_col_names \
                    if col_name.startswith("frozen_") and not getattr(output, col_name)
                ]
                thawed_key = thawed_key[len("frozen_"):]
                
                result.update({
                    f"{element.lower()}_h": getattr(output, thawed_key),
                    f"u_{element.lower()}_h": getattr(output, f"u_{thawed_key}")
                })
                
            task.output()["database"].write(result)
        

    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return dict(database=DatabaseTarget(astradb.Aspcap, self))
        



def dispatch_apstars_for_analysis(sources, grid_header_list_path, release=None, public=True, mirror=False):

    with open(grid_header_list_path, "r") as fp:
        grid_header_paths = list(map(str.strip, fp.readlines()))
    
    grid_info = utils.parse_grid_information(grid_header_paths)

    sdss_paths = {}
    for i, source in enumerate(sources):
        try:
            sdss_path = sdss_paths[release]

        except KeyError:
            sdss_paths[release] = sdss_path = SDSSPath(
                release=release,
                public=public,
                mirror=mirror
            )

        path = sdss_path.full("apStar", **source)

        try:
            header = getheader(path)
            teff = utils.safe_read_header(header, ("RV_TEFF", "RVTEFF"))
            logg = utils.safe_read_header(header, ("RV_LOGG", "RVLOGG"))
            fe_h = utils.safe_read_header(header, ("RV_FEH", "RVFEH"))

            # In order to match sources to suitable grids we need the initial parameters,
            # the fiber information, and the telescope used for observation.
            kwds = {
                "mean_fiber": header["MEANFIB"],
                "telescope": source["telescope"],
                "teff": float(teff),
                "logg": float(logg),
                "fe_h": float(fe_h)
            }

        except Exception as exception:
            log.exception(f"Exception: {exception}")
            continue

        else:
            source_kwds = source.copy()
            source_kwds.update(
                # Make sure that *all* of these inputs are given as floats! Otherwise if the
                # task is created with an integer then it is seralised like that, and it
                # creates a different hash than what it should be. Then when it is loaded
                # to/from database (or scheduler) it gets parsed as it should be (a float)
                # and you have Unfilled Dependency errors forever! In reality we should 
                # be more strict about parameter types upon initialisation of a task.
                # TODO: We may just want to make sure things are parsed correctly when we 
                #       generate the hash.
                initial_teff=teff,
                initial_logg=logg,
                initial_metals=fe_h,
                initial_log10vdop=utils.approximate_log10_microturbulence(logg),
                initial_o_mg_si_s_ca_ti=0.0,
                initial_lgvsini=0.0,
                initial_c=0.0,
                initial_n=0.0
            )

            any_suitable_grids = False
            for grid_header_path, parsed_header_path in utils.yield_suitable_grids(grid_info, **kwds):
                any_suitable_grids = True
                
                # In the initial FERRE run we freeze LOG10VDOP.
                all_kwds = source_kwds.copy()
                all_kwds.update(
                    grid_header_path=grid_header_path,
                    frozen_log10vdop=True
                )
                # Freeze C and N to zero if this is a dwarf grid.
                if parsed_header_path["gd"] == "d":
                    all_kwds.update(
                        frozen_c=True,
                        frozen_n=True
                    )

                # We yield an integer so we can see progress of unique objects.
                yield (i, source, all_kwds)
                


def doppler_estimate_in_bounds_factory(
        release, 
        public, 
        mirror, 
        grid_header_list_path=None
    ):
    """
    Returns a function that will take in source keywords for an ApStar file
    and return only sources where the radial velocity estimate from the 'Doppler'
    code is within the boundaries of available FERRE grids.

    :param release:
        The SDSS release.

    :param public:
        Whether the data to be accessed is publicly available or not.
    
    :param mirror:
        Use a SDSS mirror or not.
    
    :param grid_header_list_path: [optional]
        An optional list of paths of FERRE grid files. If None is given then this will
        default to the option for the `InitialEstimateOfStellarParametersGivenApStarFileBase`
        task.
    """

    if grid_header_list_path is None:
        grid_header_list_path = get_default(
            InitialEstimateOfStellarParametersGivenApStarFileBase,
            "grid_header_list_path"
        )

    with open(grid_header_list_path, "r") as fp:
        grid_header_paths = list(map(str.strip, fp.readlines()))
    
    grid_info = utils.parse_grid_information(grid_header_paths)

    sdss_path = SDSSPath(
        release=release,
        public=public,
        mirror=mirror
    )

    def wrapper(source):
        
        path = sdss_path.full("apStar", **source)

        try:
            header = getheader(path)
            teff = utils.safe_read_header(header, ("RV_TEFF", "RVTEFF"))
            logg = utils.safe_read_header(header, ("RV_LOGG", "RVLOGG"))
            fe_h = utils.safe_read_header(header, ("RV_FEH", "RVFEH"))

            # In order to match sources to suitable grids we need the initial parameters,
            # the fiber information, and the telescope used for observation.
            kwds = {
                "mean_fiber": header["MEANFIB"],
                "telescope": source["telescope"],
                "teff": teff,
                "logg": logg,
                "fe_h": fe_h
            }

        except Exception as exception:
            log.exception(f"Exception: {exception}")
            return False

        else:
            for match in utils.yield_suitable_grids(grid_info, **kwds):
                return True
        
        return False

    return wrapper



def get_abundance_keywords(element, header_label_names):
    """
    Return a dictionary of task parameters given a chemical element. These are adopted from DR16.

    :param element:
        The chemical element to measure.

    :param header_label_names:
        The list of label names in the FERRE header file.
    """

    # These can be inferred from running the following command on the SAS:
    # cd /uufs/chpc.utah.edu/common/home/sdss50/dr16/apogee/spectro/aspcap/r12/l33/apo25m/cal_all_apo25m007/ferre
    # egrep 'INDV|TIE|FILTERFILE' */input.nml
    
    controls = {
        "Al": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Ca": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "Ce": {
            "INDV_LABEL": ('METALS', ),
        },
        "CI": {
            "INDV_LABEL": ('C', ),
        },
        "C": {
            "INDV_LABEL": ('C', ),
        },
        "CN": {
            "INDV_LABEL": ('C', 'O Mg Si S Ca Ti', ),
        },
        "Co": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]    
        },
        "Cr": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Cu": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Fe": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Ge": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "K": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Mg": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "Mn": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Na": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Nd": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Ni": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "N": {
            "INDV_LABEL": ('N', ),
        },
        "O": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "P": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Rb": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Si": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "S": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "TiII": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "Ti": {
            "INDV_LABEL": ('O Mg Si S Ca Ti', ),
        },
        "V": {
            "INDV_LABEL": ('METALS', ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        },
        "Yb": {
            "INDV_LABEL": ('METALS'        , ),
            "TIES": [
                ('C', 0, -1),
                ('N', 0, -1),
                ('O Mg Si S Ca Ti', 0, -1)
            ]
        }
    }

    def get_header_index(label):
        # FERRE uses 1-indexing and Python uses 0-indexing.
        return 1 + header_label_names.index(label)

    c = controls[element]
    indv = [get_header_index(label) for label in c["INDV_LABEL"]]
    ties = c.get("TIES", [])

    ferre_kwds = {
        # We don't pass INDV here because this will be determined from the
        # 'frozen_<param>' arguments to the FerreGivenSDSSApStarFile tasks
        #"INDV": [get_header_index(label) for label in c["INDV_LABEL"]],
        "NTIE": len(ties),
        "TYPETIE": 1
    }
    for i, (tie_label, ttie0, ttie) in enumerate(ties, start=1):
        ferre_kwds.update({
            f"INDTIE({i:.0f})": get_header_index(tie_label),
            f"TTIE0({i:.0f})": ttie0,
            # TODO: What if we don't want to tie it back to first INDV element?
            f"TTIE({i:.0f},{indv[0]:.0f})": ttie
        })
    
    return (c["INDV_LABEL"], ferre_kwds)

