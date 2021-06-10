
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
        if not self.analyse_individual_visits:
            # Since ApStar files contain the combined spectrum and all individual visits, we are going
            # to supply a data_slice parameter to only return the first spectrum.            
            kwds.update(data_slice=(slice(0, 1), slice(None)))
        
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
            return (task.output() for task in self.get_batch_tasks())
        
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
    analyse_individual_visits = astra.BoolParameter(default=False)





class InitialEstimateOfStellarParametersGivenApStarFileBase(FerreMixin):
    
    grid_header_list_path = astra.Parameter(
        config_path=dict(section="FERRE", name="grid_header_list_path")
    )

    def requires(self):
        """ 
        The requirements of this task are initial estimates from running FERRE
        in potentially many grids per source.
        """

        try:
            return self._requirements

        except AttributeError:
            self._requirements = []
            total = self.get_batch_size()
            if total > 1:
                with tqdm(desc="Dispatching", total=total) as pbar:
                    for iteration, source_kwds, kwds in self.dispatcher():
                        pbar.update(iteration - pbar.n)
                        self._requirements.append(self.clone(self.ferre_task_factory, **kwds))
            else:
                for iteration, source_kwds, kwds in self.dispatcher():
                    self._requirements.append(self.clone(self.ferre_task_factory, **kwds))
                
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
            return (task.output() for task in self.get_batch_tasks())
        
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
            return (task.output() for task in self.get_batch_tasks())
        
        # TODO: To be defined by SDSS5/SDSS4 mixin
        new_path = AstraSource(self).path.replace("/AstraSource", "/ApStar")
        return LocalTarget(new_path)
        


class EstimateStellarParametersGivenApStarFileBase(FerreMixin):

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
                # We want to analyse all spctra at this point.
                analyse_individual_visits=True,
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
            return (task.output() for task in self.get_batch_tasks())
        
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
    
    def requires(self):
        """
        This task requires a median-filtered ApStar file, and the previously 
        determined stellar parameters.
        """
        return {
            "observation": self.clone(self.observation_task_factory),
            "stellar_parameters": self.clone(self.stellar_parameters_task_factory)
        }


    def run(self):

        analyse_individual_visits = False

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

            sanitised_parameter_names = list(map(utils.sanitise_parameter_names, header[0]["LABEL"]))    

            parameter_search_indices_one_indexed, ferre_kwds = get_abundance_keywords(self.element)

            # Since these are dynamic dependencies, we cannot build up the dependency graph at this time.
            # So we need to batch together our own tasks, and to only execute tasks that are incomplete.
            kwds = dict(
                ferre_kwds=ferre_kwds,
                grid_header_path=grid_header_path,
                # TODO: Need to put all the speclib contents in a nicer way together.
                input_weights_path=f"/uufs/chpc.utah.edu/common/home/u6020307/astra-component-data/FERRE/masks/{self.element}.mask",
                analyse_individual_visits=analyse_individual_visits,
                # Don't write source output files for chemical abundances.
                write_source_output=False,
            )

            # Set parameters.
            for index, parameter in enumerate(sanitised_parameter_names, start=1):
                # If you set analyse_individual_visits to False then you will need to take the zero-th index below.
                initial_parameter = getattr(output, parameter)
                if not analyse_individual_visits:
                    initial_parameter = initial_parameter[0]

                kwds[f"initial_{parameter}"] = initial_parameter
                kwds[f"frozen_{parameter}"] = index not in parameter_search_indices_one_indexed
            
            executable_tasks.append(task.clone(task.ferre_task_factory, **kwds))

        # Since we are running things in a batch, it is possible that Star A and Star C can be run together in FERRE.
        # So we batch together what we can, and skip tasks that are already complete, since dynamically running tasks
        # at runtime using "yield <tasks>" does not check whether things are complete o not.
        execute_tasks = batch_tasks_together(executable_tasks, skip_complete=True)
        outputs = yield execute_tasks

        # Copy outputs from the executable tasks.
        for task, execute_task in zip(self.get_batch_tasks(), executable_tasks):
            for key, target in execute_task.output().items():
                task.output()[key].copy_from(target)

        return None


    def output(self):
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())
        return dict(database=DatabaseTarget(astradb.Ferre, self))
        

        

class CheckRequirementsForChemicalAbundancesGivenApStarFileBase(FerreMixin):

    priority = 1

    def requires(self):
        return {
            "observation": self.clone(self.observation_task_factory),
            "stellar_parameters": self.clone(self.stellar_parameters_task_factory)
        }

    def run(self):
        with self.output().open("w") as fp:
            fp.write("")
        
    def output(self):
        return MockTarget(self.task_id)




class EstimateChemicalAbundancesGivenApStarFileBase(FerreMixin):

    elements = astra.ListParameter(default=[
        "Al", "Ca", "Ce", "C", "CN", "Co", "Cr", "Cu", "Mg", "Mn", "Na", 
        "Nd", "Ni", "N",  "O", "P",  "Rb", "Si", "S",  "Ti", "V",  "Yb"
    ])
    
    def requires(self):
        return dict([
            (element, self.clone(self.chemical_abundance_task_factory, element=element)) for element in self.elements
        ])
    
    def run(self):
            
        # Create an ASPCAP table for 'final' results.
        with self.output().open("w") as fp:
            fp.write("")        

        print("done")


    def output(self):
        return MockTarget(self.task_id)
        


'''
class EstimateChemicalAbundancesGivenApStarFileBase(FerreMixin):


    max_asynchronous_slurm_jobs = astra.IntParameter(default=10, significant=False)
    elements = astra.ListParameter(default=[
        "Al", "Ca", "Ce", "C", "CN", "Co", "Cr", "Cu", "Mg", "Mn", "Na", 
        "Nd", "Ni", "N",  "O", "P",  "Rb", "Si", "S",  "Ti", "V",  "Yb"
    ])
    
    def requires(self):
        raise RuntimeError("defined by the sub-class")
    
    
    def run(self):

        headers = {}
        execute_tasks = []

        total = self.requires().get_batch_size() * len(self.elements)
        with tqdm(desc="Batching abundance tasks", total=total) as pbar:

            for task in self.requires().get_batch_tasks():

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

                sanitised_parameter_names = list(map(utils.sanitise_parameter_names, header[0]["LABEL"]))    

                for element in self.elements:
                    parameter_search_indices_one_indexed, ferre_kwds = get_abundance_keywords(element)

                    # Since these are dynamic dependencies, we cannot build up the dependency graph at this time.
                    # So we need to batch together our own tasks, and to only execute tasks that are incomplete.
                    kwds = dict(
                        ferre_kwds=ferre_kwds,
                        grid_header_path=grid_header_path,
                        # TODO: Need to put all the speclib contents in a nicer way together.
                        input_weights_path=f"/uufs/chpc.utah.edu/common/home/u6020307/astra-component-data/FERRE/masks/{element}.mask",
                        # If you set analyse_individual_visits to False then you will need to change the initial parameters below.
                        analyse_individual_visits=True,
                        # Don't write source output files for chemical abundances.
                        write_source_output=False,
                    )

                    # Set parameters.
                    for index, parameter in enumerate(sanitised_parameter_names, start=1):
                        # If you set analyse_individual_visits to False then you will need to take the zero-th index below.
                        kwds[f"initial_{parameter}"] = getattr(output, parameter)
                        kwds[f"frozen_{parameter}"] = index not in parameter_search_indices_one_indexed

                    execute_tasks.append(task.clone(task.ferre_task_factory, **kwds))
                    
                    pbar.update(1)
        
        # Either submit the batched jobs to Slurm on asynchronous threads, or execute in series.
        self.submit_jobs([task.param_kwargs for task in batch_tasks_together(execute_tasks)])

        # Write tasks outputs.
        E = len(self.elements)
        for i, task in enumerate(self.get_batch_tasks()):
            for j, element in enumerate(self.elements):
                output = execute_tasks[i * E + j].output()["database"]
                task.output()[element].copy_from(output)
        return None


    def output(self):
        if self.is_batch_mode:
            return (task.output() for task in self.get_batch_tasks())

        return { 
            element: DatabaseTarget(
                astradb.Ferre, 
                self.clone(self.chemical_abundance_task_factory, element=element)
            ) \
            for element in self.elements
        }

'''


'''
def output_abundances(self):
    """ A convenience function to return the output abundance given the FERRE result. """
    if self.is_batch_mode:
        return [task.output_abundances() for task in self.get_batch_tasks()]

    abundances = {}
    for element in self.elements:
        output = self.output()[element].resolve()
        aux_task = output.task

        label_names = tuple(set(aux_task.initial_parameters).difference(aux_task.frozen_parameters))
        sanitised_label_names = list(map(utils.sanitise_parameter_names, label_names))

        result = output.read(as_dict=True)

        abundances[element] = {
            "value": np.array([result[ln] for ln in sanitised_label_names]),
            "uncertainty": np.array([result[f"u_{ln}"] for ln in sanitised_label_names]),
            "label_names": label_names
        }

    return abundances
'''



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
                


def doppler_estimate_in_bounds_factory(release, public, mirror, grid_header_list_path=None):

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



def get_abundance_keywords(element):
    """
    Return a dictionary of task parameters given a chemical element. These are adopted from DR16.

    :param element:
        The chemical element to measure.
    """

    # These can be inferred from running the following command on the SAS:
    # cd /uufs/chpc.utah.edu/common/home/sdss50/dr16/apogee/spectro/aspcap/r12/l33/apo25m/cal_all_apo25m007/ferre
    # egrep 'INDV|TIE|FILTERFILE' */input.nml

    #kwds = dict(input_weights_path=f"{element}.mask")

    indv = {
        "Al": (6, ),
        "Ca": (4, ),
        "Ce": (6, ),
        "CI": (2, ),
        "C": (2, ),
        "CN": (2, 3),
        "Co": (6, ),
        "Cr": (6, ),
        "Cu": (6, ),
        "Fe": (6, ),
        "Ge": (6, ),
        "K": (6, ),
        "Mg": (4, ),
        "Mn": (6, ),
        "Na": (6, ),
        "Nd": (6, ),
        "Ni": (6, ),
        "N": (3, ),
        "O": (4, ),
        "P": (6, ),
        "Rb": (6, ),
        "Si": (4, ),
        "S": (4, ),
        "TiII": (4, ),
        "Ti": (4, ),
        "V": (6, ),
        "Yb": (6, ),
    }

    ntie = {
        "Al": 3,
        "Ca": 0,
        "Ce": 3,
        "CI": 0,
        "C": 0,
        "CN": 0,
        "Co": 3,
        "Cr": 3,
        "Cu": 3,
        "Fe": 3,
        "Ge": 3,
        "K": 3,
        "Mg": 0,
        "Mn": 3,
        "Na": 3,
        "Nd": 3,
        "Ni": 3,
        "N": 0,
        "O": 0,
        "P": 3,
        "Rb": 3,
        "Si": 0,
        "S": 0,
        "TiII": 0,
        "Ti": 0,
        "V": 3,
        "Yb": 3,
    }

    ferre_kwds = {
        #"INDV": indv[element],
        "NTIE": ntie[element],
        "TYPETIE": 1
    }
    nties = range(1, 1 + ferre_kwds["NTIE"])
    indices = (2, 3, 4)
    for i, j in zip(nties, indices):
        ferre_kwds.update({
            f"INDTIE({i})": j,
            f"TTIE0({i})": 0,
            f"TTIE({i},6)": -1
        })

    return (indv[element], ferre_kwds)
    
