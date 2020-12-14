
import numpy as np
import os
import json
import pickle
import shutil
from astropy.io.fits import getheader
from astropy.io import fits
from sdss_access import SDSSPath
from shutil import copyfile
from tqdm import tqdm

import astra
from astra.contrib.ferre.continuum import median_filtered_correction
from astra.contrib.ferre.tasks.mixin import (FerreMixin, SourceMixin)
from astra.contrib.ferre.tasks.new_ferre import FerreBase
from astra.contrib.ferre.tasks.targets import (FerreResult, FerreResultProxy)
from astra.contrib.ferre import utils
from astra.tasks.targets import AstraSource, LocalTarget
from astra.tasks.io.sdss4 import ApStarFile
from astra.tools.spectrum import Spectrum1D
from astra.utils import log, symlink_force
from luigi import WrapperTask
from luigi.mock import MockTarget



class ApStarMixin(ApStarFile):


    def requires(self):
        """ The requirements for this task. """
        # If we are running in batch mode then the ApStar keywords will all be tuples, and we would have to
        # add the requirement for every single ApStarFile. That adds overhead, and we don't need to do it:
        # Astra will manage the batches to be expanded into individual tasks.
        if self.is_batch_mode:
            return []
        return dict(observation=self.clone(ApStarFile))


    def output(self):
        """ Outputs of this task. """
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        
        requirements = {
            "database": FerreResult(self),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements


    def read_input_observations(self, **kwargs):
        """ Read the input observations. """
        # Since ApStar files contain the combined spectrum and all individual visits, we are going
        # to supply a data_slice parameter to only return the first spectrum.        
        spectra = []
        for task in self.get_batch_tasks():
            spectra.append(Spectrum1D.read(
                task.input()["observation"].path,
                data_slice=(slice(0, 1), slice(None)),
                **kwargs
            ))
        return spectra
    
    
    def get_source_names(self):
        """ Return a list of source names for convenience in FERRE. """
        return [f"{i:.0f}_{telescope}_{obj}" for i, (telescope, obj) in enumerate(zip(self.telescope, self.obj))]


class FerreGivenApStarFile(ApStarMixin, FerreBase):
    grid_header_path = astra.Parameter()



class InitialEstimateOfStellarParametersGivenApStarFile(ApStarMixin, FerreMixin):
    
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
            with tqdm(desc="Dispatching", total=self.get_batch_size()) as pbar:
                for iteration, source_kwds, kwds in self.dispatcher():
                    pbar.update(iteration)
                    self._requirements.append(self.clone(FerreGivenApStarFile, **kwds))
            return self._requirements
            

    def output(self):
        """ Outputs of this task. """
        # We are only going to list an output if the star has initial parameters that fall within a grid.
        tasks = set([requirement.clone(self.__class__) for requirement in self.requires()])
        outputs = [dict(database=FerreResultProxy(task)) for task in tasks]
        return outputs if self.is_batch_mode else outputs[0]


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

            result = output["database"].read(as_dict=True)
            log_chisq_fit = result["log_chisq_fit"]

            parsed_header = utils.parse_header_path(task.grid_header_path)

            # Penalise chi-sq in the same way they did for DR16.
            # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L492
            if parsed_header["spectral_type"] == "GK" and result["TEFF"] < 3985:
                # \chi^2 *= 10
                log_chisq_fit += np.log(10)

            if log_chisq_fit < best_tasks[key][0]:
                best_tasks[key] = (log_chisq_fit, task)

        # The output for this task is a proxy database row that points to the FERRE result.
        for output in self.output():
            log_chisq_fit, aux_task = best_tasks[uid(output["database"].task)]           
            output["database"].write(dict(proxy_task_id=aux_task.task_id))

        return None


def dispatch_apstars_for_analysis(sources, grid_header_list_path, release=None, public=True, mirror=False):

    with open(grid_header_list_path, "r") as fp:
        grid_header_paths = list(map(str.strip, fp.readlines()))
    
    grid_limits = utils.parse_grid_limits(grid_header_paths)

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
            kwds = {
                "mean_fiber": header["MEANFIB"],
                "teff": teff,
                "logg": logg,
                "fe_h": fe_h
            }

        except:
            log.exception("Exception: ")
            continue

        else:
            source_kwds = source.copy()
            source_kwds.update(initial_parameters={
                "TEFF": teff,
                "LOGG": logg,
                "METALS": fe_h,
                "LOG10VDOP": utils.approximate_log10_microturbulence(logg),
                "O Mg Si S Ca Ti": 0,
                "LGVSINI": 0,
                "C": 0,
                "N": 0,
            })

            any_suitable_grids = False
            for grid_header_path, parsed_header_path in utils.yield_suitable_grids(grid_limits, **kwds):
                any_suitable_grids = True
                
                # In the initial FERRE run we freeze LOG10VDOP.
                all_kwds = source_kwds.copy()
                all_kwds.update(
                    grid_header_path=grid_header_path,
                    frozen_parameters=dict(LOG10VDOP=None),
                )
                # Freeze C and N to zero if this is a dwarf grid.
                if parsed_header_path["gd"] == "d":
                    all_kwds["frozen_parameters"].update(C=0.0, N=0.0)

                # We yield an integer so we can see progress of unique objects.
                yield (i, source, all_kwds)
                




class CreateMedianFilteredApStarFile(ApStarMixin, FerreMixin):

    median_filter_width = astra.IntParameter(default=151)
    bad_minimum_flux = astra.FloatParameter(default=0.01)
    non_finite_err_value = astra.FloatParameter(default=1e10)

    def requires(self):
        return {
            "observation": self.clone(ApStarFile),
            "initial_estimate": self.clone(InitialEstimateOfStellarParametersGivenApStarFile)
        }


    def run(self):

        for task in self.get_batch_tasks():

            # Resolve the proxy FERRE result to get the original initial estimate
            # (and to be able to access all of its inputs and outputs)
            proxy = task.input()["initial_estimate"]
            initial_estimate = proxy["database"].resolve().task

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

            if False:
                observation, = task.read_input_observations()
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12,3))

                for i, moo in enumerate(segment_indices):
                    ax.plot(
                        observation.wavelength.value[moo[0]:moo[1]],
                        image[1].data[0][moo[0]:moo[1]],
                        c="tab:blue"
                    )
                    ax.plot(
                        observation.wavelength.value[moo[0]:moo[1]],
                        image[1].data[0][moo[0]:moo[1]] / continuum[moo[0]:moo[1]],
                        c="k"
                    )

                    ax.plot(
                        observation.wavelength.value[moo[0]:moo[1]],
                        0.1 + continuum[moo[0]:moo[1]],
                        c="tab:red"
                    )
                fig.savefig("tmp.png")


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



class EstimateStellarParametersGivenApStarFile(ApStarMixin, FerreMixin):

    def requires(self):
        """
        The requirements of this task include a median-filtered ApStar file, and an initial
        estimate of the stellar parameters (based on a series of previous FERRE executions).
        """
        return {
            "observation": self.clone(CreateMedianFilteredApStarFile),
            "initial_estimate": self.clone(InitialEstimateOfStellarParametersGivenApStarFile)
        }

    
    def run(self):
        """ Execute this task. """

        execute_tasks = []
        for task in self.get_batch_tasks():
            # From the initial estimate we need the grid_header_path, and the previous stellar parameters
            # (which we will use for the initial guess here.)
            output = task.input()["initial_estimate"]["database"].read(as_dict=True, include_parameters=True)
            headers = utils.read_ferre_headers(output["grid_header_path"])

            execute_tasks.append(task.clone(
                FerreGivenApStarFile,
                grid_header_path=output["grid_header_path"],
                initial_parameters={ k: v for k, v in output.items() if k in headers[0]["LABEL"] }
            ))
        
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
            "database": FerreResultProxy(self),
            "AstraSource": AstraSource(self)
        }
        if not self.write_source_output:
            requirements.pop("AstraSource")
        return requirements





class EstimateChemicalAbundanceGivenApStarFile(ApStarMixin, FerreMixin):

    # Element is not a batch parameter: we run one element for many stars.
    element = astra.Parameter()
    
    def requires(self):
        """
        This task requires a median-filtered ApStar file, and the previously 
        determined stellar parameters.
        """
        #if self.is_batch_mode:
        #    return [task.requires() for task in self.get_batch_tasks()]

        return {
            "observation": self.clone(CreateMedianFilteredApStarFile),
            "stellar_parameters": self.clone(EstimateStellarParametersGivenApStarFile)
        }


    def run(self):

        execute_tasks = []
        headers = {}
        for task in self.get_batch_tasks():

            # We need the grid_header_path and the previously determined stellar parameters.
            output = task.input()["stellar_parameters"]["database"].read(as_dict=True, include_parameters=True)
            grid_header_path = output["grid_header_path"]

            try:
                header = headers[grid_header_path]
            except KeyError:
                header = headers[grid_header_path] = utils.read_ferre_headers(grid_header_path)

            label_names = header[0]["LABEL"]
            parameter_search_indices_one_indexed, ferre_kwds = get_abundance_keywords(task.element)
            
            frozen_parameters = { 
                label_name: None for i, label_name in enumerate(label_names, start=1) \
                if i not in parameter_search_indices_one_indexed
            }
            
            execute_tasks.append(task.clone(
                FerreGivenApStarFile,
                # TODO: Need to put all the speclib contents in a nicer way together.
                input_weights_path=f"/uufs/chpc.utah.edu/common/home/u6020307/astra-component-data/FERRE/masks/{task.element}.mask",
                grid_header_path=output["grid_header_path"],
                initial_parameters={ k: v for k, v in output.items() if k in label_names },
                frozen_parameters=frozen_parameters,
                ferre_kwds=ferre_kwds,
                # Don't write AstraSource objects for chemical abundances.
                write_source_output=False
            ))
        
        outputs = yield execute_tasks

        # Copy outputs from the executed tasks.
        for task, output in zip(self.get_batch_tasks(), outputs):
            for key, target in output.items():
                task.output()[key].copy_from(target)

        return None

        
    @property
    def output_abundances(self):
        """ A convenience function to return the output abundance given the FERRE result. """
        if self.is_batch_mode:
            return [task.output_abundances for task in self.get_batch_tasks()]

        output_target = self.output()["database"].resolve()
        task = output_target.task

        # Use the frozen parameters to figure out which label we should be returning.
        label_names = tuple(set(task.initial_parameters).difference(task.frozen_parameters))
        output = output_target.read(as_dict=True)

        # TODO: Return uncertainty as well.
        return { label_name: output[label_name] for label_name in label_names }


    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]
        return {
            "database": FerreResultProxy(self)
        }



class EstimateChemicalAbundancesGivenApStarFile(ApStarMixin, FerreMixin):

    elements = (
        "Al", "Ca", "Ce", "C", "CN", "Co", "Cr", "Cu", "Mg", "Mn", "Na", "Nd", "Ni", "N", "O", "P",
        "Rb", "Si", "S", "Ti", "V", "Yb"
    )

    def requires(self):
        """
        This task requires a median-filtered ApStar file, and the previously 
        determined stellar parameters.
        """
        klass = EstimateChemicalAbundanceGivenApStarFile
        return { 
            element: self.clone(klass, element=element) for element in self.elements
        }
            

    def run(self):

        for i, (task, outputs) in enumerate(zip(self.get_batch_tasks(), self.output())):

            for element, output in outputs.items():
                aux_task = task.requires()[element]

                # Copy database result.
                output["database"].copy_from(aux_task.output()["database"].resolve())
                
        return None

    
    def output(self):
        if self.is_batch_mode:
            return [task.output() for task in self.get_batch_tasks()]

        return {
            element: dict(database=FerreResultProxy(self)) for element in self.elements
        }
        


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
    

if __name__ == "__main__":


    foo = get_abundance_keywords("Fe")    



