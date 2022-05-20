
import numpy as np
import os
import pickle
from collections import ChainMap
from astra import log, __version__
from astra.utils import (executable, expand_path, flatten, logarithmic_tqdm, dict_to_list)
from astra.base import ExecutableTask, Parameter, TupleParameter, DictParameter
from astra.database.astradb import (database, DataProduct, Output, TaskOutput, TaskOutputDataProducts, ThePayneOutput)
from astra.tools.spectrum import Spectrum1D
from astra.contrib.thepayne import (training, test as testing)

from astra.operators.sdss import get_apvisit_metadata


class TrainThePayne(ExecutableTask):

    n_steps = Parameter(default=100_000)
    n_neurons = Parameter(default=300)
    weight_decay = Parameter(default=0)
    learning_rate = Parameter(default=0.001)

    def execute(self):

        # TODO: do not allow it to be run in bundled mode.
        task = self.context["tasks"][0]
        input_data_product = flatten(self.input_data_products)[0]

        wavelength, label_names, \
            training_labels, training_spectra, \
            validation_labels, validation_spectra = training.load_training_data(input_data_product.path)
    
        state, model, optimizer = training.train(
            training_spectra, 
            training_labels,
            validation_spectra,
            validation_labels,
            label_names,
            n_neurons=self.n_neurons,
            n_steps=self.n_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Create output data product.
        output = Output.create()
        path = expand_path(f"$MWM_ASTRA/{__version__}/thepayne/{output.id}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data_product = DataProduct.create(
            release=input_data_product.release,
            filetype="full",
            kwargs=dict(full=path)
        )

        TaskOutputDataProducts.create(task=task, data_product=data_product)
        with open(data_product.path, "wb") as fp:
            pickle.dump(dict(
                    state=state,
                    wavelength=wavelength,
                    label_names=label_names, 
                ),
                fp
            )
    

class ThePayne(ExecutableTask):

    model_path = Parameter(bundled=True)
    mask_path = Parameter(default=None, bundled=True)

    slice_args = TupleParameter("slice_args", default=None, bundled=False)
    normalization_method = Parameter("normalization_method", default=None, bundled=False)
    normalization_kwds = DictParameter("normalization_kwds", default=None, bundled=False)

    def execute(self):

        log.info(f"Loading model from {self.model_path}")
        state = testing.load_state(self.model_path)

        label_names = state["label_names"]
        L = len(label_names)
        log.debug(f"Label names are ({len(label_names)}): {label_names}")

        # Load any mask.
        if self.mask_path is not None:
            mask = np.load(self.mask_path)
        else:
            mask = None

        results = []
        with logarithmic_tqdm(total=len(self.context["tasks"]), miniters=100) as pb:
                
            for i, (task, data_products, parameters) in enumerate(self.iterable()):
                task_results = []
                for j, data_product in enumerate(flatten(data_products)):
                    
                    #spectrum = Spectrum1D.read(data_product.path)
                    spectrum = self.slice_and_normalize_spectrum(data_product)

                    # Assuming we are using ApStar spectra here, since that's the only training set
                    # given so far.
                    all_spectrum_meta = get_apvisit_metadata(data_product)
                    
                    p_opt, p_cov, model_flux, all_opt_meta = testing.test(
                        spectrum.wavelength.value,
                        spectrum.flux.value,
                        spectrum.uncertainty.array,
                        mask=mask,
                        **state
                    )
                    L_act = min(p_opt.shape[1], L)
                    
                    results = dict_to_list(
                        dict(ChainMap(*[
                            dict(zip(label_names, p_opt.T)),
                            dict(zip(
                                (f"u_{ln}" for ln in label_names),
                                np.sqrt(p_cov[:, np.arange(L_act), np.arange(L_act)].T)
                            ))
                        ]))
                    )
                    task_results.append(results)
                    
                    # Write outputs.
                    for k, (result, opt_meta, meta) in enumerate(zip(results, all_opt_meta, all_spectrum_meta)):
                        with database.atomic():
                            output = Output.create()
                            TaskOutput.create(task=task, output=output)
                            ThePayneOutput.create(
                                task=task,
                                output=output,
                                snr=spectrum.meta["snr"][k],
                                meta=meta,
                                **result,
                                **opt_meta
                            )
                    
                    # TODO: create data products.

                    log.debug(f"Created output {output} for task {task}")

                    task_results.append(result)
                results.append(task_results)
            
            pb.update(1)
        
        return results

    
    def slice_and_normalize_spectrum(self, data_product):

        spectrum = Spectrum1D.read(data_product.path)
        if self.slice_args is not None:
            slices = tuple([slice(*args) for args in self.slice_args])
            spectrum._data = spectrum._data[slices]
            spectrum._uncertainty.array = spectrum._uncertainty.array[slices]
            for key in ("bitmask", "snr"):
                try:
                    spectrum.meta[key] = np.array(spectrum.meta[key])[slices]
                except:
                    log.exception(f"Unable to slice '{key}' metadata with {self.slice_args} on {data_product}")
        
        if self.normalization_method is not None:
            try:
                self._normalizer
            except AttributeError:
                klass = executable(self.normalization_method)
                kwds = self.normalization_kwds or dict()
                self._normalizer = klass(spectrum, **kwds)        
            else:
                self._normalizer.spectrum = spectrum
            finally:
                return self._normalizer()
        else:
            return spectrum
