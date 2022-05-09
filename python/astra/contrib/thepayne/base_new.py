
import numpy as np
import os
import pickle
from collections import ChainMap
from astra import log, __version__
from astra.utils import (expand_path, flatten, logarithmic_tqdm)
from astra.base import ExecutableTask, Parameter, TupleParameter, DictParameter
from astra.database.astradb import (database, DataProduct, Output, TaskOutput, TaskOutputDataProducts, ThePayneOutput)
from astra.tools.spectrum import Spectrum1D
from astra.contrib.thepayne import (training, testing)

class ThePayneTrainStep(ExecutableTask):

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
    

class ThePayneTestStep(ExecutableTask):

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
                    
                    # TODO: slice args here + normalization
                    spectrum = Spectrum1D.read(data_product.path)

                    p_opt, p_cov, model_flux, meta = testing.test(
                        spectrum.wavelength.value,
                        spectrum.flux.value,
                        spectrum.uncertainty.array,
                        mask=mask,
                        **state
                    )
                    
                    result = dict(ChainMap(*[
                        dict(zip(label_names, p_opt.T)),
                        dict(zip(
                            (f"u_{ln}" for ln in label_names),
                            np.sqrt(p_cov[:, np.arange(L), np.arange(L)].T)
                        ))
                    ]))

                    # p_opt is probably large if spectrum.flux.array is large, check 
                    raise a

                    # Write outputs.
                    with database.atomic():
                        output = Output.create()
                        TaskOutput.create(task=task, output=output)
                        ThePayneOutput.create(
                            task=task,
                            output=output,
                            **result
                        )
                    
                    # TODO: create data products.

                    log.debug(f"Created output {output} for task {task}")

                    task_results.append(result)
                results.append(task_results)
            
            pb.update(1)
        
        return results

    

