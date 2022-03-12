import json
import torch
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
from astra import log, __version__
from astra.utils import flatten
from astra.database.astradb import database, Source,ClassifySourceOutput,  SourceDataProduct, Task, TaskBundle, Bundle, TaskInputDataProducts, DataProduct, TaskOutput, Output, ClassifierOutput
from astra.base import ExecutableTask, Parameter
from astra.tools.spectrum import Spectrum1D

from astra.contrib.classifier import (networks, utils)



class ClassifySource(ExecutableTask):

    #model_path = Parameter("model_path", bundled=True)

    def execute(self, context=None):
        results = []
        for task, data_products, parameters in self.iterable():
            # Get the visit classifications for these data products.
            q = (
                ClassifierOutput.select()
                                .join(Task)
                                .join(TaskInputDataProducts)
                                .join(DataProduct)
                                .where(DataProduct.pk.in_([dp.pk for dp in data_products]))
            )
            log_probs = sum_log_probs(q)
            results.append(classification_result(log_probs))
        return results

    def post_execute(self):
        total = len(self.result)
        with tqdm(total=total) as pb:
            for (task, *_), result in zip(self.iterable(), self.result):
                with database.atomic() as tx:
                    output = Output.create()
                    TaskOutput.create(output=output, task=task)
                    ClassifySourceOutput.create(
                        task=task,
                        output=output,
                        **result
                    )
                    pb.update()

class Classify(ExecutableTask):

    # TODO: Replace this to be a model data product?
    model_path = Parameter("model_path", bundled=True)

    def execute(self, context=None):

        log.info(f"Loading model from {self.model_path}")

        factory = getattr(networks, self.model_path.split("_")[-2])
        model = utils.read_network(factory, self.model_path)
        model.eval()

        results = []
        for task, data_products, parameters in self.iterable():
            
            assert len(data_products) == 1
            spectrum = Spectrum1D.read(data_products[0].path)
            flux = spectrum.flux.value

            try:
                flux = flux.reshape((1, 3, 4096))
            except:
                dithered = False
                flux = np.repeat(flux, 2).reshape((1, 3, 4096))
            else:
                dithered = True
                
            continuum = np.nanmedian(flux, axis=2).reshape((-1, 1))
            normalized_flux = torch.from_numpy((flux / continuum).astype(np.float32))

            with torch.no_grad():
                prediction = model.forward(normalized_flux)
                log_probs = prediction.cpu().numpy()

            result = classification_result(log_probs, model.class_names)
            result.update(
                snr=spectrum.meta["snr"],
                dithered=dithered,                
            )
            results.append(result)

        return results
            

    def post_execute(self):

        total = len(self.result)
        with tqdm(total=total) as pb:
            for (task, *_), result in zip(self.iterable(), self.result):
                with database.atomic() as tx:
                    output = Output.create()
                    TaskOutput.create(output=output, task=task)
                    ClassifierOutput.create(
                        task=task,
                        output=output,
                        **result
                    )
                    pb.update()



def create_task_bundle_for_source_classification(visit_classification_bundle_pk):
    sources = (
        Source.select()
              .distinct(Source)
              .join(SourceDataProduct)
              .join(DataProduct)
              .join(TaskInputDataProducts)
              .join(Task)
              .join(TaskBundle)
              .join(Bundle)
              .where(Bundle.pk == int(visit_classification_bundle_pk))
    )

    # for each source, get the classifier outputs for all visits
    all_data_product_pks = []
    for source in sources:
        q = (
            ClassifierOutput.select(DataProduct.pk)
                            .join(Task)
                            .join(TaskInputDataProducts)
                            .join(DataProduct)
                            .join(SourceDataProduct)
                            .join(Source)
                            .where(Source.catalogid == source.catalogid)
                            .tuples()
        )
        q = flatten(q)
        log.debug(f"Source {source} has {len(q)} data products: {q}")
        all_data_product_pks.append(flatten(q))
        
    with database.atomic() as txn:
        bundle = Bundle.create()
        for data_product_pks in all_data_product_pks:
            task = Task.create(
                name="astra.contrib.classifier.ClassifySource",
                parameters={},
                version=__version__
            )
            log.debug(f"Created task {task} with {len(data_product_pks)} inputs")
            for data_product_pk in data_product_pks:
                TaskInputDataProducts.create(
                    task=task,
                    data_product_id=data_product_pk
                )
            TaskBundle.create(task=task, bundle=bundle)

    log.info(f"Created task bundle {bundle}")
    return bundle.pk


def sum_log_probs(iterable):
    log_probs = {}
    for result in iterable:
        for attr in result._meta.fields.keys():
            if attr.startswith("lp_"):
                name = attr[3:]
                log_probs.setdefault(name, 0)
                value = getattr(result, attr)
                if np.isfinite(value):
                    log_probs[name] += value
    return log_probs

def classification_result(log_probs, class_names=None, decimals=30):

    if class_names is None:
        if not isinstance(log_probs, dict):
            raise TypeError(f"If class_names is None then log_probs must be a dictionary")        
        class_names, log_probs = zip(*log_probs.items())

    log_probs = np.array(log_probs).flatten()
    # Calculate normalized probabilities.
    with np.errstate(under="ignore"):
        relative_log_probs = log_probs - logsumexp(log_probs)

    # Round for PostgreSQL 'real' type.
    # https://www.postgresql.org/docs/9.1/datatype-numeric.html
    # and
    # https://stackoverflow.com/questions/9556586/floating-point-numbers-of-python-float-and-postgresql-double-precision
    probs = np.round(np.exp(relative_log_probs), decimals)
    log_probs = np.round(log_probs, decimals)
    
    result = { f"p_{cn}": p for cn, p in zip(class_names, probs) }
    result.update({ f"lp_{cn}": p for cn, p in zip(class_names, log_probs) })    
    return result