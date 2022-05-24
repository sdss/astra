import json
import torch
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
from astra import log, __version__
from astra.utils import flatten
from astropy.io import fits
from astra.database.astradb import database, Source,ClassifySourceOutput,  SourceDataProduct, Task, TaskBundle, Bundle, TaskInputDataProducts, DataProduct, TaskOutput, Output, ClassifierOutput
from astra.base import ExecutableTask, Parameter
from astra.tools.spectrum import Spectrum1D, calculate_snr
from tqdm import tqdm

from astra.contrib.classifier import (networks, utils)

CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda:0") if CUDA_AVAILABLE else torch.device("cpu")


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
                                .where(DataProduct.id.in_([dp.id for dp in data_products]))
            )
            log_probs = sum_log_probs(q)
            result = classification_result(log_probs)
            log.info(f"Task {task} with data product {data_products} has result {result}")
            results.append(result)

        total = len(results)
        with tqdm(total=total) as pb:
            for (task, *_), result in zip(self.iterable(), results):
                with database.atomic() as tx:
                    output = Output.create()
                    TaskOutput.create(output=output, task=task)
                    ClassifySourceOutput.create(
                        task=task,
                        output=output,
                        **result
                    )
                    pb.update()

        return results


class ClassifyApVisit(ExecutableTask):

    model_path = Parameter("model_path", bundled=True)

    def execute(self):

        total = len(self.context["tasks"]) # TODO: fix bundle size.
        shape = (3, 4096)
        batch = np.empty((total, *shape), dtype=np.float32)
        dithereds = np.empty(total, dtype=bool)
        snrs = np.empty(total, dtype=float)
        
        iterable = tqdm(self.iterable(), total=self.bundle_size, desc="Loading data")
        for i, (task, data_products, _) in enumerate(iterable):

            with fits.open(data_products[0].path) as image:
                flux = image[1].data
                snrs[i] = image[0].header["SNR"]
            
            try:
                flux = flux.reshape(shape)
            except:
                dithereds[i] = False
                flux = np.repeat(flux, 2).reshape(shape)
            else:
                dithereds[i] = True
            
            continuum = np.nanmedian(flux, axis=1)
            batch[i] = flux / continuum.reshape((-1, 1))

        log.info(f"Loading model from {self.model_path}")
        log.info(f"Using {device}")
        factory = getattr(networks, self.model_path.split("_")[-2])
        model = utils.read_network(factory, self.model_path)
        model.to(device)
        model.eval()
        batch = torch.from_numpy(batch).to(device)

        log.info(f"Making predictions")
        with torch.no_grad():
            prediction = model.forward(batch)
        
        log_probs = prediction.cpu().numpy()

        results = []
        for i, (task, *_) in enumerate(tqdm(self.iterable(), total=total, desc="Updating database")):
            result = classification_result(log_probs[i], model.class_names)
            result.update(
                snr=snrs[i],
                dithered=dithereds[i]
            )
            output = Output.create()
            TaskOutput.create(output=output, task=task)
            ClassifierOutput.create(
                task=task,
                output=output,
                **result
            )
            results.append(result)

        return results
            

class ClassifySpecLite(ExecutableTask):

    model_path = Parameter("model_path", bundled=True)

    def execute(self, **kwargs):

        # TODO: make the spectrum1d loaders fast enough that we don't have to
        #       do this,.. or have some option for the 1D loader so that we
        #       can select just the things we need (in this case, a flux array)
        use_spectrum1d_loader = kwargs.get("use_spectrum1d_loader", False) # MAGIC

        log.info(f"Loading model from {self.model_path}")

        factory = getattr(networks, self.model_path.split("_")[-2])
        model = utils.read_network(factory, self.model_path)
        model.to(device)
        model.eval()

        si, ei = (0, 3800) # MAGIC: same done in training

        log.info(f"Using device {device}")
        
        total = len(self.context["tasks"]) # TODO: fix bundle size.
        batch = np.empty((total, 1, ei - si), dtype=np.float32)

        iterable = tqdm(self.iterable(), total=total, desc="Loading data")

        for i, (task, data_products, *_) in enumerate(iterable):
            assert len(data_products) == 1
            if use_spectrum1d_loader:
                spectrum = Spectrum1D.read(data_products[0].path)
                flux = spectrum.flux.value[0, si:ei]        
                continuum = np.nanmedian(flux)
            else:
                with fits.open(data_products[0].path) as image:
                    flux = image[1].data["flux"][si:ei]
                continuum = np.nanmedian(flux)
            
            batch[i, 0, :] = flux/continuum

        log.debug(f"Formatting data")
        batch = torch.from_numpy(batch).to(device)

        log.debug(f"Making predictions")
        with torch.no_grad():
            prediction = model.forward(batch)
        log_probs = prediction.cpu().numpy()

        log.debug("Updating database")
        results = []
        for (task, *_), log_prob in tqdm(zip(self.iterable(), log_probs), total=total):
            result = classification_result(log_prob, model.class_names)
            result.update(
                snr=-1,         # no S/N measured here, and none in BOSS headers
                dithered=False, # not relevant for BOSS spectra
            )
            output = Output.create()
            TaskOutput.create(output=output, task=task)
            ClassifierOutput.create(
                task=task,
                output=output,
                **result
            )
            results.append(result)
        return results
            

def create_task_bundle_for_source_classification(visit_classification_bundle_id):
    sources = (
        Source.select()
              .distinct(Source)
              .join(SourceDataProduct)
              .join(DataProduct)
              .join(TaskInputDataProducts)
              .join(Task)
              .join(TaskBundle)
              .join(Bundle)
              .where(Bundle.id == int(visit_classification_bundle_id))
    )

    # for each source, get the classifier outputs for all visits
    all_data_product_ids = []
    for source in sources:
        q = (
            ClassifierOutput.select(DataProduct.id)
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
        all_data_product_ids.append(flatten(q))
        
    with database.atomic() as txn:
        bundle = Bundle.create()
        for data_product_ids in all_data_product_ids:
            task = Task.create(
                name="astra.contrib.classifier.ClassifySource",
                parameters={},
                version=__version__
            )
            log.debug(f"Created task {task} with {len(data_product_ids)} inputs")
            for data_product_id in data_product_ids:
                TaskInputDataProducts.create(
                    task=task,
                    data_product_id=data_product_id
                )
            TaskBundle.create(task=task, bundle=bundle)

    log.info(f"Created task bundle {bundle}")
    return bundle.id


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
