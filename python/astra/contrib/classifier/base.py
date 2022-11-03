
from astra.tools.spectrum import Spectrum1D

import json
import torch
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
from collections import OrderedDict
from astra import log, __version__
from astra.utils import flatten, expand_path
from astropy.io import fits
from astra.database.astradb import (
    database,
    Source,
    ClassifySourceOutput,
    SourceDataProduct,
    Task,
    TaskBundle,
    Bundle,
    TaskInputDataProducts,
    DataProduct,
    TaskOutput,
    Output,
    ClassifierOutput,
)
from astra.base import TaskInstance, Parameter, TupleParameter
#from astra.tools.spectrum import Spectrum1D, calculate_snr
from tqdm import tqdm

from astra.contrib.classifier import networks, utils

CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda:0") if CUDA_AVAILABLE else torch.device("cpu")



class ClassifySource(TaskInstance):

    output_ids = TupleParameter()

    def execute(self):
        for task, data_products, parameters in tqdm(self.iterable()):
            # Get the visit classifications for these data products. 
            q = (
                ClassifierOutput
                .select()
                .where(ClassifierOutput.output_id.in_(list(map(int, parameters["output_ids"]))))
            )
            log_probs = sum_log_probs(q)
            result = classification_result(log_probs)
            result.update(source_id=q[0].source_id)
            task.create_or_update_outputs(ClassifySourceOutput, [result])
        return None


class ClassifyApVisit(TaskInstance):

    model_path = Parameter(
        default="$MWM_ASTRA/component_data/classifier/classifier_NIRCNN_77804646.pt",
        bundled=True
    )

    def execute(self):

        total = len(self.context["tasks"])  # TODO: fix bundle size.
        shape = (3, 4096)
        batch = np.nan * np.ones((total, *shape), dtype=np.float32)
        dithereds = np.empty(total, dtype=bool)
        snrs = np.empty(total, dtype=float)
        source_ids = []
        parent_data_product_ids = []

        iterable = tqdm(self.iterable(), total=self.bundle_size, desc="Loading data")
        for i, (task, data_products, _) in enumerate(iterable):

            try:
                with fits.open(data_products[0].path) as image:
                    flux = image[1].data
                    snrs[i] = image[0].header["SNR"]
                    source_ids.append(image[0].header["CATID"])
            except:
                log.exception(f"Exception while loading {data_products[0].path}")
                source_ids.append(None)
                continue

            try:
                flux = flux.reshape(shape)
            except:
                faux_dither_flux = np.empty(shape)
                for j in range(3):
                    faux_dither_flux[j, ::2] = flux[j]
                    faux_dither_flux[j, 1::2] = flux[j]
                flux = faux_dither_flux
                dithereds[i] = False
            else:
                dithereds[i] = True

            continuum = np.nanmedian(flux, axis=1)
            batch[i] = flux / continuum.reshape((-1, 1))
            parent_data_product_ids.append(data_products[0].id)


        log.info(f"Loading model from {self.model_path}")
        log.info(f"Using {device}")
        
        model_path = expand_path(self.model_path)
        factory = getattr(networks, model_path.split("_")[-2])
        model = utils.read_network(factory, model_path)
        model.to(device)
        model.eval()
        batch = torch.from_numpy(batch).to(device)

        log.info(f"Making predictions")
        with torch.no_grad():
            prediction = model.forward(batch)

        log_probs = prediction.cpu().numpy()

        for i, (task, (data_product, ), *_) in enumerate(
            tqdm(self.iterable(), total=total, desc="Updating database")
        ):
            if source_ids[i] is None: continue
            result = classification_result(log_probs[i], model.class_names)
            result.update(
                snr=snrs[i], 
                dithered=dithereds[i],
                source_id=source_ids[i],
                parent_data_product_id=data_product.id,
            )
            task.create_or_update_outputs(ClassifierOutput, [result])



class ClassifySpecFull(TaskInstance):

    model_path = Parameter(default="$MWM_ASTRA/component_data/classifier/classifier_OpticalCNN_40bb9164.pt", bundled=True)

    def execute(self, **kwargs):

        model_path = expand_path(self.model_path)

        log.info(f"Loading model from {model_path}")

        factory = getattr(networks, model_path.split("_")[-2])
        model = utils.read_network(factory, model_path)
        model.to(device)
        model.eval()

        si, ei = (0, 3800)  # MAGIC: same done in training

        log.info(f"Using device {device}")

        total = len(self.context["tasks"])  # TODO: fix bundle size.
        #batch = np.empty((total, 1, ei - si), dtype=np.float32)
        batch = []
        snrs = []
        source_ids = []
        data_product_ids = []

        iterable = tqdm(self.iterable(), total=total, desc="Loading data")
        for i, (task, (data_product, ), *_) in enumerate(iterable):
                
            # TODO: only assuming BOSS/APO data, and only using the stack.
            try:
                spectrum = Spectrum1D.read(data_product.path, hdu=1)
            except:
                snrs.append(None)
                source_ids.append(None)
                data_product_ids.append(None)
                batch.append(np.nan * np.ones(ei - si))
                continue

            flux = np.atleast_2d(spectrum.flux.value)[0, si:ei]
            continuum = np.nanmedian(flux)
            rectified_flux = flux / continuum
            # Remove NaNs
            finite = np.isfinite(rectified_flux)
            if any(finite):
                rectified_flux[~finite] = np.interp(
                    spectrum.wavelength.value[si:ei][~finite],
                    spectrum.wavelength.value[si:ei][finite],
                    rectified_flux[finite],
                )

            #batch[i, 0, :] = rectified_flux
            batch.append(rectified_flux)
            snrs.append(spectrum.meta.get("SNR", np.array([np.nan])).flatten()[0])
            source_id = spectrum.meta.get("CAT_ID", None)
            if source_id is None:
                source_id = data_product.kwargs["catalogid"]

            source_ids.append(source_id)
            data_product_ids.append(data_product.id)
            
        log.debug(f"Formatting data")
        batch = np.array(batch).reshape((-1, 1, ei - si)).astype(np.float32)
        batch = torch.from_numpy(batch).to(device)

        log.debug(f"Making predictions")
        with torch.no_grad():
            prediction = model.forward(batch)
        
        log_probs = prediction.cpu().numpy()

        for i, (task, (data_product, ), *_) in enumerate(self.iterable()):
            if source_ids[i] is None:
                continue

            result = classification_result(log_probs[i], model.class_names)
            result.update(
                snr=snrs[i],
                source_id=source_ids[i],
                parent_data_product_id=data_product.id,
            )

            task.create_or_update_outputs(ClassifierOutput, [result])

        return None


def create_source_classification_bundle(apogee_bundle_ids, boss_bundle_ids):

    if isinstance(apogee_bundle_ids, str):
        apogee_bundle_ids = json.loads(apogee_bundle_ids)
    if isinstance(boss_bundle_ids, str):
        boss_bundle_ids = json.loads(boss_bundle_ids)
    
    log.info(f"There are {len(apogee_bundle_ids)} APOGEE bundles")
    log.info(f"There are {len(boss_bundle_ids)} BOSS bundles")

    q = (
        ClassifierOutput
        .select(
            ClassifierOutput.source_id,
            ClassifierOutput.output_id,
            ClassifierOutput.parent_data_product_id
        )
        .join(Task, on=(ClassifierOutput.task_id == Task.id))
        .join(TaskBundle)
        .where(TaskBundle.bundle_id.in_(apogee_bundle_ids) | TaskBundle.bundle_id.in_(boss_bundle_ids))
        .tuples()
    )

    outputs = OrderedDict()
    data_products = OrderedDict()
    for source_id, output_id, data_product_id in q:
        outputs.setdefault(source_id, [])
        outputs[source_id].append(output_id)
        data_products.setdefault(source_id, [])
        data_products[source_id].append(data_product_id)
    
    tasks = []
    for source_id, output_ids in outputs.items():
        tasks.append(
            Task(
                name="astra.contrib.classifier.ClassifySource", 
                parameters=dict(output_ids=output_ids), 
                version=__version__
            )
        )

    log.info(f"Found {len(outputs)} sources")
    
    with database.atomic():
        Task.bulk_create(tasks)

        # assign the data products as inputs
        rows = []
        for task, (source_id, data_product_ids) in zip(tasks, data_products.items()):
            for data_product_id in data_product_ids:
                rows.append(dict(task_id=task.id, data_product_id=data_product_id))
        TaskInputDataProducts.insert_many(rows).execute()

        bundle = Bundle.create()
        TaskBundle.insert_many([dict(task_id=task.id, bundle_id=bundle.id) for task in tasks]).execute()
    
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
            raise TypeError(
                f"If class_names is None then log_probs must be a dictionary"
            )
        class_names, log_probs = zip(*log_probs.items())

    log_probs = np.array(log_probs).flatten()
    # Calculate normalized probabilities.
    with np.errstate(under="ignore"):
        relative_log_probs = log_probs - logsumexp(log_probs)#, axis=1)[:, None]

    # Round for PostgreSQL 'real' type.
    # https://www.postgresql.org/docs/9.1/datatype-numeric.html
    # and
    # https://stackoverflow.com/questions/9556586/floating-point-numbers-of-python-float-and-postgresql-double-precision
    probs = np.round(np.exp(relative_log_probs), decimals)
    log_probs = np.round(log_probs, decimals)

    result = {f"p_{cn}": p for cn, p in zip(class_names, probs.T)}
    result.update({f"lp_{cn}": p for cn, p in zip(class_names, log_probs.T)})
    return result
