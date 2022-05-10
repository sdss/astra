import numpy as np
import os
import pickle
from astra import log, __version__
from astra.base import ExecutableTask, Parameter, DictParameter, TupleParameter
from astra.database.astradb import Task, DataProduct, TaskInputDataProducts, database, TaskOutputDataProducts
from astropy.table import Table

from astra.tools.spectrum import Spectrum1D
from astra.utils import flatten, executable, expand_path
from sdss_access import SDSSPath
from tqdm import tqdm


class CreateLabelledSetFromTable(ExecutableTask):

    """
    Create a labelled set for The Cannon given some summary table that
    contains columns to identify the data product, and expert labels.
    """

    path = Parameter("path", bundled=True)
    hdu = Parameter("hdu", default=1, bundled=True)

    label_names = TupleParameter("label_names", bundled=True)
    limits = DictParameter("limits", default=None, bundled=True)

    release = Parameter("release", default=None, bundled=True)
    filetype = Parameter("filetype", default=None, bundled=True)

    translate = DictParameter("translate", default=None, bundled=True)
    defaults = DictParameter("defaults", default=None, bundled=True)

    normalization_method = Parameter("normalization_method", default=None, bundled=True)
    normalization_kwds = DictParameter("normalization_kwds", default=None, bundled=True)
    slice_args = TupleParameter("slice_args", default=None, bundled=True)

    auto_translations = {
        "prefix": lambda c: np.array([f"{ea}".lstrip()[:2] for ea in c("file")]),
    }

    convert_x_fe_columns_to_x_h = Parameter("convert_x_fe_to_x_h", default=False, bundled=True)

    def execute(self):
        """ Execute the task. """

        table = Table.read(expand_path(self.path), hdu=self.hdu)
        field_names = list(table.dtype.names)
        field_names_lower = list(map(str.lower, field_names))
        
        def get_column(name):
            # Case insensitive column name
            name = name.lower()
            try:
                return table[field_names[field_names_lower.index(name.lower())]]
            except:
                if name in self.translate:
                    return get_column(self.translate[name])
                else:
                    if name in self.auto_translations:
                        callable = self.auto_translations[name]
                        return callable(get_column)
                
                raise ValueError(f"Unable to find column '{name}' in table {self.path}, or among any translations")

        unknown = set(list(map(str.lower, self.label_names))).difference(field_names_lower)
        if unknown:
            raise ValueError(f"{len(unknown)} unknown label names: {unknown} not among {field_names_lower}")
            
        paths = SDSSPath(release=self.release)

        keys = paths.lookup_keys(self.filetype)
        missing = set(keys).difference(field_names_lower)
        missing = missing.difference(self.defaults or dict())
        missing = missing.difference(self.auto_translations)
        if self.translate is not None:
            missing = missing.difference(self.translate)
        if missing:
            raise ValueError(
                f"{self.release} file of type {self.filetype} requires {missing} keys. Use `defaults` to set defaults or "
                f"Use `translate` parameter like `translate=dict(missing_column_name='use_column_name')` to translate."
            )
        
        # Define the subset.
        N = len(table)
        subset = np.ones(N, dtype=bool)
        if self.limits is not None:
            for key, value in self.limits.items():
                if isinstance(value, (tuple, list, np.ndarray)):
                    if len(value) != 2:
                        raise ValueError(f"{key} value must be a single value or a two-length tuple")

                    # upper/lower
                    try:
                        lower, upper = sorted(value)
                    except: # can't sort things like (int, None)
                        lower, upper = value
                    values = get_column(key)
                    if lower is not None:
                        subset *= (values >= lower)
                    if upper is not None:
                        subset *= (values <= upper)

                    N_now = subset.sum()
                    log.debug(f"Requiring {key} between {lower} and {upper} removed {N - N_now} rows")

                elif isinstance(value, (int, float, str)):
                    # Single value.
                    subset *= (get_column(key) == value)
                    N_now = subset.sum()
                    log.debug(f"Requiring {key} = {value} removed {N - N_now} rows")
                else:
                    raise TypeError(f"Unknown type for {key} = {value}")
                N = N_now

        log.debug(f"{subset.size} rows before applying limits")
        log.debug(f"{subset.sum()} rows after applying limits")
        N = subset.sum()

        # Create data products.
        log.info(f"Finding data products ...")
        data_products = []
        missing_paths = []

        # Get all keywords first.
        defaults = self.defaults or dict()
        non_default_keys = set(keys).difference(defaults)
        data_product_kwargs = { k: get_column(k) for k in non_default_keys }

        for index in tqdm(np.where(subset)[0]):
            kwargs = { k: data_product_kwargs[k][index] for k in non_default_keys }
            kwargs.update(defaults)
            path = paths.full(self.filetype, **kwargs)
            if not os.path.exists(path):
                log.warning(f"Row index {index} yields data path {path} that does not exist.")
                missing_paths.append(index)
                continue

            data_product, created = DataProduct.get_or_create(
                release=self.release,
                filetype=self.filetype,
                kwargs=kwargs
            )
            data_products.append(data_product)

        # Skip over missing paths.
        log.warning(f"There were {len(missing_paths)} missing paths.")
        for index in missing_paths:
            subset[index] = False

        if self.convert_x_fe_columns_to_x_h:
            labels = {}
            for key in map(str.lower, self.label_names):
                if key.endswith("_fe"):
                    new_key = key.split("_")[0] + "_h"
                    labels[new_key] = np.array(get_column(key)[subset]) + np.array(get_column("fe_h")[subset])
                else:
                    labels[key] = np.array(get_column(key)[subset])
        
        else:
            labels = { key: np.array(get_column(key)[subset]) for key in map(str.lower, self.label_names) }

        wl, flux, ivar, bitmask = ([], [], [], [])
        for data_product in tqdm(data_products, desc="Loading and normalizing data"):
            spectrum = self.slice_and_normalize_spectrum(data_product)

            # We flatten because we want one entry per label.
            wl.append(spectrum.wavelength.value.flatten())
            flux.append(spectrum.flux.value.flatten())
            ivar.append(spectrum.uncertainty.array.flatten())
            bitmask.append(spectrum.meta["bitmask"].flatten())

        wl = np.atleast_2d(wl)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)
        bitmask = np.atleast_2d(bitmask)

        log.debug(f"Data array shapes: wl={wl.shape}, flux={flux.shape}, ivar={ivar.shape}, bitmask={bitmask.shape}")

        # Nicely check if all wavelengths are the same?
        if all(np.allclose(a, b) for a, b in zip(wl[0], wl.T)):
            log.info(f"All wavelengths are the same. Only storing the first entry.")
            wl = wl[[0]]

        # Associate data products to this task.
        this_task, = self.context["tasks"]
        with database.atomic():
            for data_product in data_products:
                TaskInputDataProducts.create(
                    task=this_task,
                    data_product=data_product,
                )

        # Return the training set.
        labelled_set = dict(
            data=dict(
                wavelength=wl,
                flux=flux,
                ivar=ivar,
                bitmask=bitmask
            ),
            labels=labels,
            meta=dict(
                # This is not meant to be a complete metadata description.
                # It's just if someone came across the file and had no access to the database, etc.
                task_id=this_task.id,
                task_name=this_task.name,
                task_parameters=dict(
                    path=self.path,
                    hdu=self.hdu,
                    limits=self.limits,
                    defaults=self.defaults,
                    release=self.release,
                    filetype=self.filetype,
                    translate=self.translate,
                    normalization_method=self.normalization_method,
                    normalization_kwds=self.normalization_kwds,
                    slice_args=self.slice_args
                )
            )
        )
        return labelled_set


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

    def post_execute(self):
        # Create output data product, and write to disk.
        task, = self.context["tasks"]
        path = expand_path(f"$MWM_ASTRA/{__version__}/thecannon/labelled-set-task-{task.id}.pkl")
        log.info(f"Writing labelled set to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as fp:
            pickle.dump(self.result, fp)

        log.info(f"Created labelled set at {path}")
        
        # Create and assign data product.
        data_product = DataProduct.create(
            release=self.release,
            filetype="full",
            kwargs=dict(full=path)
        )
        TaskOutputDataProducts.create(task=task, data_product=data_product)
        log.info(f"Created data product {data_product} and assigned it as output to task {task}")



class SplitLabelledSet(ExecutableTask):
    
    # input data product is the labelled set
    test_set_fraction = Parameter("test_set_fraction", default=0.2)
    training_set_fraction = Parameter("training_set_fraction", default=0.6)
    validation_set_fraction = Parameter("validation_set_fraction", default=0.2)
    release = Parameter("release", default="sdss5")

    random_seed = Parameter("random_seed", default=0)

    def execute(self):

        if self.training_set_fraction + self.validation_set_fraction + self.test_set_fraction != 1:
            raise ValueError(f"Sum of fractions must be 1. Got {self.training_set_fraction + self.validation_set_fraction + self.test_set_fraction}")

        log.debug(f"Using random seed {self.random_seed}")

        input_data_products = self.context["input_data_products"]

        training_set, validation_set, test_set = (
            CreateTrainingSet(
                input_data_products=input_data_products, 
                random_seed=self.random_seed,
                release=self.release,
                fraction=self.training_set_fraction,
                offset_fraction=0
            ),
            CreateValidationSet(
                input_data_products=input_data_products, 
                random_seed=self.random_seed,
                release=self.release,
                fraction=self.validation_set_fraction,
                offset_fraction=self.training_set_fraction
            ),
            CreateTestSet(
                input_data_products=input_data_products,
                random_seed=self.random_seed,
                release=self.release,
                fraction=self.test_set_fraction,
                offset_fraction=self.training_set_fraction + self.validation_set_fraction
            )
        )
        outputs = dict(
            training_set=training_set.execute(),
            validation_set=validation_set.execute(),
            test_set=test_set.execute()
        )
        return outputs



class CreateTypeSetBase(ExecutableTask):

    random_seed = Parameter("random_seed", default=0)
    fraction = Parameter("fraction")
    offset_fraction = Parameter("offset")
    release = Parameter("release", default="sdss5")

    def execute(self):
        
        task = self.context["tasks"][0]
        (labelled_set_dp, *_) = self.context["input_data_products"]
    
        log.info(f"Executing {self} (task {task}) on {labelled_set_dp}")
        
        with open(labelled_set_dp.path, "rb") as fp:
            labelled_set = pickle.load(fp)        
        N, P = labelled_set["data"]["flux"].shape

        np.random.seed(self.random_seed)
        S = int(self.offset_fraction * N)
        E = int((self.offset_fraction + self.fraction) * N)
        indices = np.random.permutation(N)[S:E]

        labelled_set_task, = (
            Task.select()
                .join(TaskOutputDataProducts)
                .where(TaskOutputDataProducts.data_product_id == labelled_set_dp.id)
        )

        input_data_products = []
        for i, input_data_product in enumerate(labelled_set_task.input_data_products):
            if i in indices:
                input_data_products.append(input_data_product)
        
        # Assign those data products to this task.
        with database.atomic():
            for data_product in input_data_products:
                TaskInputDataProducts.create(
                    task=task,
                    data_product=data_product,
                )

        # Create the output data product.
        path = expand_path(f"$MWM_ASTRA/{__version__}/thecannon/{self.__class__.__name__[6:]}-{task.id}.pkl")

        meta = labelled_set["meta"]
        meta["subset_meta"] = dict(
            random_seed=self.random_seed,
            fraction=self.fraction,
            offset_fraction=self.offset_fraction,
            task_id=task.id
        )
        labels = { k: v[indices] for k, v in labelled_set["labels"].items() }
        Nw, P = labelled_set["data"]["wavelength"].shape
        Nf, P = labelled_set["data"]["flux"].shape
        data = { k: v[indices] for k, v in labelled_set["data"].items() if k != "wavelength" }
        data["wavelength"] = labelled_set["data"]["wavelength"]
        if Nf > 1 and Nf == Nw:
            data["wavelength"] = data["wavelength"][indices]

        subset = dict(
            meta=meta,
            labels=labels,
            data=data,
        )
        with open(path, "wb") as fp:
            pickle.dump(subset, fp)
        log.info(f"Created subset at {path}")

        # Create and assign data product.
        with database.atomic():
            data_product = DataProduct.create(
                release=self.release,
                filetype="full",
                kwargs=dict(full=path)
            )
            TaskOutputDataProducts.create(task=task, data_product=data_product)

        log.info(f"Created data product {data_product} and assigned it as output to task {task}")

        # Return the data product ID
        return data_product.id
        

class CreateTrainingSet(CreateTypeSetBase):
    random_seed = Parameter("random_seed", default=0)
    fraction = Parameter("fraction")
    offset_fraction = Parameter("offset")
    release = Parameter("release", default="sdss5")

    # TODO: fix bug where parameters are not inherited by astra tasks

class CreateValidationSet(CreateTypeSetBase):
    random_seed = Parameter("random_seed", default=0)
    fraction = Parameter("fraction")
    offset_fraction = Parameter("offset")
    release = Parameter("release", default="sdss5")

    # TODO: fix bug where parameters are not inherited by astra tasks


class CreateTestSet(CreateTypeSetBase):
    random_seed = Parameter("random_seed", default=0)
    fraction = Parameter("fraction")
    offset_fraction = Parameter("offset")
    release = Parameter("release", default="sdss5")

    # TODO: fix bug where parameters are not inherited by astra tasks
