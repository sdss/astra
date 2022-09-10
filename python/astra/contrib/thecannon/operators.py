import os
import json
import pickle
import numpy as np

from astra.database import AstraBase, astradb, catalogdb, session
from astra.database.utils import get_or_create_task_instance
from astra import log
from tqdm import tqdm
from sqlalchemy import func

from astra.contrib import thecannon as tc
from astra.utils import get_base_output_path


def select_training_set_labels_from_database(
    label_columns, filter_args=None, filter_func=None, limit=None, **kwargs
):
    """
    Construct a set of training labels from a query to the SDSS-V database.

    :param label_columns:
        A tuple of database columns that will be taken as labels to train The Cannon model.
        These should be database columns (e.g., `astra.database.catalogdb.SDSSDR16ApogeeStar.teff`)
        and can be re-labelled using the `.label()` attribute:

        `astra.database.catalogdb.SDSSDR16ApogeeStar.x_h[1].label("c_h")`

    :param filter_args: [optional]
        arguments to supply to the `session.query(...).filter_by()` function to limit the
        number of sources in the training set

    :param filter_func: [optional]
        A callable function that returns True if a row should be kept in the training set,
        or False if it should be removed. This can be used for more complex queries than
        what might be possible using just `filter_args`.

    :param limit: [optional]
        Optionally set a limit for the number of rows returned.

    :returns:
        A dictionary containing label names and a N-length list of values.
    """
    labels, data_model_identifiers = _select_training_set_data_from_database(
        label_columns,
        filter_args=filter_args,
        filter_func=filter_func,
        limit=limit,
        **kwargs,
    )
    return labels


def select_training_set_data_model_identifiers_from_database(
    label_columns, filter_args=None, filter_func=None, limit=None, **kwargs
):
    """
    Construct a set of data model identifiers from a query to the SDSS-V database.

    :param label_columns:
        A tuple of database columns that will be taken as labels to train The Cannon model.
        These should be database columns (e.g., `astra.database.catalogdb.SDSSDR16ApogeeStar.teff`)
        and can be re-labelled using the `.label()` attribute:

        `astra.database.catalogdb.SDSSDR16ApogeeStar.x_h[1].label("c_h")`

    :param filter_args: [optional]
        arguments to supply to the `session.query(...).filter_by()` function to limit the
        number of sources in the training set

    :param filter_func: [optional]
        A callable function that returns True if a row should be kept in the training set,
        or False if it should be removed. This can be used for more complex queries than
        what might be possible using just `filter_args`.

    :param limit: [optional]
        Optionally set a limit for the number of rows returned.

    :returns:
        A N-length list of dictionaries.
    """
    labels, data_model_identifiers = _select_training_set_data_from_database(
        label_columns,
        filter_args=filter_args,
        filter_func=filter_func,
        limit=limit,
        **kwargs,
    )
    return data_model_identifiers


def create_task_instances_for_training_set_data_model_identifiers(
    dag_id,
    task_id,
    run_id,
    label_columns,
    filter_args=None,
    filter_func=None,
    limit=None,
    parameters=None,
    **kwargs,
):
    """
    Perform a datbase query to retrieve data model idenitfiers for a training set, and create
    task instances for each of those data model identifiers.

    :param dag_id:
        the identifier string of the directed acyclic graph

    :param task_id:
        the task identifier

    :param run_id:
        the string identifying the execution run

    :param label_columns:
        A tuple of database columns that will be taken as labels to train The Cannon model.
        These should be database columns (e.g., `astra.database.catalogdb.SDSSDR16ApogeeStar.teff`)
        and can be re-labelled using the `.label()` attribute:

        `astra.database.catalogdb.SDSSDR16ApogeeStar.x_h[1].label("c_h")`

    :param filter_args: [optional]
        arguments to supply to the `session.query(...).filter_by()` function to limit the
        number of sources in the training set

    :param filter_func: [optional]
        A callable function that returns True if a row should be kept in the training set,
        or False if it should be removed. This can be used for more complex queries than
        what might be possible using just `filter_args`.

    :param limit: [optional]
        Optionally set a limit for the number of rows returned.

    :param parameters: [optional]
        additional parameters to be assigned to the task instances

    :returns:
        The primary keys of the created tasks.
    """
    parameters = parameters or dict()

    data_model_identifiers = select_training_set_data_model_identifiers_from_database(
        label_columns,
        filter_args=filter_args,
        filter_func=filter_func,
        limit=limit,
        **kwargs,
    )

    pks = []
    for kwds in tqdm(data_model_identifiers):
        instance = get_or_create_task_instance(
            dag_id, task_id, run_id, parameters={**parameters, **kwds}
        )
        pks.append(instance.pk)

    return pks


def _select_training_set_data_from_database(
    label_columns, filter_args=None, filter_func=None, limit=None, **kwargs
):
    label_columns = list(label_columns)
    label_names = [column.key for column in label_columns]
    L = len(label_names)

    if filter_func is None:
        filter_func = lambda *_, **__: True

    # Get the label names.
    log.info(f"Querying for label names {label_names} from {label_columns}")

    # Figure out what other columns we will need to identify the input file.
    for column in label_columns:
        try:
            primary_parent = column.class_
        except AttributeError:
            continue
        else:
            break
    else:
        raise ValueError("Can't get primary parent. are you labelling every column?")

    log.debug(f"Identified primary parent table as {primary_parent}")

    if primary_parent == catalogdb.SDSSApogeeAllStarMergeR13:

        log.debug(f"Adding columns and setting data_model_func for {primary_parent}")
        additional_columns = [
            catalogdb.SDSSDR16ApogeeStar.apstar_version.label("apstar"),
            catalogdb.SDSSDR16ApogeeStar.field,
            catalogdb.SDSSDR16ApogeeStar.apogee_id.label("obj"),
            catalogdb.SDSSDR16ApogeeStar.file,
            catalogdb.SDSSDR16ApogeeStar.telescope,
            # Things that we might want for filtering on.
            catalogdb.SDSSDR16ApogeeStar.snr,
        ]

        columns = label_columns + additional_columns

        q = session.query(*columns).join(
            catalogdb.SDSSApogeeAllStarMergeR13,
            func.trim(catalogdb.SDSSApogeeAllStarMergeR13.apstar_ids)
            == catalogdb.SDSSDR16ApogeeStar.apstar_id,
        )

        data_model_func = lambda apstar, field, obj, filename, telescope, *_,: {
            "release": "DR16",
            "filetype": "apStar",
            "apstar": apstar,
            "field": field,
            "obj": obj,
            "prefix": filename[:2],
            "telescope": telescope,
            "apred": filename.split("-")[1],
        }

    else:
        raise NotImplementedError(
            f"Cannot intelligently figure out what data model keywords will be necessary."
        )

    if filter_args is not None:
        q = q.filter(*filter_args)

    if limit is not None:
        q = q.limit(limit)

    log.debug(f"Querying {q}")

    data_model_identifiers = []
    labels = {label_name: [] for label_name in label_names}
    for i, row in enumerate(tqdm(q.yield_per(1), total=q.count())):
        if not filter_func(*row):
            continue

        for label_name, value in zip(label_names, row[:L]):
            if not np.isfinite(value) or value is None:
                log.warning(f"Label {label_name} in {i} row is not finite: {value}!")
            labels[label_name].append(value)
        data_model_identifiers.append(data_model_func(*row[L:]))

    return (labels, data_model_identifiers)


def select_training_set_from_table(path, labels, limits, filter_function):

    # needs to return array of labels, and filetype identifiers (or optionally return paths)

    raise a


def train_polynomial_model(labels, data, order=2, regularization=0, threads=1):

    log.debug(f"Inputs are: ({type(labels)}) {labels}")
    log.debug(f"Data are: {data}")
    # labels could be in JSON format.
    if isinstance(labels, str):
        labels = json.loads(labels.replace("'", '"'))
        # TODO: use a general deserializer that fixes the single quote issues with json loading

    if isinstance(data, str) and os.path.exists(data):
        with open(data, "rb") as fp:
            data = pickle.load(fp)

    for key in ("dispersion", "wavelength"):
        try:
            dispersion = data[key]
        except KeyError:
            continue
        else:
            break
    else:
        raise ValueError(f"unable to find {key} in data")

    training_set_flux = data["normalized_flux"]
    training_set_ivar = data["normalized_ivar"]

    try:
        num_spectra = data["num_spectra"]
    except:
        log.debug(
            f"Keeping all items in training set; not checking for missing spectra."
        )
    else:
        keep = num_spectra == 1
        if not all(keep):
            log.warning(
                f"Excluding {sum(~keep)} objects from the training set that had missing spectra"
            )

            labels = {k: np.array(v)[keep] for k, v in labels.items()}
            training_set_flux = training_set_flux[keep]
            training_set_ivar = training_set_ivar[keep]

    # Set the vectorizer.
    vectorizer = tc.vectorizer.PolynomialVectorizer(
        labels.keys(),
        order=order,
    )

    # Initiate model.
    model = tc.model.CannonModel(
        labels,
        training_set_flux,
        training_set_ivar,
        vectorizer=vectorizer,
        dispersion=dispersion,
        regularization=regularization,
    )

    model.train(threads=threads)

    output_path = os.path.join(get_base_output_path(), "thecannon", "model.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log.info(f"Writing The Cannon model {model} to disk {output_path}")
    model.write(output_path, include_training_set_spectra=True, overwrite=True)
    return output_path


def plot_coefficients(model_path):

    from astra.contrib.thecannon import plot

    model = tc.CannonModel.read(model_path)

    output_dir = os.path.dirname(model_path)

    # Plot zeroth and first order coefficients.
    fig = plot.theta(
        model, indices=np.arange(1 + len(model.vectorizer.label_names)), normalize=False
    )
    fig.savefig(os.path.join(output_dir, "theta.png"), dpi=300)

    # Plot scatter.
    fig = plot.scatter(model)
    fig.savefig(os.path.join(output_dir, "scatter.png"), dpi=300)


def plot_one_to_one(model_path):
    from astra.contrib.thecannon import plot

    model = tc.CannonModel.read(model_path)
    output_dir = os.path.dirname(model_path)

    # Plot one-to-one.
    test_labels, test_cov, test_meta = model.test(
        model.training_set_flux,
        model.training_set_ivar,
        initial_labels=model.training_set_labels,
    )
    fig = plot.one_to_one(model, test_labels, cov=test_cov)
    fig.savefig(os.path.join(output_dir, "one-to-one.png"), dpi=300)
