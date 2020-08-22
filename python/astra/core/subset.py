import datetime
import os
import re
from astra.utils import log
from astra.db.connection import session
from astra.db.models import (DataSubset, DataProduct, DataProductSubsetBridge)


def _get_likely_subset(identifier):
    """
    Return the most likely subset, given some identifier about that subset. The identifier could be
    the subset `id`, or the name of the `subset`.
    """

    if isinstance(identifier, DataSubset):
        # This is a subset.
        return identifier

    try:
        identifier = int(identifier)

    except:
        # The identifier is a name.
        result = session.query(DataSubset).filter_by(name=identifier).one_or_none()
        if result is None:
            raise ValueError(f"no subset matching name '{identifier}' found")

        return result

    else:
        # The identifier is an integer. It could be the id, or a dataset with the same name.
        result_by_name = session.query(DataSubset).filter_by(name=f"{identifier}").one_or_none()
        result_by_id = session.query(DataSubset).filter_by(id=identifier).one_or_none()

        if result_by_id is not None and result_by_name is None:
            return result_by_id

        elif result_by_id is None and result_by_name is not None:
            return result_by_name

        elif result_by_id is None and result_by_name is None:
            raise ValueError(f"no subset found with id = {identifier} or name = {identifier}")

        else:
            raise ValueError(f"ambigious subset identifier: found subset with id = {identifier} "
                             f"and name = {identifier}")



def create_from_data_paths(data_paths, name=None, is_visible=False, add_unrecognised=True):
    r"""
    Create a data subset from a list of paths.
    
    :param paths:
        A list of local paths.

    :param name: [optional]
        A name to give to this subset. If given, it must be unique from all other subset names.

    :param is_visible: [optional]
        Make this subset visible to all users.

    :param add_unrecognised: [optional]
        Add unrecognized data paths to the database (default: True). If this is not given then
        an exception will be raised.
    """

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    if data_paths is None or len(data_paths) < 1:
        raise ValueError("data paths is empty")

    if name is not None:
        name = str(name).strip()

        s = session.query(DataSubset).filter_by(name=name).one_or_none()
        if s is not None:
            raise ValueError(f"subset name '{name}' already exists: {s}")

    # For each data path, find the corresponding id.

    data_ids = []
    unrecognised_data_paths = []
    for data_path in data_paths:

        path = os.path.abspath(data_path)

        dp = session.query(DataProduct).filter_by(path=path).one_or_none()
        if dp is None:
            # TODO: this should be a warning
            if add_unrecognised:
                log.info(f"Adding unrecognized path {path} to database")
            else:
                log.error(f"Could not find path {path} in database: not including it in subset")
            unrecognised_data_paths.append(path)

        else:
            data_ids.append(dp.id)

    if unrecognised_data_paths:
        if not add_unrecognised:
            raise ValueError(f"did not recognise {len(unrecognised_data_paths)} paths in database")

        else:
            # Add the unrecognized data paths.
            for path in unrecognised_data_paths:
                session.add(DataProduct(path=path))

            session.commit()

            # To understand recursion you must understand recursion.
            return create_from_data_paths(data_paths, name=name, is_visible=is_visible,
                                          add_unrecognised=False)

    if not len(data_ids):
        raise ValueError(f"no data to add to subset")

    # Create the subset.
    subset = DataSubset(name=name, is_visible=is_visible)
    session.add(subset)
    session.commit()

    # Add the data items to the bridge.
    data_ids = set(data_ids)
    log.info(f"Adding {len(data_ids)} to subset {subset}")
    for data_id in set(data_ids):
        session.add(DataProductSubsetBridge(data_subset_id=subset.id, data_product_id=data_id))

    session.commit()

    return subset


def create_from_regular_expression(pattern, name=None, is_visible=False, auto_update=True):
    r"""
    Create a data subset from a regular expression pattern.

    :param pattern:
        A regular expression pattern to use on data paths. Data that match this path will be
        added to the subset.

    :param name: [optional]
        A name to give to this subset. If given, it must be unique from all other subset names.

    :param is_visible: [optional]
        Make this subset visible to all users.

    :param auto_update: [optional]
        Periodically update the subset when new data is available, based on the regular expression
        pattern.
    """
    if name is not None:
        name = str(name).strip()
        s = session.query(DataSubset).filter_by(name=name).one_or_none()
        if s is not None:
            raise ValueError(f"subset name '{name}' already exists: {s}")

    subset = DataSubset(name=name, regex_match_pattern=pattern, is_visible=is_visible,
                        auto_update=auto_update)
    session.add(subset)
    session.commit()

    log.info(f"Created data subset {subset} from regular expression pattern")

    # Check for data paths.
    added, _ = refresh(subset_id)

    return (subset, added)


def refresh(subset_id):
    r"""
    Check for new data that match the regular expression pattern given for a subset.

    :param subset_id:
        The unique identifier of the subset to refresh.

    :returns:
        A two-length tuple containing the number of paths added, and the number of paths skipped 
        (e.g., paths that were already part of this subset.)
    """
    subset = _get_likely_subset(subset_id)

    if subset.regex_match_pattern is None:
        log.warn(f"Nothing to refresh on data subset with id {subset_id}: no regex match pattern.")    
        return False

    added, skipped = (0, 0)
    for data_product in session.query(DataProduct).all():
        if re.search(subset.regex_match_pattern, data_product.path):

            # Check that we don't have a match already.
            kwds = dict(data_subset_id=subset.id, data_product_id=data_product.id)
            if session.query(DataProductSubsetBridge).filter_by(**kwds).one_or_none() is None:
                log.info(f"Adding {data_product} to subset {subset} based on regex match")
                session.add(DataProductSubsetBridge(**kwds))
                added += 1

            else:
                # Already in this subset.
                skipped += 1

    log.info(f"Added {added} paths to subset {subset}; skipped {skipped} existing paths")

    session.commit()

    return (added, skipped)



def update(subset_id, **kwargs):
    r"""
    Update attributes of an existing data subset.

    :param subset_id:
        The unique identifier of the subset to update.

    :param name: [optional]
        A name to give to the subset. If given, this must be unique from all others.

    :param is_visible: [optional]
        A boolean flag indicating whether to make the subset visible.

    :param regex_match_pattern: [optional]
        Provide a regular expression pattern to use to match against new data paths.

    :param auto_update: [optional]
        Automatically update this subset when new data become available. This is only useful for
        subsets that are defined by a regular expression pattern.
    """

    subset = _get_likely_subset(subset_id)

    available = ("name", "regex_match_pattern", "is_visible", "auto_update")
    unknown = set(available).difference(kwargs.keys())
    if unknown:
        log.error(f"Ignoring subset keywords: {unknown}")


    kwds = dict([(k, kwargs[k]) for k in available if kwargs.get(k, None) is not None])
    for k in kwds.keys():
        if k in ("is_visible", "auto_update"):
            kwds[k] = bool(kwds[k])

    # Check name.
    if "name" in kwds:
        if kwds["name"] == subset.name:
            log.info("Ignoring subset name change because name is same")
            del kwds["name"]

        else:
            existing_subset = session.query(DataSubset).filter_by(name=kwds["name"]).one_or_none()
            if existing_subset is not None:
                raise ValueError(f"subset already exists by name {kwds['name']}")

    if kwds:
        log.info(f"Updating subset {subset} with {kwds}")
        for k, v in kwds.items():
            setattr(subset, k, v)

        subset.modified = datetime.datetime.utcnow()
        session.commit()

    return subset

def list_named_and_visible():
    r"""
    Return a list of the named (and visible) subsets availalble.
    """

    named_subsets = session.query(DataSubset).filter_by(is_visible=True).all()
    # TODO: filter by not None?
    named_subsets = [ns.name for ns in named_subsets if ns.name is not None]
    return named_subsets




def delete(subset_id):
    r"""
    Delete an existing data subset.

    :param subset_id:
        The unique identifier of the subset to delete.
    """
    subset = _get_likely_subset(subset_id)

    # Delete any references in the bridge.
    references = session.query(DataProductSubsetBridge).filter_by(data_subset_id=subset.id).all()
    N = len(references)
    for reference in references:
        session.delete(reference)

    # Commit so we don't violate foreign key constraints.
    session.commit()

    # Delete the subset.
    session.delete(subset)

    session.commit()
    log.info(f"Deleted subset {subset} with id {subset_id} ({N} entries)")

    return True
