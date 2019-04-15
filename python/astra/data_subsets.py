

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from astra import log
from astra.db.connection import session
from astra.db.models import (DataSubset, DataProduct, DataProductSubsetBridge)


def create_subset_from_data_paths(data_paths, name=None, is_visible=False):
    r"""
    Create a data subset from a list of paths.
    
    :param paths:
        A list of local paths.

    :param name: [optional]
        A name to give to this subset. If given, it must be unique from all
        other subset names.

    :param is_visible: [optional]
        Make this subset visible to all users.
    """

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
            log.info(f"Could not find path {path} in database")
            unrecognised_data_paths.append(path)

        else:
            data_ids.append(dp.id)

    if unrecognised_data_paths:
        raise ValueError(f"did not recognise {len(unrecognised_data_paths)} paths in database")

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
        session.add(DataProductSubsetBridge(data_subset_id=subset.id, 
                                            data_product_id=data_id))

    session.commit()

    return subset


def create_subset_from_regular_expression_pattern(pattern, name=None, is_visible=False,
                                                  auto_update=True):

    raise NotImplementedError("soon")


def update_subset(subset_id, **kwargs):
    raise NotImplementedError("soon")


def delete_subset(subset_id):
    raise NotImplementedError("soon")
