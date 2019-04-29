from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime
from astra import log
from astra.db.connection import session
from astra.db.models import (Component, DataSubset, Task)


def create(component_id, data_subset_id, scheduled=None):
    r"""
    Create a task for a component to execute on some data subset.

    :param component_id:
        The id of the component to execute.

    :param data_subset_id:
        The id of the data subset to execute.

    :param scheduled: [optional]
        Specify a scheduled time for this task to be executed.
    """

    # Verify that the component exists.
    component = session.query(Component).filter_by(id=component_id).one_or_none()
    if component is None:
        raise ValueError(f"unrecognized component id {component_id}")

    # Verify that the data subset exists.
    subset = session.query(DataSubset).filter_by(id=data_subset_id).one_or_none()
    if subset is None:
        raise ValueError(f"unrecognized subset id {data_subset_id}")

    if scheduled is None:
        scheduled = datetime.datetime.now()

    elif not isinstance(scheduled, datetime.datetime):
        raise TypeError("if a scheduled time is given then it must be a datetime.datetime object")

    # Create the task.
    task = Task(component_id=component_id, data_subset_id=data_subset_id,
                scheduled=scheduled)
    session.add(task)
    session.commit()

    # Generate the output directory.
    task.output_dir = os.path.join("$ASTRA_TASK_DIR", f"{task.id:0>10.0f}")
    task.log_path = os.path.join(task.output_dir, f"{task.id:0>10.0f}.log")

    output_dir = task.output_dir.replace("$ASTRA_TASK_DIR",
                                         os.getenv("ASTRA_TASK_DIR", ""))

    log.debug(f"Creating output directory {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Commit the output_dir and log_path changes
    session.commit()

    return task


def update(task_id, **kwargs):
    r"""
    Update an attribute of an existing task.

    :param task_id:
        The id of the task to update.

    Optional keywords include:
    
    :param status: [optional]
        The status of the task.
    """

    acceptable_keywords = ("status", )
    relevant_keywords = set(acceptable_keywords).intersection(kwargs)
    kwds = dict([(k, kwargs[k]) for k in relevant_keywords])

    if kwds:
        log.info(f"Updating task {task_id} with {kwds}")

        task = session.query(Task).filter_by(id=task_id).one_or_none()
        if task is None:
            raise ValueError(f"no task found with id {task.id}")

        for k, v in kwds.items():
            setattr(task, k, v)

        task.modified = datetime.datetime.utcnow()
        session.commit()

        return task

    else:
        # TODO: this should be a warning
        log.info(f"Nothing to update on task with id {task_id}")

        return None



def delete(task_id):
    r"""
    Delete an existing task.

    :param task_id:
        The id of the task to delete.
    """
   
    task = session.query(Task).filter_by(id=task_id).one_or_none()
    if task is None:
        raise ValueError(f"no task found with id {task.id}")

    log.info(f"Deleting task {task}")

    # TODO: handle status' better
    task.status = "CANCELLED"
    session.commit()

    return task


