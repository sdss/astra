from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime
from astra import log
from astra.db.connection import session
from astra.db.models import (Component, DataSubset, Task)


def create(component, subset, args=None, schedule=None, output_dir=None):
    r"""
    Create a task for a component to execute on some data subset.

    :param component:
        The id of the component to execute.

    :param subset:
        The id of the data subset to execute.

    :param schedule: [optional]
        Specify a schedule datetime for this task to be executed.
    """

    if not isinstance(component, Component):
        # Verify that the component exists.
        result = session.query(Component).filter_by(id=component).one_or_none()
        if result is None:
            raise ValueError(f"unrecognized component id {component}")
        component = result

    if not isinstance(subset, DataSubset):
        # Verify that the data subset exists.
        result = session.query(DataSubset).filter_by(id=subset).one_or_none()
        if result is None:
            raise ValueError(f"unrecognized subset id {subset}")
        subset = result

    if schedule is None:
        schedule = datetime.datetime.now()

    elif not isinstance(schedule, datetime.datetime):
        raise TypeError("if a schedule time is given then it must be a datetime.datetime object")

    if output_dir is None:
        # TODO: Assumee CWD? Something from environment variables?
        output_dir = os.getcwd()
        log.info(f"Assuming output directory as {output_dir}")

    # Create the task.
    task = Task(component_id=component.id, data_subset_id=subset.id,
                schedule=schedule, output_dir=output_dir, args=args)
    session.add(task)
    session.commit()

    if not os.path.exists(task.output_dir):
        log.info(f"Creating output directory {output_dir}")
        os.makedirs(task.output_dir, exist_ok=True)

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
    task.status = "DELETED"
    session.commit()

    return task


