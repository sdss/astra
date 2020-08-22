import click

from astra.utils import log
from astra.core import task
from astra.core.subset import _get_likely_subset
from astra.core.component import _get_likely_component
from astra.db.connection import session
from astra.db.models import (Component, Task, DataProductSubsetBridge)


@click.group()
@click.pass_context
def parser(context):
    r"""Create, update, and delete tasks."""
    log.debug("task")
    pass


@parser.command()
@click.argument("component", nargs=1, required=True)
@click.argument("subset", nargs=1, required=True)
@click.option("--output-dir", nargs=1, default=None,
              help="Specify an output directory for this task.")
@click.option("--schedule", nargs=1, default=None,
              help="Schedule this task for a time in the future.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def create(context, component, subset, schedule, output_dir, args):
    """
    Create a task.
    """
    log.debug("task.create")

    # Get the component from the given identifier:
    component = _get_likely_component(component)

    # Get the subset from the given identifier.
    subset = _get_likely_subset(subset)

    if schedule is not None:
        raise NotImplementedError("scheduling tasks is not possible yet")

    return task.create(component, subset, args=args, output_dir=output_dir)


@parser.command()
@click.argument("task_id", nargs=1, required=True)
@click.pass_context
def delete(context, task_id):
    r"""
    Delete a task.
    """
    log.debug("task.delete")

    return task.delete(task_id)


@parser.command()
@click.pass_context
def list(context):
    r""" List all the current tasks. """

    # TODO: List recent 100 or something
    log.info("Tasks:")
    tasks = session.query(Task).all()
    for task in tasks:
        component = session.query(Component).filter_by(id=task.component_id)
        N = session.query(DataProductSubsetBridge).filter_by(data_subset_id=task.subset_id).count()

        log.info(f"\t{task.id}: run {component.owner}/{component.product} ({component.version}) on "
                 f"subset {task.subset_id} ({N} entries)")

    return None



@parser.command()
@click.argument("task_id", nargs=1, required=True)
@click.option("--status", nargs=1)
@click.pass_context
def update(context, task_id, status):
    r"""
    Update attributes of a task.
    """
    log.debug("task.update")

    kwds = dict(status=status)
    for k in kwds.keys():
        if kwds[k] is None: 
            del kwds[k]

    if not kwds:
        raise click.UsageError("Nothing to update.")

    return task.update(task_id, **kwds)


