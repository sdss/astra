import click

from astra import log
from astra.core import subset
from astra.db.connection import session
from astra.db.models import DataSubset

def _get_subset_by_identifier(identifier):

    try:
        int(identifier)

    except:
        # definitely a name
        result = session.query(DataSubset).filter_by(name=identifier).one_or_none()
        if result is None:
            click.UsageError(f"no subset found with name '{identifier}'")

    else:
        # Could be a name or an id.
        subset_by_name = session.query(DataSubset).filter_by(name=identifier).one_or_none()
        subset_by_id = session.query(DataSubset).filter_by(id=int(identifier)).one_or_none()

        if subset_by_id is not None and subset_by_name is not None:
            click.UsageError(f"ambiguous identifier: subset exists with name='{identifier}' and "
                             f"with id={identifier}")

        elif subset_by_id is None and subset_by_name is None:
            click.UsageError(f"no subset found with name '{identifier}' or id {identifier}")

        elif subset_by_id is None and subset_by_name is not None:
            log.info(f"Identified subset by name '{identifier}': {subset_by_name}")
            result = subset_by_name

        elif subset_by_id is not None and subset_by_name is None:
            log.info(f"Identified subset by id {identifier}: {subset_by_id}")
            result = subset_by_id

    return result



@click.group()
@click.pass_context
def parser(context):
    r"""Create, update, and delete data subsets"""
    log.debug("subset")
    pass


@parser.command()
@click.pass_context
def create(context):
    r"""Create a subset of the data"""
    # (1) From list of paths.
    # (2) From regular expression..
    log.debug("subset.create")
    raise NotImplementedError()


@parser.command()
@click.argument("identifier", nargs=1, required=True, dtype=str)
@click.pass_context
def refresh(context, identifier):
    r"""Refresh an existing subset. Identifier can be the subset id or name"""
    log.debug("subset.refresh")
    return subset.refresh(_get_subset_by_identifier(identifier))



@parser.command()
@click.pass_context
def update(context):
    r"""Update an existing named subset"""
    log.debug("subset.update")
    raise NotImplementedError()


@parser.command()
@click.pass_context
def delete(context):
    r"""Delete a named subset"""
    log.debug("subset.delete")
    return subset.delete(_get_subset_by_identifier(identifier))
