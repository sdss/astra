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
@click.option("--data-paths", nargs="?", default=None,
              help="Supply data paths that will form this subset.")
@click.option("--regex-match-pattern", nargs=1, default=None,
              help="Supply a regular expression pattern to match against all data paths.")
@click.option("--name", nargs=1, default=None,
              help="Provide a name for this subset.")
@click.option("--visible", is_flag=True, default=False,
              help="Make this subset visible when users search for subsets.")
@click.option("--auto-update", is_flag=True, default=False,
              help="Automatically update this subset when new data are available. This is only "
                   "relevant for subsets that have a regular expression pattern to match against.")
@click.pass_context
def create(context, data_paths, regex_match_pattern, name, visible, auto_update):
    r"""
    Create a subset from the available data products. A subset can be created from data paths, 
    and/or by providing a regular expression pattern to match against data paths.
    """

    # If there are data paths then create it from that.
    if data_paths is None and regex_match_pattern is None:
        click.UsageError("Either data paths or a regular expression pattern is required.")

    result = subset.create_from_data_paths(data_paths, name=name, is_visible=visible)

    if regex_match_pattern is not None:
        result = subset.update(result,
                               regex_match_pattern=regex_match_pattern, 
                               auto_update=auto_update)

    return result
    

@parser.command()
@click.argument("identifier", nargs=1, required=True, dtype=str)
@click.pass_context
def refresh(context, identifier):
    r"""Refresh an existing subset. Identifier can be the subset id or name"""
    log.debug("subset.refresh")
    return subset.refresh(_get_subset_by_identifier(identifier))



@parser.command()
@click.argument("identifier", nargs=1, required=True, dtype=str)
@click.option("--visible/--invisible", "is_visible", default=None,
              help="Set the subset as visible or invisible.")
@click.option("--auto-update/--no-auto-update", "auto_update", default=None,
              help="Automatically update the subset as new data become available. This is only "
                   "relevant for subsets that have a regular expression pattern to match data paths")
@click.option("--regex-match-pattern", nargs=1, default=None,
              help="Supply a regular expression pattern to match against all data paths.")
@click.option("--name", nargs=1, default=None,
              help="Provide a name for the specified subset.")
@click.pass_context
def update(context, identifier, is_visible, auto_update, regex_match_pattern, name):
    r"""
    Update attributes of an existing named subset.
    """
    _subset = _get_subset_by_identifier(identifier)

    # Only send non-None inputs.
    kwds = dict(is_active=is_active, auto_update=auto_update,
                name=name, regex_match_pattern=regex_match_pattern)
    for k in list(kwds.keys()):
        if kwds[k] is None:
            del kwds[k]

    # TODO: Consider a custom class to check that at least one option 
    #       is required. See: https://stackoverflow.com/questions/44247099/click-command-line-interfaces-make-options-required-if-other-optional-option-is
    if not kwds:
        raise click.UsageError("At least one option is required")

    return subset.update(_subset, **kwds)



@parser.command()
@click.pass_context
def delete(context):
    r"""Delete a named subset"""
    log.debug("subset.delete")
    return subset.delete(_get_subset_by_identifier(identifier))
