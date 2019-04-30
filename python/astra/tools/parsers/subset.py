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
            raise click.UsageError(f"no subset found with name '{identifier}'")

    else:
        # Could be a name or an id.
        subset_by_name = session.query(DataSubset).filter_by(name=identifier).one_or_none()
        subset_by_id = session.query(DataSubset).filter_by(id=int(identifier)).one_or_none()

        if subset_by_id is not None and subset_by_name is not None:
            raise click.UsageError(f"ambiguous identifier: subset exists with name='{identifier}' "
                                   f"and with id={identifier}")

        elif subset_by_id is None and subset_by_name is None:
            raise click.UsageError(f"no subset found with name '{identifier}' or id {identifier}")

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
    r"""Create, update, and delete data subsets."""
    log.debug("subset")
    pass

# Thanks https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
class OptionEatAll(click.Option):

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


@parser.command()
@click.option("--data-paths", default=None, cls=OptionEatAll,
              help="Supply data paths that will form this subset.")
@click.option("--regex-match-pattern", nargs=1, default=None,
              help="Supply a regular expression pattern to match against all data paths.")
@click.option("--name", nargs=1, default=None,
              help="Provide a name for this subset.")
@click.option("--visible", is_flag=True, default=None,
              help="Make this subset visible when users search for subsets.")
@click.option("--auto-update", is_flag=True, default=False,
              help=("Automatically update this subset when new data are available. This is only "
                    "relevant for subsets that have a regular expression pattern to match against."))
@click.option("--raise-on-unrecognised", "raise_on_unrecognised", is_flag=True, default=False,
              help="Raise an exception if a data path is given that is not recognised in the "
                   "database. If this flag is not given then the unrecognised data paths will be "
                   "added to the database.")
@click.pass_context
def create(context, data_paths, regex_match_pattern, name, visible, auto_update,
           raise_on_unrecognised):
    r"""
    Create a subset from the available data products. A subset can be created from data paths, 
    and/or by providing a regular expression pattern to match against data paths.
    """

    # If there are data paths then create it from that.
    if (data_paths is None or len(data_paths) < 1) and regex_match_pattern is None:
        raise click.UsageError("Either data paths or a regular expression pattern is required.")

    if name is not None and visible is None:
        visible = True

    result = subset.create_from_data_paths(data_paths, name=name, is_visible=visible,
                                           add_unrecognised=not raise_on_unrecognised)

    if regex_match_pattern is not None:
        result = subset.update(result,
                               regex_match_pattern=regex_match_pattern, 
                               auto_update=auto_update)

    return result

@parser.command()
@click.pass_context
def list(context):
    r"""List the named subsets available."""
    log.debug("subset.list")
    named_str = "\n\t".join(subset.list_named_and_visible())
    log.info(f"Named subsets available:\n\t{named_str}")
    return named_str



@parser.command()
@click.argument("identifier", nargs=1, required=True, type=str)
@click.pass_context
def refresh(context, identifier):
    r"""Refresh an existing subset. Identifier can be the subset id or name"""
    log.debug("subset.refresh")
    return subset.refresh(_get_subset_by_identifier(identifier))



@parser.command()
@click.argument("identifier", nargs=1, required=True, type=str)
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
    kwds = dict(is_visible=is_visible, auto_update=auto_update,
                name=name, regex_match_pattern=regex_match_pattern)
    kwds = {k: v for k, v in kwds.items() if v is not None}
    log.info(f"Updating with keywords: {kwds}")

    # TODO: Consider a custom class to check that at least one option 
    #       is required. See: https://stackoverflow.com/questions/44247099/click-command-line-interfaces-make-options-required-if-other-optional-option-is
    if not kwds:
        raise click.UsageError("At least one option is required")

    return subset.update(_subset, **kwds)



@parser.command()
@click.argument("identifier", nargs=1, required=True, type=str)
@click.pass_context
def delete(context, identifier):
    r"""Delete a named subset"""
    log.debug("subset.delete")
    return subset.delete(_get_subset_by_identifier(identifier))
