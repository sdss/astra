import click

from astra import (folders, log)
from astra.db.models import WatchedFolder


@click.group()
@click.pass_context
def data(context):
    r""" Manage data monitoring """
    log.debug("data")
    pass

@data.command()
@click.argument("path", nargs=1, required=True)
@click.option("-r", "--recursive", is_flag=True, default=False,
              help="monitor recursive directories")
@click.option("--interval", default=3600,
              help="number of seconds to wait between checking for new data")
@click.option("--regex-ignore-pattern", default=None,
              help="regular expression pattern for ignoring files")
@click.pass_context
def watch(context, path, recursive, interval, regex_ignore_pattern):
    r"""
    Start monitoring a folder for new data products.

    TODO: Docs on params.
    """
    log.debug(f"data.watch {path} {recursive} {interval} {regex_ignore_pattern}")
    result = folders.watch(path, recursive, interval, regex_ignore_pattern)
    log.info(result)
    return True


@data.command()
@click.argument("path", nargs=1, required=True)
@click.option("--quiet", is_flag=True, default=False,
              help="stay quiet if the given path is not being watched")
@click.pass_context
def unwatch(context, path, quiet):
    r"""
    Stop monitoring a folder for new data products.

    TODO: Docs on params/
    """
    log.debug(f"data.unwatch {path} {quiet}")
    result = folders.unwatch(path, quiet)
    log.info(result)
    return result


@data.command()
@click.argument("path", nargs=-1)
@click.option("--quiet", is_flag=True, default=False,
              help="stay quiet if the given path is not being watched")
@click.pass_context
def refresh(context, path, quiet):
    r"""
    Check a watched folder for new data products.
    """
    log.debug(f"data.refresh {path}")

    if not len(path):
        # Refresh all.
        counts = folders.refresh_all()

    else:
        # Refresh given paths.
        counts = dict([(p, folders.refresh(p, quiet)) for p in path])

    log.info(counts)
    return counts
