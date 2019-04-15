from __future__ import absolute_import, division, print_function, unicode_literals

import click
from astra import (folders, log)

@click.group()
@click.pass_context
def data(context):
    r""" Manage data monitoring """
    log.debug("data")
    pass

@data.command()
@click.argument("path", nargs=1, required=True)
@click.option("-r", "--recursive", is_flag=True, default=False,
              help="Monitor directory recursively (default: false).")
@click.option("--interval", default=3600,
              help="Number of seconds to wait between checking for new data (default: 3600).")
@click.option("--regex-ignore-pattern", default=None,
              help="Regular expression pattern for ignoring files.")
@click.pass_context
def watch(context, path, recursive, interval, regex_ignore_pattern):
    r"""
    Start monitoring a folder for new data products.
    """
    log.debug(f"data.watch {path} {recursive} {interval} {regex_ignore_pattern}")
    result = folders.watch(path, recursive, interval, regex_ignore_pattern)
    log.info(result)
    return True


@data.command()
@click.argument("path", nargs=1, required=True)
@click.option("--quiet", is_flag=True, default=False,
              help="Stay quiet if the given path is not being watched.")
@click.pass_context
def unwatch(context, path, quiet):
    r"""
    Stop monitoring a folder for new data products.
    """
    log.debug(f"data.unwatch {path} {quiet}")
    result = folders.unwatch(path, quiet)
    log.info(result)
    return result


@data.command()
@click.argument("path", nargs=-1)
@click.option("--quiet", is_flag=True, default=False,
              help="Stay quiet if the given path is not being watched.")
@click.pass_context
def refresh(context, path, quiet):
    r"""
    Check watched folder(s) for new data products.

    If no PATH is given then all watched folders will be refreshed.
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
