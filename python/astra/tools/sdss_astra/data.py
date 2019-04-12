import click

from astra import log
from astra.data import (watch_folder, )


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
    Monitor a folder for new data products.
    
    """
    log.debug("data.watch")

    result = watch_folder(path, recursive, interval, regex_ignore_pattern)

    log.info(result)
    


    raise a


@data.command()
@click.pass_context
def unwatch(context):
    r""" Stop monitoring a folder for new SDSS data products """
    log.debug("data.unwatch")

@data.command()
@click.pass_context
def refresh(context):
    r""" Check all watched folders for new data products"""
    log.debug("data.refresh")

