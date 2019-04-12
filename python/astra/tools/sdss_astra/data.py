import click

from astra import log

@click.group()
@click.pass_context
def data(context):
    r""" Manage data monitoring """
    log.debug("data")
    pass

@data.command()
@click.pass_context
def watch(context):
    r""" Monitor a folder for new SDSS data products """
    log.debug("data.watch")
    pass


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

