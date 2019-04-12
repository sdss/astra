import click

from astra import log

@click.command()
@click.pass_context
def schedule(context):
    r"""Schedule a component to be run on some data"""
    log.debug("schedule")
    pass
