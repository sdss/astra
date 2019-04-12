import click

from astra import log

@click.command()
@click.pass_context
def query_execute(context):
    r"""Query whether a component can run on some data"""
    log.debug("query_execute")
    pass
