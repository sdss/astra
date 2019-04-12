import click

from astra import log

@click.command()
@click.pass_context
def execute(context):
    r"""Execute a component"""
    log.debug("execute")
    pass
