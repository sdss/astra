import click

from astra import log

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
    log.debug("subset.create")
    raise NotImplementedError()


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
    raise NotImplementedError()
