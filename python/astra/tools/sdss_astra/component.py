import click

from astra import log

@click.group()
@click.pass_context
def component(context):
    r"""Create, update, and delete components"""
    log.debug("component")
    pass

@component.command()
@click.pass_context
def create(context):
    r"""Create a component"""
    log.debug("component.create")
    pass

@component.command()
@click.pass_context
def update(context):
    r"""Update an existing component"""
    log.debug("component.update")
    pass

@component.command()
@click.pass_context
def delete(context):
    r"""Delete an existing component"""
    log.debug("component.delete")
    pass


@component.command()
@click.pass_context
def refresh(context):
    r"""Check a component for an updated version"""
    log.debug("component.refresh")
    pass
