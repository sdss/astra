import click
import os
import yaml
from astra import (components, log)

@click.group()
@click.pass_context
def component(context):
    r"""Create, update, and delete components"""
    log.debug("component")
    pass

@component.command()
@click.argument("from_path", nargs=1, required=True)
@click.pass_context
def create(context, from_path):
    r"""Create a component"""
    log.debug("component.create")

    if not os.path.exists(from_path):
        raise IOError(f"given path ({from_path}) does not exist")

    with open(from_path, "r") as fp:
        kwds = yaml.load(fp)

    log.info(f"Creating component from keywords: {kwds}")

    return components.create(**kwds)


@component.command()
@click.argument("github_repo_slug", nargs=1, required=True)
@click.argument("release", nargs=1, required=True)
@click.option("--active/--inactive", "is_active", default=None,
              help="Set a component as active or inactive.")
@click.option("--enable-auto-update/--disable-auto-update", "auto_update", default=None,
              help="Enable or disable automatic checks to GitHub for new releases.")
@click.option("--short-name", "short_name", nargs=1,
              help="Set the short descriptive name for a component.")
@click.option("--execution-order", "execution_order", type=int,
              help="Set the execution order for a component.")
@click.option("--component-cli", "component_cli", nargs=1,
              help="Set the command line interface tool for a component.")
@click.pass_context
def update(context, github_repo_slug, release, is_active, auto_update,
           short_name, execution_order, component_cli):
    r"""Update attributes of an existing component."""

    log.debug("component.update")

    # Only send non-None inputs.
    kwds = dict(is_active=is_active, auto_update=auto_update,
                short_name=short_name, execution_order=execution_order,
                component_cli=component_cli)
    for k in list(kwds.keys()):
        if kwds[k] is None:
            del kwds[k]

    # TODO: Consider a custom class to check that at least one option 
    #       is required. See: https://stackoverflow.com/questions/44247099/click-command-line-interfaces-make-options-required-if-other-optional-option-is
    if not kwds:
        raise click.UsageError("At least one option is required. "\
                               "Use 'component refresh [GITHUB_REPO_SLUG]' "\
                               "command to check GitHub for new releases. "\
                               "Use 'component update --help' to see the "\
                               "available options.")

    return components.update(github_repo_slug, release, **kwds)


@component.command()
@click.argument("github_repo_slug", nargs=1, required=True)
@click.argument("release", nargs=1, required=True)
@click.pass_context
def delete(context, github_repo_slug, release):
    r"""Delete an existing component."""
    log.debug("component.delete")

    return components.delete(github_repo_slug, release)


@component.command()
@click.pass_context
def refresh(context):
    r"""Check a component for an updated version"""
    log.debug("component.refresh")
    pass
