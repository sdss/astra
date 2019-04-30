import click
from astra.utils import log
from astra.core import component
from astra.db.connection import session
from astra.db.models.component import Component



def _get_assumed_version(product, owner="sdss"):
    # Get default version.
    result = session.query(Component).filter_by(product=product, owner="sdss").one_or_none()
    if result is not None:
        return result.version

    else:
        return None

@click.group()
@click.pass_context
def parser(context):
    r"""Add, update, and delete components."""
    log.debug("component")
    pass


@parser.command()
@click.argument("product", nargs=1, required=True)
@click.option("--version", nargs=1, default=None,
              help="The version of this product to use. If no version is given then this will "
                   "default to the last release made available on GitHub.")
@click.option("--owner", nargs=1, default="sdss",
              help="The owner of the repository on GitHub (default: sdss).")
@click.option("--execution-order", "execution_order", default=0,
              help="Set the execution order for the component (default: 0).")
@click.option("--command", nargs=1, default=None,
              help="Specify the name of the command line utility to execute from that component. "
                   "This is only required if there are more than one executable components in the "
                   "bin/ directory of that repository.")
@click.option("--description", "description", nargs=1, default=None,
              help="A short description for this component. If no description is given then this "
                   "will default to the description that exists on GitHub.")
@click.option("-a", "--alt-module",  nargs=1, default=None,
              help="Specify an alternate module name for this component.")
@click.option("--default-args", nargs=1, default=None,
              help="Default arguments to supply to the command.")
@click.option("-t", "--test", is_flag=True, default=False,
              help="Test mode. Do not actually install anything.")
@click.pass_context
def add(context, product, version, owner, execution_order, command, description,
        alt_module, default_args, test):
    r"""
    Add a new component in Astra from an existing GitHub repository (`product`) and a 
    command line tool in that repository (`command`).
    """
    log.debug("component.add")

    return component.add(product=product,
                         version=version,
                         owner=owner,
                         execution_order=execution_order,
                         command=command,
                         description=description,
                         alt_module=alt_module,
                         default_args=default_args,
                         test=test)


# Update
@parser.command()
@click.argument("product", nargs=1, required=True)
@click.option("--version", nargs=1, default=None,
              help="The version of the product to update. If `None` is given then it will default "\
                   "to the most recent version.")
@click.option("--owner", nargs=1, default="sdss",
              help="The owner of the repository on GitHub (default: sdss).")
@click.option("--default-args", nargs=1, default=None,
              help="Default arguments to supply to the command line utility.")
@click.option("--active/--inactive", "is_active", default=None,
              help="Set the component as active or inactive.")
@click.option("--enable-auto-update/--disable-auto-update", "auto_update", default=None,
              help="Enable or disable automatic checks to GitHub for new releases.")
@click.option("--description", nargs=1,
              help="Set the short descriptive name for this component.")
@click.option("--execution-order", "execution_order", type=int,
              help="Set the execution order for this component.")
@click.option("--command", nargs=1,
              help="Set the command line interface tool for this component.")
@click.pass_context
def update(context, product, version, owner, default_args, is_active, auto_update,
           description, execution_order, command):
    r"""
    Update attribute(s) of an existing component, where the component is uniquely
    specified by the ``GITHUB_REPO_SLUG`` and the ``RELEASE`` version.
    """
    log.debug("component.update")


    if version is None:
        version = _get_assumed_version(product, owner)
        if version is None:
            click.UsageError("unknown version")

    # Only send non-None inputs.
    kwds = dict(is_active=is_active, auto_update=auto_update,
                description=description, execution_order=execution_order,
                command=command)
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

    return components.update(product=product, version=version, owner=owner, **kwds)


@parser.command()
@click.argument("product", nargs=1, required=True)
@click.option("--version", nargs=1, default=None,
              help="The version of the product to update. If `None` is given then it will default "\
                   "to the most recent version.")
@click.option("--owner", nargs=1, default="sdss",
              help="The owner of the repository on GitHub (default: sdss).")
@click.pass_context
def delete(context, product, version, owner):
    r"""
    Delete an existing component, where the component is uniquely specified by
    the ``GITHUB_REPO_SLUG`` and the ``RELEASE`` version.
    """
    log.debug("component.delete")

    if version is None:
        version = _get_assumed_version(product, owner)
        if version is None:
            click.UsageError("unknown version")

    return component.delete(product=product, version=version, owner=owner)


