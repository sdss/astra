from __future__ import absolute_import, division, print_function, unicode_literals

import click
import os
from shutil import rmtree
from astra.utils import log
from astra.db.connection import Base, engine
from sqlalchemy_utils import database_exists, create_database

@click.command()
@click.option("-y", "confirm", default=False, is_flag=True,
              help="Do not prompt the user for confirmation if the database already exists.")
@click.pass_context
def parser(context, confirm):
    r"""
    Setup Astra using the current configuration.
    """

    log.debug("Running setup")

    if not database_exists(engine.url):
        log.info(f"Creating database {engine.url}")
        create_database(engine.url)

    elif not confirm \
         and click.confirm("Database already exists. This will wipe the database, including all "\
                           "downloaded components, and start again. Are you sure?", abort=True):
        None

    # TODO: Remove the folder where old components are stored.
    """
    component_dir = os.getenv("ASTRA_COMPONENT_DIR", None)
    if component_dir is not None:
        if os.path.exists(component_dir):
            log.debug(f"Removing existing component directory {component_dir}")
            rmtree(component_dir)
        log.debug(f"Creating component directory {component_dir}")
        os.makedirs(component_dir, exist_ok=True)
    """

    log.debug("Dropping all tables")
    Base.metadata.drop_all(engine)

    log.debug("Creating all tables")
    Base.metadata.create_all(engine)

    log.debug("Removing old components")

    log.info("Per aspera ad astra; Astra is ready")
    return None