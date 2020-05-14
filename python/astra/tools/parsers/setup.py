from __future__ import absolute_import, division, print_function, unicode_literals

import click
import os
from shutil import rmtree
from astra.utils import log
from sqlalchemy_utils import database_exists, create_database
from astra.db.connection import Base, engine
from astra.utils.github import get_authentication_token

@click.command()
@click.option("-y", "confirm", default=False, is_flag=True,
              help="Do not prompt the user for confirmation if the database already exists.")
@click.pass_context
def parser(context, confirm):
    r"""
    Setup Astra using the current configuration.
    """

    log.debug(f"Running setup with {engine}")

    # TODO Should we do the imports here so that we can change the config if needed.
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

    # Check for environment variables / configurations.
    try:
        get_authentication_token()

    except ValueError:
        log.warning("No GitHub personal access token found in SDSS_GITHUB_KEY environment variable "
                    "or through Astra configuration setting 'github.token'. Without a GitHub "
                    "personal access token you will be unable to add remote components. "
                    "See https://sdss-astra.readthedocs.io/en/latest/installation.html for details.")
    
    log.info("Per aspera ad astra; Astra is ready")
    return None