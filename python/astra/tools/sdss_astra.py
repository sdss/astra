from logging import DEBUG, INFO

import click
from astra import config, log
from astra.db.connection import Base, engine
from sqlalchemy_utils import database_exists, create_database


# Common options.
@click.group()
@click.option("-v", "verbose", default=False, is_flag=True,
              help="verbose mode")
@click.pass_context
def cli(context, verbose):
    context.ensure_object(dict)
    context.obj["verbose"] = verbose

    # TODO: This isn't correctly followed for sqlalchemy output. 
    # It defaults to verbose!
    log.set_level(DEBUG if verbose else INFO)


@cli.command()
@click.pass_context
def setup(context):
    r"""
    Setup Astra databases based on the current configuration.

    :param context:
        The command line interface context object.
    """

    log.debug("Running setup")

    if not database_exists(engine.url):
        log.info(f"Creating database {engine.url}")
        create_database(engine.url)

    elif click.confirm("Database already exists. This will wipe the database "\
                       "and start again. Are you sure?", abort=True):
        None

    log.debug("Dropping all tables")
    Base.metadata.drop_all(engine)

    log.debug("Creating all tables")
    Base.metadata.create_all(engine)

    log.info("Astra is ready.")
    return None



if __name__ == "__main__":
    cli(obj=dict())