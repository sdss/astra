from logging import DEBUG, INFO

import click
from astra import config, log
from astra.db.connection import Base, engine
from sqlalchemy_utils import database_exists, create_database

from astra.tools.sdss_astra.data import data
from astra.tools.sdss_astra.subset import subset
from astra.tools.sdss_astra.component import component
from astra.tools.sdss_astra.execute import execute
from astra.tools.sdss_astra.query_execute import query_execute
from astra.tools.sdss_astra.schedule import schedule


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
@click.option("-y", "confirm", default=False, is_flag=True,
              help="drop the database if it already exists")
@click.pass_context
def setup(context, confirm):
    r""" Setup databases using the current configuration. """

    log.debug("Running setup")

    if not database_exists(engine.url):
        log.info(f"Creating database {engine.url}")
        create_database(engine.url)

    elif not confirm \
         and click.confirm("Database already exists. "\
                           "This will wipe the database and start again. "\
                           "Are you sure?", abort=True):
        None

    log.debug("Dropping all tables")
    Base.metadata.drop_all(engine)

    log.debug("Creating all tables")
    Base.metadata.create_all(engine)

    log.info("Astra is ready.")
    return None


# Add various commands
cli.add_command(data)
cli.add_command(subset)
cli.add_command(component)
cli.add_command(execute)
cli.add_command(query_execute)
cli.add_command(schedule)




if __name__ == "__main__":
    cli(obj=dict())