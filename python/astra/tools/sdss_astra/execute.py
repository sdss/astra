from __future__ import absolute_import, division, print_function, unicode_literals

import click
from astra import (folders, log)
from astra.db.connection import session
from astra.db.models import Components, Task


@click.command()
@click.argument("github_repo_slug", nargs=1, required=True)
@click.argument("input_path")
@click.argument("output_dir")
@click.option("--release", nargs=1, default=None,
              help="Release version of the component to use. If none is given "\
                   "then it defaults to the most recent release.")
@click.pass_context
def execute(context, github_repo_slug, release, input_path, output_dir, **kwargs):
    r"""Execute a component on some reduced data products. """
    log.debug("execute")

    # Check release.
    query = session.query(Components).filter_by(github_repo_slug=github_repo_slug)
    if release is None:
        component = query.order_by(Components.created.desc()).first()

    else:
        component = query.filter_by(release=release).one_or_none()
        if component is None:
            raise ValueError(f"no component found with slug {github_repo_slug} and release {release}")

    log.info(f"Executing {component}")

    # Create a task, and then we will execute it immediately.
    
    # Need: environment variables, etc

    raise a