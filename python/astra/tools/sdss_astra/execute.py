from __future__ import absolute_import, division, print_function, unicode_literals

import click
from astra import log
from astra.db.connection import session
from astra.db.models import Component
from astra import (data_subsets, tasks)


@click.command()
@click.argument("github_repo_slug", nargs=1, required=True)
@click.argument("input_path")
@click.argument("output_dir")
@click.option("--release", nargs=1, default=None,
              help="Release version of the component to use. If `None` is given "\
                   "then it defaults to the most recent release.")
@click.option("--from-path", "from_path", is_flag=True, default=False,
              help="Read a list of input files from the `INPUT_PATH` text file.")
@click.pass_context
def execute(context, github_repo_slug, input_path, output_dir, release, 
            from_path, **kwargs):
    r"""
    Execute a component on a reduced data product (``INPUT_PATH``) and write
    the outputs to ``OUTPUT_DIR``. The component is uniquely
    specified by the ``GITHUB_REPO_SLUG`` and the release version. The most
    recent release will be assumed if no release is specified in the ``--release``
    option.
    """
    log.debug("execute")

    # Check release.
    query = session.query(Component).filter_by(github_repo_slug=github_repo_slug)
    if release is None:
        # Get the version with the highest release number.
        component = query.order_by(Component.release.desc()).first()

    else:
        component = query.filter_by(release=release).one_or_none()
        if component is None:
            raise ValueError(f"no component found with slug {github_repo_slug} and release {release}")

    log.info(f"Executing {component}")

    if from_path:
        with open(input_path, "r") as fp:
            data_paths = [ea.strip() for ea in fp.readlines() if len(ea.strip())]
    else:
        data_paths = [input_path]

    # Create a task, and then we will execute it immediately.
    subset = data_subsets.create_subset_from_data_paths(data_paths)
    task = tasks.create(component.id, subset.id)

    # Need to module_load the right thing.


    # Actually run the damn thing.


    raise a