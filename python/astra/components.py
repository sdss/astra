from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime
import requests
import tempfile
from shutil import copyfileobj
from astra import log
from astra.db.connection import session
from astra.db.models.components import Component
from astra.utils import github

_valid_github_repo_slug = lambda _: _.strip().lower()

def create(github_repo_slug, component_cli, short_name=None, release=None,
           execution_order=0, owner=None, **kwargs):
    r"""
    Create a component for analysing reduced data products.

    :param github_repo_slug:
        The GitHub repository in 'slug' form: {OWNER}/{REPOSITORY_NAME}.

    :param component_cli:
        The command line utility that will be executed by this component.

    :param short_name: [None]
        A short, descriptive name for this component. If ``None`` is given
        then it will default to the short description that exists on GitHub.

    :param release: [optional]
        The version release to use. If `None` is given then the most recent
        release on GitHub will be used.

    :param execution_order: [optional]
        The execution order priority (ascending non-negative execution order).
        See the documentation online for more details about the execution order.

    :param owner: [optional]
        A string describing the owner (e.g., "Andy Casey <andrew.casey@monash.edu>").
        TODO: This may be deprecated in future in place of the GitHub information.
    """

    # If necessary, fetch the repository to get the most recent release.
    github_repo_slug = _valid_github_repo_slug(github_repo_slug)
    owner, repository_name = github_repo_slug.split("/")

    # Check that this repository exists.
    repository = github.get_repository_summary(owner, repository_name)
    if repository is None:
        raise ValueError(f"cannot find GitHub repository '{repository_name}' "\
                         f"with owner '{owner}'")

    if release is None:
        log.info(f"Querying GitHub for the last release on {github_repo_slug}")

        last_release = github.get_most_recent_release(owner, repository_name)
        log.info(f"GitHub response: {last_release}")

        if not last_release:
            raise ValueError(f"no releases available for {github_repo_slug}")

        release = last_release["name"]
        release_info = last_release

    else:
        log.info("Making sure this release exists on GitHub")
        releases = github.get_most_recent_release(owner, repository_name, n=100)
        for release_info in releases:
            if release_info["name"] == release: break

        else:
            raise ValueError(f"could not find release {release} in last 100 "\
                             "releases on GitHub")


    # Check for a component that matches this slug.
    item = _get_component_or_none(github_repo_slug, release)
    if item is not None:
        raise ValueError(f"component already exists "\
                         f"({github_repo_slug} release {release}) "\
                         f"as component id {item.id}")

    # Check if we need to give it a short name.
    if short_name is None:
        short_name = repository["description"]


    # Create the component.
    component = Component(github_repo_slug=github_repo_slug, release=release,
                          component_cli=component_cli, short_name=short_name,
                          execution_order=execution_order, owner_name=owner,
                          is_active=True, auto_update=False)

    # Now actually check out the repository.
    # $ASTRA_COMPONENT_DIR/{GITHUB_REPO_SLUG}/{RELEASE}/
    # But the tarball will extract it to a folder, so we will rename later.
    ASTRA_COMPONENT_DIR = os.getenv("ASTRA_COMPONENT_DIR")
    dirname = os.path.abspath(os.path.join(ASTRA_COMPONENT_DIR, github_repo_slug))

    # Create directory.
    os.makedirs(dirname, exist_ok=True)

    # Checkout repository at this tag version.
    _, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    log.info(f"Checking out repository {github_repo_slug} release {release} "\
             f"to {tmp_path}")
    r = requests.get(release_info["target"]["tarballUrl"], stream=True)
    with open(tmp_path, "wb") as fp:
        copyfileobj(r.raw, fp)

    log.info(f"Download complete. Untarring to {dirname}")
    os.system(f"tar -xzf {tmp_path} --directory {dirname}")

    # It will extract things into it's own folder, so move things.
    sha = release_info["target"]["sha"]
    os.rename(f"{dirname}/{owner}-{repository_name}-{sha[:7]}/",
              f"{dirname}/{release}/")
    os.unlink(tmp_path)

    dirname = os.path.join(dirname, release)

    # Strip the $ASTRA_COMPONENT_DIR.
    component.local_path = "$ASTRA_COMPONENT_DIR" \
                         + dirname[len(ASTRA_COMPONENT_DIR):]

    # TODO: Check that the component cli works? That it installs?

    session.add(component)

    log.info(f"Component {github_repo_slug} release {release} lives at {component.local_path}")

    session.commit()

    # TODO: What data should it run on?
    # TODO: Should we trigger it to run on data immediately?
    return component


def refresh(github_repo_slug):
    r"""
    Check GitHub for a new release of this repository. If a new release exists,
    then mark all earlier releases as inactive and create a new component with
    the latest release. 

    :param github_repo_slug:
        The GitHub repository in 'slug' form: {OWNER}/{REPOSITORY_NAME}.
    """

    # Do we have any components with this repo slug?
    github_repo_slug = _valid_github_repo_slug(github_repo_slug)

    # TODO: ascending or descending?
    last_release = session.query(Component) \
                          .filter_by(github_repo_slug=github_repo_slug) \
                          .order_by(Component.release.desc()) \
                          .first()

    # Check GitHub for new version.
    raise NotImplementedYet("""because requires a thinko w.r.t. multiple active 
                               components with the same github_repo_slug, 
                               different releases, and both set to auto-update
                            """)


def update(github_repo_slug, release, **kwargs):
    r"""
    Update attributes of an existing component.
    
    :param github_repo_slug:
        The GitHub repository in 'slug' form: {OWNER}/{REPOSITORY_NAME}.

    :param release:
        The version release.


    Optional keyword arguments include:

    :param is_active: [optional]
        Toggle the component to be active or not. Only active components are
        executed on reduced data products.

    :param auto_update: [optional]
        Toggle the component to automatically update with new releases from
        GitHub.

    :param short_name: [optional]
        Set the short descriptive name for this component.

    :param execution_order: [optional]
        Set the execution order for this component.

    :param component_cli: [optional]
        Set the command line utility to be executed.

    # TODO: long_description, owner_*, ... others ...
    """

    github_repo_slug = _valid_github_repo_slug(github_repo_slug)
    component = _get_component_or_none(github_repo_slug, release)
    if component is None:
        raise ValueError(f"no component found with slug {github_repo_slug} "\
                         f"and release {release}")

    acceptable_keywords = ("is_active", "auto_update", "short_name", 
                           "execution_order", "component_cli")
    relevant_keywords = set(acceptable_keywords).intersection(kwargs)
    kwds = dict([(k, kwargs[k]) for k in relevant_keywords])

    if kwds:
        log.info(f"Updating component {component} with {kwds}")
        for k, v in kwds.items():
            setattr(component, k, v)

        component.modified = datetime.datetime.utcnow()
        session.commit()

    else:
        log.info(f"Nothing to update on component {component} "\
                 f"(acceptable keywords: {acceptable_keywords})")

    return component


def delete(github_repo_slug, release):
    r"""
    'Delete' a component by marking it as inactive.

    :param github_repo_slug:
        The GitHub repository in 'slug' form: {OWNER}/{REPOSITORY_NAME}.

    :param release:
        The version release.
    """

    return update(github_repo_slug, release, is_active=False)



def _get_component_or_none(github_repo_slug, release):
    return session.query(Component) \
                  .filter_by(github_repo_slug=github_repo_slug,
                             release=release) \
                  .one_or_none()
