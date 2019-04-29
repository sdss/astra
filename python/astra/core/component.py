from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime
import requests
import tempfile
from shutil import copyfileobj
from astra import log
from astra.db.connection import session
from astra.db.models.component import Component
from astra.utils import github

from sdss_install.install import Install
from sdss_install.application import Argument#, sdss_install as sdss_install_parser

import pkg_resources
from importlib import import_module
from pkg_resources import DistributionNotFound, VersionConflict


def installer(options):

    installation = Install(options=options)

    log.info("Getting ready")
    installation.set_ready()

    log.info("Setting product/directory/directory_install/directory_work")
    installation.set_product()
    installation.set_directory()
    installation.set_directory_install()
    installation.set_directory_work()

    log.info("Cleaning directory_install and setting remote GitHub URL")
    installation.clean_directory_install()
    installation.set_github_remote_url()

    log.info("Fetching repository")
    installation.fetch()

    return installation

    installation.reset_options_from_config()
    installation.set_build_type()

    if not options.module_only:
        installation.logger_build_message()
        installation.make_directory_install()

    if installation.ready:
        installation.set_modules()
        installation.modules.set_ready()
        installation.modules.set_file()
        installation.modules.load_dependencies()
        installation.modules.set_keywords()
        installation.modules.set_directory()
        installation.modules.build()
    
    installation.set_environ()
    if not options.module_only:
        installation.build()
        installation.build_documentation()
        installation.build_package()
        if not options.keep: installation.clean()

    return installation


def add(product, version=None, execution_order=0, component_cli=None, description=None,
        module_name=None, test=False, **kwargs):
    r"""
    Add a component for analysing reduced data products.

    :param product:
        The GitHub repository name, or the name of the SDSS product. This should refer to a
        repository name that is owned by the SDSS organization.

    :param version: [optional]
        The version to use. If `None` is given then the most recent release on GitHub will be used.

    :param execution_order: [optional]
        The execution order priority (ascending non-negative execution order).
        See the documentation online for more details about the execution order.

    :param component_cli: [optional]
        The command line utility that will be executed by this component. If `None` is given then
        this will default to the first executable in the `bin/` directory of the repository. If
        there is more than one executable in the `bin/` directory and no `component_cli` is given
        then an exception will be raised.

    :param description: [None]
        A short, descriptive name for this component. If ``None`` is given
        then it will default to the short description that exists on GitHub.

    :param test: [optional]
        A boolean flag indicating whether to consider this a dry run (no installation) or not.

    :param module_name: [optional]
        Specify the module name for this component. This is normally inferred from the repository
        name.
    """

    product = github.validate_repository_name(product)
    owner = kwargs.get("__github_owner", "sdss")
    github_repo_slug = f"{owner}/{product}"

    # Check that this repository exists.
    repository = github.get_repository_summary(owner, product)
    if repository is None:
        raise ValueError(f"cannot find GitHub repository '{product}' with owner '{owner}'")

    log.debug(f"Repository summary: {product}")

    if version is None:
        log.info(f"Querying GitHub for the last release on {github_repo_slug}")

        last_release = github.get_most_recent_release(owner, product)
        log.info(f"GitHub response: {last_release}")

        if not last_release:
            raise ValueError(f"no releases available for {github_repo_slug}")

        version = last_release["name"]
        release_info = last_release

        log.info(f"Selecting release {version} ({release_info})")

    else:
        log.info(f"Checking that version {version} exists on GitHub")
        releases = github.get_most_recent_release(owner, product, n=100)
        for release_info in releases:
            if release_info["name"] == version:
                log.info(f"Release {version} found: {release_info}")
                break

        else:
            raise ValueError(f"could not find version {version} in last 100 releases on GitHub")

    # Check for a component that matches this slug.
    log.info(f"Checking for existing component ({github_repo_slug}: {version})")
    item = _get_component_or_none(github_repo_slug, version)
    if item is not None:
        raise ValueError(f"component already exists ({github_repo_slug} version {version}) as "\
                         f"component id {item.id}")

    # Check if we need to give it a short name.
    if description is None:
        description = repository["description"]
        log.info(f"Using GitHub description: {description}")

    if module_name is None:
        module_name = product
        log.info(f"Assuming {module_name} for the module name")

    args = ["--github", "--force", "--keep"]
    # TODO: Pass on alt-module
    if test or True:
        args += ["--test"]

    # TODO: Should we be doing this?

    # TODO: Here we are assuming the owner is SDSS and not allowing non-SDSS owned repositories to
    # be installed
    args.extend([product, version])

    options = Argument("sdss_install", args=args).options

    # Specify installation options.
    log.info(f"Installation options: {options}")

    installation = Install(options=options)

    log.info("Getting ready")
    installation.set_ready()

    log.info("Setting product/directory/directory_install/directory_work")
    installation.set_product()
    installation.set_directory()
    installation.set_directory_install()
    installation.set_directory_work()

    # Update work directory
    # If we *don't* make the temporary working directory here then sdss_install is just going to
    # ignore the full path and we'd be fucked anyways.
    twd = tempfile.mkdtemp(dir=os.getcwd())    
    installation.directory["work"] = twd
    installation.export_data()

    log.info("Cleaning directory_install and setting remote GitHub URL")
    installation.clean_directory_install()
    installation.set_github_remote_url()

    log.info("Fetching repository")
    installation.fetch()

    # Check for requirements.txt to see what is missing.
    requirements_path = os.path.join(twd, "requirements.txt")
    if os.path.exists(requirements_path):
        log.info(f"Checking requirements in {requirements_path}")
        with open(requirements_path, "r") as fp:
            requirements = [ea.strip() for ea in fp.readlines() if not ea.strip().startswith("#")]
        
        try:
            pkg_resources.require(requirements)

        except (DistributionNotFound, VersionConflict):
            log.exception(f"Cannot install {owner}/{product} because of version requirements")

        else:
            log.info("All requirements OK:\n{0}".format('\n'.join(requirements)))

    else:
        log.error(f"No requirements.txt file found at {requirements_path}")




    raise a


    # Create the component.
    component = Component(github_repo_slug=github_repo_slug, release=release,
                          component_cli=component_cli, description=description,
                          execution_order=execution_order, owner_name=owner,
                          module_name=module_name,
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

    # TODO: Use sdss_install to do the checkout & install
    tarball_url = f"https://github.com/{owner}/{repository_name}/archive/{release}.tar.gz"
    r = requests.get(tarball_url, stream=True)
    with open(tmp_path, "wb") as fp:
        copyfileobj(r.raw, fp)

    log.info(f"Download complete. Untarring to {dirname}")
    os.system(f"tar -xzf {tmp_path} --directory {dirname}")

    # It will extract things into it's own folder, so move things.
    os.rename(f"{dirname}/{repository_name}-{release}",
              f"{dirname}/{release}")
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
    github_repo_slug = github.validate_slug(github_repo_slug)

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

    github_repo_slug = github.validate_slug(github_repo_slug)
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
