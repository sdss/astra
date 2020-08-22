from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime
import requests
import tempfile
from glob import glob
import shutil
from astra import __file__ as astra___file__
from astra.utils import log
from astra.db.connection import session
from astra.db.models.component import Component
from astra.utils import github

#from sdss_install.install import Install
from astra.utils.sdss_install_monkey_patch import Install
from sdss_install.application import Argument#, sdss_install as sdss_install_parser

import pkg_resources
from importlib import import_module
from pkg_resources import DistributionNotFound, VersionConflict




def add(product, version=None, owner=None, execution_order=0, command=None, description=None,
        module_name=None, install_args=None, default_args=None, test=False, **kwargs):
    r"""
    Add a component for analysing reduced data products.

    :param product:
        The GitHub repository name, or the name of the SDSS product. This should refer to a
        repository name that is owned by the SDSS organization.

    :param version: [optional]
        The version to use. If `None` is given then the most recent release on GitHub will be used.

    :param owner: [optional]
        The owner of the repository on GitHub. If `None` is given then the owner will be assumed 
        to be the SDSS organization.

    :param execution_order: [optional]
        The execution order priority (ascending non-negative execution order).
        See the documentation online for more details about the execution order.

    :param command: [optional]
        The command line utility that will be executed by this component. If `None` is given then
        this will default to the first executable in the `bin/` directory of the repository. If
        there is more than one executable in the `bin/` directory and no `command` is given
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
    owner = owner or "sdss"
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
    item = _get_component_or_none(owner, product, version)
    if item is not None:
        if item.is_active:
            raise ValueError(f"component already exists ({github_repo_slug} version {version}) as "\
                             f"component id {item.id}")

        else:
            item.is_active = True
            log.info(f"Setting component {item} as active")
            session.commit()

            return item


    # Check if we need to give it a short name.
    if description is None:
        description = repository["description"]
        log.info(f"Using GitHub description: {description}")

    if module_name is None:
        module_name = product
        log.info(f"Assuming {module_name} for the module name")

    args = ["--github", "--force", "--keep"] + ["-v"]        
    # TODO: Pass on alt-module
    if test:
        args += ["--test"]

    if "sdss" != owner:
        args += ["--public", f"{owner}/{product}", version]
    else:
        args += [product, version]

    options = Argument("sdss_install", args=args).options
    # TODO: the no_python_package may depend on whether we are using modules or not..
    options.no_python_package = False 
    options.evilmake = False
    

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

    directory_work = "" + installation.directory["work"]

    # Update work directory
    # If we *don't* make the temporary working directory here then sdss_install is just going to
    # ignore the full path and we'd be fucked anyways.
    twd = tempfile.mkdtemp(dir=os.getcwd())    
    installation.directory["work"] = twd
    installation.export_data()

    log.info("Cleaning directory_install and setting remote GitHub URL")
    installation.clean_directory_install()
    installation.set_sdss_github_remote_url()

    log.info("Fetching repository")
    installation.fetch()

    '''
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

            # Clean up the temporary directory.
            log.info(f"Cleaning up work directory {twd}")
            shutil.rmtree(twd)
            raise

        else:
            log.info("All requirements OK:\n{0}".format('\n'.join(requirements)))

    else:
        log.error(f"No requirements.txt file found at {requirements_path}")
    '''

    # Get the component CLI
    if command is None:
        # TODO: Should we parse this from setup.py instead?
        executables = glob(os.path.join(twd, "bin", "*"))
        if len(executables) < 1:
            raise ValueError(f"no executable found in bin/ directory ({twd}/bin) of "
                             f"{owner}/{product} {version}")
        elif len(executables) > 2:
            raise ValueError(f"multiple component CLIs found: {executables}")

        else:
            log.info(f"Setting component cli as {executables[0]}")
            command = os.path.basename(executables[0])

    # Update directory work.
    log.info(f"Updating work directory {twd} -> {directory_work}")
    shutil.move(twd, directory_work)
    installation.directory["work"] = directory_work

    # Continue the installation.
    installation.set_build_type()

    if not options.module_only:
        installation.logger_build_message()
        installation.make_directory_install()

    
    if installation.ready:
        installation.set_modules()

        # If we are using modules then we want `sdss_install` to manage the location of SDSS products.
        # But if we aren't using modules then we want this to be installed to the same location as
        # other python packages in this environment.

        if installation.options.modules_home is None:
            log.warning("No modules home path found!")
            
            site_package_dir = os.path.abspath(
                os.path.dirname(
                    os.path.join(
                        astra___file__, 
                        "../../../",
                    )
                )
            )
            log.info("Since you're not using modules, this component will be installed to the same "
                     "location as other site packages for this Python environment.")
            
            log.info(f"Previous installation directories: {installation.directory}")

            install_dir = os.path.join(site_package_dir, product)

            # TODO: Can we specify the root directory for sdss_install instead of having to do this?
            installation.directory["install"] = install_dir

            if os.path.exists(install_dir):
                # TODO: when running locally without modules we may have to state that you can only
                #       have one version of a component.
                shutil.rmtree(install_dir)

        else:
            installation.modules.set_ready()
            installation.modules.set_file()
            installation.modules.load_dependencies()
            installation.modules.set_keywords()
            installation.modules.set_directory()
            installation.modules.build()

            site_package_dir = installation.directory["install"]

    log.info(f"Site package directory: {site_package_dir}")

    installation.set_environ()

    if not options.module_only:
        installation.build(install_args=install_args)
        installation.build_documentation()
        installation.build_package()
        if not options.keep: installation.clean()

    installation.finalize()

    if not installation.ready:
        raise RuntimeError("installation failed")

    # Create the component.
    component = Component(owner=owner, product=product, version=version,
                          command=command, description=description,
                          execution_order=execution_order, module_name=module_name,
                          default_args=default_args,
                          is_active=True, auto_update=False)
    session.add(component)
    session.commit()

    log.info(f"Component {product} version {version} installed: {component}")

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


def update(product, version, owner=None, **kwargs):
    r"""
    Update attributes of an existing component.

    :param product:
        The GitHub repository name, or the name of the SDSS product. This should refer to a
        repository name that is owned by the SDSS organization.

    :param version: 
        The version to delete.

    :param owner: [optional]
        The owner of the repository on GitHub. If `None` is given then the owner will be assumed 
        to be the SDSS organization.


    Optional keyword arguments include:

    :param is_active: [optional]
        Toggle the component to be active or not. Only active components are
        executed on reduced data products.

    :param auto_update: [optional]
        Toggle the component to automatically update with new releases from
        GitHub.

    :param execution_order: [optional]
        Set the execution order for this component.

    :param command: [optional]
        Set the command line utility to be executed.
    """

    product = github.validate_repository_name(product)
    owner = owner or "sdss"
    github_repo_slug = f"{owner}/{product}"

    component = _get_component_or_none(owner, product, version)
    if component is None:
        raise ValueError(f"no component found with {owner}/{product} and version {version}")

    acceptable_keywords = ("is_active", "auto_update", "short_name", 
                           "execution_order", "command")
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


def delete(product, version, owner=None):
    r"""
    Delete a component by marking it as inactive.

    :param product:
        The GitHub repository name, or the name of the SDSS product. This should refer to a
        repository name that is owned by the SDSS organization.

    :param version: 
        The version to delete.

    :param owner: [optional]
        The owner of the repository on GitHub. If `None` is given then the owner will be assumed 
        to be the SDSS organization.
    """

    return update(product, version, owner=owner, is_active=False)


def _get_component_or_none(owner, product, version):
    return session.query(Component) \
                  .filter_by(owner=owner, product=product, version=version) \
                  .one_or_none()


def _get_likely_component(identifier):
    """
    Return the most likely coponent, given some identifier about that coponent. The identifier could
    be the coponent `id`, or the name of the `component` (if only one version exists), or it could
    be a string of the form "{product}/{version}".
    """

    if isinstance(identifier, Component):
        # This is a component.
        return identifier

    possible_filters = []

    try:
        identifier = int(identifier)

    except:
        if "/" in identifier:

            num_fs = identifier.count("/")
            if num_fs == 2:
                # It's a "{owner}/{product}/{version}" string.
                owner, product, version = identifier.split("/")
                possible_filters.append(dict(owner=owner, product=product, version=version))

            elif num_fs == 1:
                # It's either a {owner}/{product} string (where version is assumed)
                # or it's a {product}/{version} string.
                a, b = identifier.split("/")
                possible_filters.extend([
                    dict(owner=a, product=b),
                    dict(product=a, version=b)
                ])

            else:
                raise ValueError(f"unrecognised component identifier format '{identifier}'")

        else:
            # The identifier is a product name. Let's hope it's unique.
            possible_filters.append(dict(product=identifier))

    else:
        possible_filters.append(dict(id=identifier))

    results = []
    for filters in possible_filters:
        results.extend(session.query(Component).filter_by(**filters).all())

    results = list(set(results))

    if len(results) == 0:
        raise ValueError(f"No component found with identifier '{identifier}'. "
                         f"Filters: {possible_filters}")

    elif len(results) > 1:
        raise ValueError(f"Multiple components found with identifier '{identifier}': {results} "
                         f"Filters: {possible_filters}")

    else:
        return results[0]

