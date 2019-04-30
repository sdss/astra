from __future__ import absolute_import, division, print_function, unicode_literals


import click
from astra.utils import log
from astra.core import folder

@click.group()
@click.pass_context
def parser(context):
    r""" Manage monitoring of data folders. """
    log.debug("folder")
    pass

@parser.command()
@click.argument("path", nargs=1, required=True)
@click.option("-r", "--recursive", is_flag=True, default=False,
              help="Monitor recursively (default: `False`).")
@click.option("--interval", default=3600,
              help="Number of seconds to wait between checking for new data (default: `3600`).")
@click.option("--regex-match-pattern", default=None,
              help="A regular expression pattern that, if given, only files that match this pattern"
                   " will be acted upon.")
@click.option("--regex-ignore-pattern", default=None,
              help="A regular expression pattern that, when matched, will ignore files in the "
                   "watched folder. If both --regex-ignore-pattern and --regex-match-pattern are "
                   "given, then any paths that match *both* will be ignored.")
@click.pass_context
def watch(context, path, recursive, interval, regex_match_pattern, regex_ignore_pattern):
    r"""
    Start monitoring a  folder for new data.
    """
    log.debug(f"folder.watch {path} {recursive} {interval} {regex_match_pattern} {regex_ignore_pattern}")
    result = folder.watch(path, recursive, interval, regex_match_pattern, regex_ignore_pattern)
    log.info(result)

    # Now refresh it.
    folder.refresh(result.path)
    return True


@parser.command()
@click.argument("path", nargs=1, required=True)
@click.option("--quiet", is_flag=True, default=False,
              help="Do not raise an exception if the given path is not actively being watched.")
@click.pass_context
def unwatch(context, path, quiet):
    r"""
    Stop monitoring a folder for new data.
    """
    log.debug(f"folder.unwatch {path} {quiet}")
    result = folder.unwatch(path, quiet)
    log.info(result)
    return result


@parser.command()
@click.argument("path", nargs=-1)
@click.option("--quiet", is_flag=True, default=False,
              help="Do not raise an exception if the given path is not actively being watched.")
@click.pass_context
def refresh(context, path, quiet):
    r"""
    Refresh watched folder(s) for new data.
    """
    log.debug(f"folder.refresh {path}")

    if not len(path):
        # Refresh all.
        counts = folder.refresh_all()

    else:
        # Refresh given paths.
        counts = dict([(p, folder.refresh(p, quiet)) for p in path])

    log.info(counts)
    return counts

