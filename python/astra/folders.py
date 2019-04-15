from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import datetime
from astra import log
from astra.db.connection import session
from astra.db.models.watched_folders import WatchedFolder
from astra.db.models.data import DataProduct


# Right-hand strip any / values to match the input we would expect.
# For example, if the path /this/is/that *was* in the database, and here the
# `path` parameter was exactly: "/this/is/that" then if we executed
# os.path.dirname("/this/is/that") -> "/this/is"
# which is not what we want, obviously.
full_path = lambda path: os.path.abspath(os.path.realpath(path.rstrip("/")))


def watch(path, recursive=False, interval=3600, regex_ignore_pattern=None):
    f"""
    Start monitoring a directory for reduced SDSS data products.

    :param path:
        The local directory path to monitor.

    :param recursive: [optional]
        Recursively monitor all sub-directories (default: False).

    :param interval: [optional]
        The number of seconds between checking for new files (default: 3600).

    :param regex_ignore_pattern: [optional]
        A regular expression pattern that when matched against a new path, will
        cause the path to be ignored by Astra. For example, if "((\.log)|(\.txt))$"
        is provided to ``regex_ignore_pattern`` then Astra will *ignore*
        all files in ``path`` that have ".log" or ".txt" extensions.
    """

    path = full_path(path)
    if not os.path.exists(path):
        raise IOError(f"path '{path}' does not exist")

    if not os.path.isdir(path):
        raise IOError(f"path '{path}' is not a directory")

    recursive = bool(recursive)
    interval = int(interval)
    if 0 > interval:
        raise ValueError("interval must be a non-negative integer")

    # Check if path is already monitored.
    item = session.query(WatchedFolder).filter_by(path=path).one_or_none()
    if item is not None:
        item.is_active = True
        log.info(f"Re-activated previously watched folder {item}")

    else:
        item = WatchedFolder(path=path, recursive=recursive, interval=interval,
                             regex_ignore_pattern=regex_ignore_pattern)
        session.add(item)
        log.info(f"Created new watched folder {item}")

    session.commit()

    return item


def unwatch(path, quiet=False):
    r"""
    Stop monitoring a directory for reduced data products.

    :param path:
        The local directory path to stop monitoring.

    :param quiet: [optional]
        Be quiet if the specified `path` was not found in the database (default:
        False).
    """

    path = full_path(path)
    result = session.query(WatchedFolder).filter_by(path=path).one_or_none()

    if result is None:
        if quiet: return False
        raise ValueError(f"no watched folders matching '{path}'")

    result.is_active = False
    session.commit()

    return True


def refresh(path, quiet=False):
    r"""
    Check a watched folder for new reduced data products.

    :param path:
        The local directory path to refresh.

    :param quiet: [optional]
        Do not raise an exception if the folder is not being watched.

    :returns:
        A three-length tuple with the number of new paths added, the number of
        new paths that were ignored, and the number of existing paths that were
        skipped.
    """

    path = full_path(path)
    folder = session.query(WatchedFolder).filter_by(path=path).one_or_none()

    if folder is None or not folder.is_active:
        if quiet: return (0, 0, 0)
        raise ValueError(f"path '{path}' not being actively watched")

    # TODO: query if there is a faster way to do this.
    added, ignored, skipped = (0, 0, 0)
    for top_directory, directories, basenames in os.walk(path, topdown=True):
        print(top_directory, directories, basenames)

        for basename in basenames:
            p = os.path.join(top_directory, basename)

            if folder.regex_ignore_pattern is not None \
            and re.search(folder.regex_ignore_pattern, basename):
                log.info(f"Ignoring {p} because it matched regular expression "\
                         f"for ignoring files: '{folder.regex_ignore_pattern}'")
                ignored += 1
                continue

            # Add the data product if it doesn't exist already
            if session.query(DataProduct).filter_by(path=p).one_or_none() is None:
                session.add(DataProduct(path=p, folder_id=folder.id))
                log.info(f"Added {p} from watched folder {folder}")
                added += 1

            else:
                skipped += 1

        if not folder.recursive: break

    # Update the last checked time for this folder.
    folder.last_checked = datetime.datetime.utcnow()
    session.commit()

    return (added, ignored, skipped)


def refresh_all():
    r""" Check all watched folders for new reduced data products. """

    return dict([(f.path, refresh(f.path, True)) \
                 for f in session.query(WatchedFolder).all()])

