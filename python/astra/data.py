from __future__ import absolute_import, division, print_function, unicode_literals

import os
from astra.db.connection import engine, session
from astra.db.models.watched_folders import WatchedFolder



def watch_folder(path, recursive=False, interval=3600, regex_ignore_pattern=None):
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

    path = os.path.abspath(os.path.realpath(path))
    if not os.path.exists(path):
        raise IOError(f"path '{path}' does not exist")

    if not os.path.isdir(path):
        raise IOError(f"path '{path}' is not a directory")

    recursive = bool(recursive)
    interval = int(interval)
    if 0 > interval:
        raise ValueError("interval must be a non-negative integer")

    # Check if path is already monitored, or just let SQL work it out?
    item = WatchedFolder(path=path, recursive=recursive, interval=interval,
                         regex_ignore_pattern=regex_ignore_pattern)
    session.add(item)
    session.commit()

    return item


def unwatch_folder():

    # Check if 
    pass


def refresh_folders():
    pass

def refresh_folder():
    pass


