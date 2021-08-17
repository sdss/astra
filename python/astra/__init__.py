# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import os
import re
import sys
import yaml
from pkg_resources import parse_version
from sdsstools.logger import (color_text, get_logger)

__version__ = '0.2.0-dev'

def merge(user, default):
    """Merges a user configuration with the default one."""

    if isinstance(user, dict) and isinstance(default, dict):
        for kk, vv in default.items():
            if kk not in user:
                user[kk] = vv
            else:
                user[kk] = merge(user[kk], vv)

    return user


NAME = 'astra'


# Loads config
yaml_kwds = dict()
if parse_version(yaml.__version__) >= parse_version("5.1"):
    yaml_kwds.update(Loader=yaml.FullLoader)

config_path = os.path.join(os.path.dirname(__file__), 'etc/{0}.yml'.format(NAME))
with open(config_path, "r") as fp:
    config = yaml.load(fp, **yaml_kwds)

# If there is a custom configuration file, updates the defaults using it.
custom_config_path = os.path.expanduser('~/.{0}/{0}.yml'.format(NAME))
if os.path.exists(custom_config_path):
    with open(custom_config_path, "r") as fp:
        custom_config = yaml.load(custom_config_path, **yaml_kwds)
    config = merge(custom_config, config)


def _colored_formatter(record):
    """Prints log messages with colours."""

    colours = {'info': 'blue',
               'debug': 'magenta',
               'warning': 'yellow',
               'critical': 'red',
               'error': 'red'}

    timestamp = datetime.fromtimestamp(record.created).strftime("[%H:%M:%S]") # %y/%m/%d 
    levelname = record.levelname.lower()

    if levelname.lower() in colours:
        levelname_color = colours[levelname]
        header = color_text('[{}]: '.format(levelname.upper()),
                            levelname_color)
    else:
        header = f'[{levelname}]'

    message = record.getMessage()

    if levelname == 'warning':
        warning_category_groups = re.match(r'^.*?\s*?(\w*?Warning): (.*)', message)
        if warning_category_groups is not None:
            warning_category, warning_text = warning_category_groups.groups()

            # Temporary ignore warnings from pymodbus. The normal warnings.simplefilter
            # does not work because pymodbus forces them to show.
            if re.match('"@coroutine" decorator is deprecated.+', warning_text):
                return

            warning_category_colour = color_text('({})'.format(warning_category), 'cyan')
            message = '{} {}'.format(color_text(warning_text, ''), warning_category_colour)

    sys.__stdout__.write('{} {}{}\n'.format(timestamp, header, message))
    sys.__stdout__.flush()

    return


def _setup_astra_logging():
    # Only set up logging ONCE.
    astra_log = get_logger("astra")
    # Remove excessive handlers.
    for handler in astra_log.handlers[1:]:
        astra_log.removeHandler(handler)
    
    astra_log.handlers[0].setLevel(astra_log.getEffectiveLevel())
    astra_log.handlers[0].emit = _colored_formatter
    # If you set propagate = True then the astra logs do not appear
    # in the AirFlow UI logs.
    # The downside to this is that when running 'dag test' on the command
    # line, the logs get repeated.
    # TODO: solve this some day in the future when you have more patience.
    #astra_log.propagate = False


    return astra_log

_log = _setup_astra_logging()