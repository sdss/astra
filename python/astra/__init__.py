# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import yaml
from pkg_resources import parse_version


__version__ = '0.1.11-dev'

try:
    from luigi import build
    from luigi.util import inherits, requires
    from luigi.parameter import (Parameter, OptionalParameter, DateParameter,
        IntParameter, FloatParameter, BoolParameter, TaskParameter, DictParameter,
        EnumParameter, EnumListParameter, ListParameter, NumericalParameter, ChoiceParameter)

except ImportError:
    # When the user is running python setup.py install we want them to be able to 
    print(f"Warning: cannot import luigi requirements")

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


# Deal with luigi logging.
"""
import logging
from sdsstools.logger import get_logger

log = get_logger(__name__.split(".")[0])
log.propagate = False

luigi_interface = logging.getLogger("luigi-interface")
luigi_interface.propagate = False
luigi_interface.handlers = log.handlers
"""