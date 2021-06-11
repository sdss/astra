# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import os
import re
import logging
import signal
import sys
import yaml
from pkg_resources import parse_version
from sdsstools.logger import (color_text, get_logger)

__version__ = '0.1.12-dev'

try:
    from luigi import build as luigi_build
    from luigi.util import inherits, requires
    from luigi.parameter import (Parameter, OptionalParameter, DateParameter,
        IntParameter, FloatParameter, BoolParameter, TaskParameter, DictParameter,
        EnumParameter, EnumListParameter, ListParameter, NumericalParameter, ChoiceParameter)

    from luigi.interface import (
        _WorkerSchedulerFactory, core, lock, PidLockAlreadyTakenExit, LuigiRunResult,
        InterfaceLogging
    )

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
    
    astra_log.handlers[0].emit = _colored_formatter
    astra_log.propagate = False

    return astra_log

_log = _setup_astra_logging()

luigi_interface = logging.getLogger("luigi-interface")
luigi_interface.setLevel(_log.getEffectiveLevel())    
if not luigi_interface.handlers:
    luigi_interface.addHandler(_log.handlers[0])
luigi_interface.propagate = False


def build(
        tasks,
        worker_scheduler_factory=None,
        detailed_summary=False,
        **override_defaults
    ):

    if worker_scheduler_factory is None:
        worker_scheduler_factory = _WorkerSchedulerFactory()

    if "no_lock" not in override_defaults:
        override_defaults["no_lock"] = True

    env_params = core(**override_defaults)
        

    if worker_scheduler_factory is None:
        worker_scheduler_factory = _WorkerSchedulerFactory()
    if override_defaults is None:
        override_defaults = {}
    env_params = core(**override_defaults)


    kill_signal = signal.SIGUSR1 if env_params.take_lock else None
    if (not env_params.no_lock and
            not(lock.acquire_for(env_params.lock_pid_dir, env_params.lock_size, kill_signal))):
        raise PidLockAlreadyTakenExit()

    if env_params.local_scheduler:
        sch = worker_scheduler_factory.create_local_scheduler()
    else:
        if env_params.scheduler_url != '':
            url = env_params.scheduler_url
        else:
            url = 'http://{host}:{port:d}/'.format(
                host=env_params.scheduler_host,
                port=env_params.scheduler_port,
            )
        sch = worker_scheduler_factory.create_remote_scheduler(url=url)

    worker = worker_scheduler_factory.create_worker(
        scheduler=sch, worker_processes=env_params.workers, assistant=env_params.assistant)

    success = True
    logger = logging.getLogger('luigi-interface')
    with worker:
        for t in tasks:
            success &= worker.add(t, env_params.parallel_scheduling, env_params.parallel_scheduling_processes)
        logger.info('Done scheduling tasks')
        success &= worker.run()
    luigi_run_result = LuigiRunResult(worker, success)
    logger.info(luigi_run_result.summary_text)

    return luigi_run_result if detailed_summary else luigi_run_result.scheduling_succeeded

#redirect_luigi_interface_logging()