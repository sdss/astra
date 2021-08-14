import os
from re import A
import uuid
import pickle
import inspect
import numpy as np
from tqdm import tqdm
from tempfile import mktemp
from time import sleep, time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from airflow.models import BaseOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from astropy.time import Time
from sqlalchemy import and_, func
from sdss_access import SDSSPath

from astra.utils import log, get_base_output_path
from astra.database import astradb, session, apogee_drpdb
from astra.database.utils import (create_task_instance, deserialize_pks)
from astra.tools.spectrum import Spectrum1D
from astra.utils.continuum.sines_and_cosines import normalize as normalize_with_sines_and_cosines


class AstraOperator(BaseOperator):
    """
    A base operator for performing work on SDSS data products.
    
    :param spectrum_kwargs: a dictionary of keyword arguments that will get unpacked
        in to `astra.tools.spectrum.Spectrum1D.read`
    :type spectrum_kwargs: dict
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict (templated)
    :param slurm_kwargs: a dictionary of keyword arguments that will be submitted
        to the slurm queue
    :type slurm_kwargs: dict
    """

    ui_color = "#CEE8F2"

    template_fields = ("op_kwargs", "slurm_kwargs")
    template_fields_renderers = {
        "op_kwargs": "py",
        "slurm_kwargs": "py"
    }

    shallow_copy_attrs = (
        "op_kwargs",
        "spectrum_kwargs",
        "slurm_kwargs"
    )

    def __init__(
        self,
        *,
        op_kwargs: Optional[Dict] = None,
        slurm_kwargs: Optional[Dict] = None,
        spectrum_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.op_kwargs = op_kwargs or dict()
        self.slurm_kwargs = slurm_kwargs or dict()
        self.spectrum_kwargs = spectrum_kwargs or dict()
        return None


    def yield_data(self):
        yield from _yield_data(self.pks, **self.spectrum_kwargs)
    
    
    def pre_execute(self, context: Any):
        """
        Create task instances for all the data model identifiers. 
        
        :param context:
            The Airflow context dictionary.
        """

        args = (context["dag"].dag_id, context["task"].task_id, context["run_id"])

        # Get parameters from the parent class initialisation that should also be stored.
        common_parameter_names = set(inspect.signature(self.__init__).parameters).difference(("kwargs", ))
        common_parameters = { pn: getattr(self, pn) for pn in common_parameter_names }

        pks = []
        for data_model_identifiers in self.yield_data_model_identifiers(context):
            parameters = {**common_parameters, **data_model_identifiers}
            pks.append(create_task_instance(*args, parameters).pk)

        if not pks:
            raise AirflowSkipException("no data model identifiers found for this time period")
        
        self.pks = pks
        return pks
        

    def execute_by_slurm(
            self, 
            context: Any, 
            bash_command: Any,
            poke_interval: Optional[int] = 60
        ):

        log.info(f"Submitting Slurm job with command:\n{bash_command}")
        
        uid = str(uuid.uuid4())[:8]
        label = ".".join([
            context["dag"].dag_id,
            context["task"].task_id,
            context["execution_date"].strftime('%Y-%m-%d'),
            # run_id is None if triggered by command line
            uid
        ])

        # It's bad practice to import here, but the slurm package is
        # not easily installable outside of Utah, and is not a "must-have"
        # requirement. 
        from slurm import queue

        slurm_kwargs = (self.slurm_kwargs or dict())
        q = queue(verbose=False)
        q.create(label=label, **slurm_kwargs)
        q.append(bash_command)
        q.commit(hard=True, submit=True)

        log.info(f"Slurm job submitted with {q.key} and keywords {slurm_kwargs}")
        log.info(f"\tJob directory: {q.job_dir}")

        stdout_path = os.path.join(q.job_dir, f"{label}_01.o")
        stderr_path = os.path.join(q.job_dir, f"{label}_01.e")    

        # Now we wait until the Slurm job is complete.
        t_init = time()
        while 100 > q.get_percent_complete():

            sleep(poke_interval)

            t = time() - t_init

            if not os.path.exists(stderr_path) and not os.path.exists(stdout_path):
                log.info(f"Waiting on job {q.key} to start (elapsed: {t / 60:.0f} min)")

            else:
                log.info(f"Waiting on job {q.key} to finish (elapsed: {t / 60:.0f} min)")
                # Open last line of stdout path?

        log.info(f"Job {q.key} in {q.job_dir} is complete after {(time() - t_init)/60:.0f} minutes.")

        with open(stderr_path, "r", newline="\n") as fp:
            stderr = fp.read()
        log.info(f"Contents of {stderr_path}:\n{stderr}")

        with open(stdout_path, "r", newline="\n") as fp:
            stdout = fp.read()
        log.info(f"Contents of {stdout_path}:\n{stdout}")
        
        # TODO: Better parsing for critical errors.
        if "Error" in stderr.rstrip().split("\n")[-1]:
            raise RuntimeError(f"detected exception at task end-point")

        return None
        



class ApStarOperator(AstraOperator):

    template_fields = ("op_kwargs", "slurm_kwargs")
    template_fields_renderers = {
        "op_kwargs": "py",
        "slurm_kwargs": "py"
    }

    shallow_copy_attrs = (
        "op_kwargs",
        "spectrum_kwargs",
        "slurm_kwargs"
    )
    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        release: Optional[str] = "sdss5",
        skip_sources_with_more_recent_observations: Optional[bool] = True,
        query_filter_by_kwargs: Optional[Dict] = None,
        op_kwargs: Optional[Dict] = None,
        slurm_kwargs: Optional[Dict] = None,
        spectrum_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.limit = limit
        self.release = release
        self.query_filter_by_kwargs = query_filter_by_kwargs
        self.skip_sources_with_more_recent_observations = skip_sources_with_more_recent_observations


    def execute(self, context: Any):

        # pre_execute: create task instances for all the data model identifiers,
        #              and store the primary keys somewhere...

        
        raise NotImplementedError
    

    def yield_data_model_identifiers(self, context: Any):
        """ Yield data model identifiers matching this operator's constraints. """

        #try:
        #    yield from super().yield_data_model_identifiers(context)
        #except:
        #    None
        
        if self.release != "sdss5":
            raise NotImplementedError
        
        obs_start = parse_as_mjd(context["prev_ds"])
        obs_end = parse_as_mjd(context["ds"])

        log.debug(f"Parsed MJD range as between {obs_start} and {obs_end}")
        
        filetype, apstar = ("apStar", "stars") # TODO: Raise apstar keyword with Nidever
        columns = (
            apogee_drpdb.Star.apred_vers.label("apred"), # TODO: Raise with Nidever
            apogee_drpdb.Star.healpix,
            apogee_drpdb.Star.telescope,
            apogee_drpdb.Star.apogee_id.label("obj"), # TODO: Raise with Nidever
        )
        
        if not self.skip_sources_with_more_recent_observations:
            q = session.query(*columns).distinct(*columns)
        else:
            # Get the max MJD of any observations for this source.
            sq = session.query(
                *columns, 
                func.max(apogee_drpdb.Star.mjdend).label('max_mjdend')
            ).group_by(*columns).subquery()

            q = session.query(*columns).join(
                sq, 
                and_(
                    apogee_drpdb.Star.mjdend == sq.c.max_mjdend,
                    apogee_drpdb.Star.apred_vers == sq.c.apred,
                    apogee_drpdb.Star.healpix == sq.c.healpix,
                    apogee_drpdb.Star.telescope == sq.c.telescope,
                    apogee_drpdb.Star.apogee_id == sq.c.obj
                )        
            )
        
        # Filter on number of good RV measurements, and the MJD of last obs 
        q = q.filter(apogee_drpdb.Star.mjdend < obs_end)\
             .filter(apogee_drpdb.Star.mjdend >= obs_start)

        if self.query_filter_by_kwargs is not None:
            q = q.filter_by(**self.query_filter_by_kwargs)

        if self.limit is not None:
            q = q.limit(self.limit)

        log.debug(f"Preparing query {q}")
        total = q.count()
        log.debug(f"Retrieved {total} rows")

        keys = [column.name for column in columns]

        for values in q.yield_per(1):
            d = dict(zip(keys, values))
            d.update(
                release=self.release,
                filetype=filetype,
                apstar=apstar,
            )
            yield d



def _yield_data(pks, **kwargs):
    
    trees = {}

    for pk in deserialize_pks(pks, flatten=True):
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == pk)
        instance = q.one_or_none()

        if instance is None:
            log.warning(f"No task instance found for primary key {pk}")
            path = None
            spectrum = None

        else:
            release = instance.parameters["release"]
            tree = trees.get(release, None)
            if tree is None:
                trees[release] = tree = SDSSPath(release=release)

            path = tree.full(**instance.parameters)
        
            try:
                spectrum = Spectrum1D.read(path, **kwargs)

            except:
                log.exception(f"Unable to load Spectrum1D from path {path} on task instance {instance}")
                spectrum = None

        yield (pk, instance, path, spectrum)
    

def parse_as_mjd(mjd):
    """
    Parse Modified Julian Date, which might be in the form of an execution date
    from Apache Airflow (e.g., YYYY-MM-DD), or as a MJD integer. The order of
    checks here is:

        1. if it is not a string, just return the input
        2. if it is a string, try to parse the input as an integer
        3. if it is a string and cannot be parsed as an integer, parse it as
           a date time string

    :param mjd:
        the Modified Julian Date, in various possible forms.
    
    :returns:
        the parsed Modified Julian Date
    """
    if isinstance(mjd, str):
        try:
            mjd = int(mjd)
        except:
            return Time(mjd).mjd
    return mjd