import os
import uuid
import inspect
from time import sleep, time
from airflow.models import BaseOperator
from airflow.exceptions import AirflowSkipException
from astropy.time import Time
from sqlalchemy import and_, func
from sdss_access import SDSSPath
from datetime import datetime
from subprocess import CalledProcessError

from astra.utils import log
from astra.database import astradb, catalogdb, session, apogee_drpdb
from astra.database.utils import (create_task_instance, deserialize_pks)
from astra.tools.spectrum import Spectrum1D


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

    def execute_by_slurm(
            self, 
            context, 
            bash_command,
            poke_interval=60
        ):
        
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
        
        # TODO: HACK to be able to use local astra installation while in development
        if bash_command.startswith("astra "):
            bash_command = f"/uufs/chpc.utah.edu/common/home/u6020307/.local/bin/astra {bash_command[6:]}"

        slurm_kwargs = (self.slurm_kwargs or dict())

        log.info(f"Submitting Slurm job {label} with command:\n\t{bash_command}\nAnd Slurm keyword arguments: {slurm_kwargs}")        
        q = queue(verbose=True)
        q.create(label=label, **slurm_kwargs)
        q.append(bash_command)
        try:
            q.commit(hard=True, submit=True)
        except CalledProcessError as e:
            log.exception(f"Exception occurred when committing Slurm job with output:\n{e.output}")
            raise

        log.info(f"Slurm job submitted with {q.key} and keywords {slurm_kwargs}")
        log.info(f"\tJob directory: {q.job_dir}")

        stdout_path = os.path.join(q.job_dir, f"{label}_01.o")
        stderr_path = os.path.join(q.job_dir, f"{label}_01.e")    

        # Now we wait until the Slurm job is complete.
        t_submitted, t_started = (time(), None)
        while 100 > q.get_percent_complete():

            sleep(poke_interval)

            t = time() - t_submitted

            if not os.path.exists(stderr_path) and not os.path.exists(stdout_path):
                log.info(f"Waiting on job {q.key} to start (elapsed: {t / 60:.0f} min)")

            else:
                # Check if this is the first time it has started.
                if t_started is None:
                    t_started = time()
                    log.debug(f"Recording job {q.key} as starting at {t_started} (took {t / 60:.0f} min to start)")

                log.info(f"Waiting on job {q.key} to finish (elapsed: {t / 60:.0f} min)")
                # Open last line of stdout path?

                # If this has been going much longer than the walltime, then something went wrong.
                # TODO: Check on the status of the job from Slurm.

        log.info(f"Job {q.key} in {q.job_dir} is complete after {(time() - t_submitted)/60:.0f} minutes.")

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
        

class SDSSDataProductOperator(AstraOperator):


    def yield_data(self):
        spectrum_kwargs = getattr(self, "spectrum_kwargs", {})
        yield from _yield_data(self.pks, **spectrum_kwargs)
    

    def pre_execute(self, context):
        """
        Create task instances for all the data model identifiers. 
        
        :param context:
            The Airflow context dictionary.
        """

        args = (context["dag"].dag_id, context["task"].task_id, context["run_id"])

        # Get parameters from the parent class initialisation that should also be stored.
        ignore = ("kwargs", "slurm_kwargs")
        common_parameter_names = set(inspect.signature(self.__init__).parameters).difference(ignore)
        common_parameters = { pn: getattr(self, pn) for pn in common_parameter_names }

        pks = []
        for data_model_identifiers in self.yield_data_model_identifiers(context):
            parameters = {**common_parameters, **data_model_identifiers}
            pks.append(create_task_instance(*args, parameters).pk)

        if not pks:
            raise AirflowSkipException("No data model identifiers found for this time period.")
        
        self.pks = pks
        return None
        


    def infer_release(self, context):
        """
        Infer the SDSS release based on the DAG execution date. 
        
        If the start date is greater than or equal to 2020-10-24 then we infer the release to be 'sdss5'. 
        
        If it is earlier then we infer it to be 'DR17'.

        :param context:
            The DAG execution context.
        """
        
        start_date = datetime.strptime(context["ds"], "%Y-%m-%d")
        sdss5_start_date = datetime(2020, 10, 24)
        release = "sdss5" if start_date >= sdss5_start_date else "DR17"

        log.info(f"Inferring `release` for SDSS data product operator {self} to be '{release}' based on {start_date}")
        return release


    def yield_data_model_identifiers(self, context):
        """ Yield data model identifiers matching this operator's constraints. """

        # Hierarchically yield so that we can have mixed operators that supply data model identifiers
        # for two or mode data model classes. For example:
        #
        # class MyOperator(ApVisitOperator, BossSpecOperator):
        #    pass
        #
        # This would yield data model identifiers for ApVisit files *and* for BossSpec files.

        yield from self.query_data_model_identifiers_from_database(context)

        # TODO: This needs testing! 
        '''
        for mro in self.__class__.__mro__:
            try:
                yield from super(mro, self).query_data_model_identifiers_from_database(context)
            except AttributeError:
                break
        '''
        
        # TODO: Allow data model identifiers to be read from an AllStar file, if a filename
        #       is supplied when the ApStar operator was __init__'d.


class ApStarOperator(SDSSDataProductOperator):

    def __init__(
        self,
        *,
        limit = None,
        release = None,
        skip_sources_with_more_recent_observations = True,
        query_filter_by_kwargs = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.limit = limit
        self.release = release
        self.query_filter_by_kwargs = query_filter_by_kwargs
        self.skip_sources_with_more_recent_observations = skip_sources_with_more_recent_observations
    
    
    def query_data_model_identifiers_from_database(self, context):
        """
        Query the SDSS-V database for data model identifiers.
    
        :param context:
            The DAG execution context.
        """

        # If the release is None, infer it from the execution date.
        release = self.release or self.infer_release(context)
                
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
                release=release,
                filetype=filetype,
                apstar=apstar,
            )
            yield d



class ApVisitOperator(SDSSDataProductOperator):

    def __init__(
        self,
        *,
        limit = None,
        release = None,
        query_filter_by_kwargs = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.limit = limit
        self.release = release
        self.query_filter_by_kwargs = query_filter_by_kwargs


    def query_data_model_identifiers_from_database(self, context):
        """
        Query the SDSS-V database for ApVisit data model identifiers.
        
        :param context:
            The DAG execution context.
        """

        # If the release is None, infer it from the execution date.
        filetype = "apVisit"
        release = self.release or self.infer_release(context)
                
        obs_start = parse_as_mjd(context["prev_ds"])
        obs_end = parse_as_mjd(context["ds"])

        log.debug(f"Parsed MJD range as between {obs_start} and {obs_end}")

        columns = (
            apogee_drpdb.Visit.apogee_id.label("obj"), # TODO: Raise with Nidever
            apogee_drpdb.Visit.telescope,
            apogee_drpdb.Visit.fiberid.label("fiber"), # TODO: Raise with Nidever
            apogee_drpdb.Visit.plate,
            apogee_drpdb.Visit.field,
            apogee_drpdb.Visit.mjd,
            apogee_drpdb.Visit.apred_vers.label("apred"), # TODO: Raise with Nidever
            func.left(apogee_drpdb.Visit.file, 2).label("prefix")
        )
        q = session.query(*columns).distinct(*columns)
        q = q.filter(apogee_drpdb.Visit.mjd >= obs_start)\
             .filter(apogee_drpdb.Visit.mjd < obs_end)
        
        if self.query_filter_by_kwargs is not None:
            q = q.filter_by(**self.query_filter_by_kwargs)

        if self.limit is not None:
            q = q.limit(self.limit)

        log.debug(f"Found {q.count()} {release} {filetype} files between MJD {obs_start} and {obs_end}")

        common = dict(release=release, filetype=filetype)
        keys = [column.name for column in columns]
        for values in q.yield_per(1):
            yield { **common, **dict(zip(keys, values)) }
                


class BossSpecOperator(SDSSDataProductOperator):
    
    def __init__(
        self,
        *,
        limit = None,
        release = None,
        query_filter_by_kwargs = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.limit = limit
        self.release = release
        self.query_filter_by_kwargs = query_filter_by_kwargs
        return None


    def query_data_model_identifiers_from_database(self, context):

        """
        Query the SDSS-V database for BOSS spectra observed in the execution period.

        :param context:
            The DAG execution context.
        """

        filetype = "spec"
        release = self.release or self.infer_release(context)
                
        obs_start = parse_as_mjd(context["prev_ds"])
        obs_end = parse_as_mjd(context["ds"])

        columns = (
            catalogdb.SDSSVBossSpall.catalogid,
            catalogdb.SDSSVBossSpall.run2d,
            catalogdb.SDSSVBossSpall.plate,
            catalogdb.SDSSVBossSpall.mjd,
            catalogdb.SDSSVBossSpall.fiberid
        )
        q = session.query(*columns).distinct(*columns)
        q = q.filter(catalogdb.SDSSVBossSpall.mjd >= obs_start)\
             .filter(catalogdb.SDSSVBossSpall.mjd < obs_end)

        if self.query_filter_by_kwargs is not None:
            q = q.filter_by(**self.query_filter_by_kwargs)

        if self.limit is not None:
            q = q.limit(self.limit)

        log.debug(f"Found {q.count()} {release} {filetype} files between MJD {obs_start} and {obs_end}")

        common = dict(release=release, filetype=filetype)
        keys = [column.name for column in columns]
        for values in q.yield_per(1):
            yield { **common, **dict(zip(keys, values)) }



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

            # Monkey-patch BOSS Spec paths.
            try:
                path = tree.full(**instance.parameters)
            except:
                if instance.parameters["filetype"] == "spec":

                    from astra.utils import monkey_patch_get_boss_spec_path
                    path = monkey_patch_get_boss_spec_path(**instance.parameters)
            
                else:
                    raise

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


