import inspect
import importlib
import os
import uuid
from time import sleep, time
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.compat.functools import cached_property
from airflow.models import BaseOperator
from astra.hooks.subprocess import SubprocessHook
from airflow.utils.operator_helpers import context_to_airflow_vars
from datetime import datetime
from subprocess import CalledProcessError

from astra.database.utils import (create_task_instance, serialize_pks_to_path)
from astra.utils import log, get_scratch_dir
from astra.utils.slurm import cancel_slurm_job_given_name

from astra.operators.utils import (string_to_callable, callable_to_string)


class AstraOperator(BaseOperator):
    """
    A base operator for performing work.
    
    :param spectrum_callback: 
        The name of a function to execute on spectra after they are loaded 
        (e.g., 'astra.tools.continuum.mean').
    :type spectrum_callback: str
    :param spectrum_callback_kwargs: 
        A dictionary of keyword arguments that will be submitted to the `spectrum_callback`.
    :param slurm_kwargs: 
        A dictionary of keyword arguments that will be submitted to the slurm queue.
    :type slurm_kwargs: dict
    :param _data_model_identifiers:
        Directly supply data model identifiers to this operator and *do not* find data model
        identifiers for the given DAG execution context. Only use this if you know what you
        are doing. Using this argument will override the default behaviour of the operators,
        and will make the operator only process the data that is given by `_data_model_identifiers`,
        regardless of the execution date. This means the operator will process the same set
        of data every time it is triggered! This argument should only be used for testing.
        An example use of this argument might be:

        >> _data_model_identifiers = [
        >>     {
        >>         "obj": "2M06343180+0546387", 
        >>         "telescope": "apo25m",
        >>         "apstar": "stars",
        >>         "healpix": "88460",
        >>         "apred": "daily",
        >>         "filetype": "ApStar", 
        >>         "release": "sdss5"
        >>     }
        >> ]

    :type _data_model_identifiers:
        An iterable that includes dictionaries that fully define a data model product, or a
        callable function that returns an iterable.
    """

    ui_color = "#CCCCCC"
    template_fields = ("parameters", "op_kwargs", )
    template_fields_renderers = {
        "parameters": "py",
        "op_kwargs": "py",
    }

    def __init__(
        self,
        parameters=None,
        python_callable=None,
        op_kwargs=None,
        spectrum_callback=None,
        spectrum_callback_kwargs=None,
        slurm_kwargs=None,
        _data_model_identifiers=None,
        **kwargs
    ):
        super(AstraOperator, self).__init__(**kwargs)
        self.parameters = parameters
        self.python_callable = python_callable
        self.op_kwargs = op_kwargs
        self.spectrum_callback = spectrum_callback
        self.spectrum_callback_kwargs = spectrum_callback_kwargs 
        self.slurm_kwargs = slurm_kwargs 
        self._data_model_identifiers = _data_model_identifiers


    def pre_execute(self, context):
        """
        Create a task instance for this execution.
        
        :param context:
            The Airflow context dictionary.
        """

        if self.python_callable is None:
            raise RuntimeError("No python_callable specified")
        
        args = (context["dag"].dag_id, context["task"].task_id, context["run_id"])

        # Get a string representation of the python callable to store in the database.
        parameters = dict(python_callable=callable_to_string(self.python_callable))
        parameters.update(self.parameters)
        instance = create_task_instance(*args, parameters)
        self.pks = instance.pk
        return None


    def execute(self, context):
        """
        Execute the operator.

        :param context:
            The Airflow DAG execution context.
        """

        if self.slurm_kwargs:

            # Serialize the primary keys.
            if len(self.pks) > 1:
                primary_key_path = serialize_pks_to_path(self.pks, dir=get_scratch_dir())
                log.info(f"Serialized {len(self.pks)} primary keys to {primary_key_path}. First 10 primary keys are {self.pks[:10]}")
                
                # Store the primary key path, because we will clean up later.
                self._primary_key_path = primary_key_path

                bash_command = f"astra execute {primary_key_path}"
            else:   
                bash_command = f"astra execute {self.pks[0]}"

            self.execute_by_slurm(
                context,
                bash_command
            )
        
        else:
            # This is essentially what "astra execute [PK]" does.
            function = string_to_callable(self.python_callable)
            
            result = function(self.pks, **self.op_kwargs)
            log.info(f"Result from {function} with op kwargs {self.op_kwargs} was: {result}")
        
        return self.pks


    def post_execute(self, **kwargs):
        """
        Clean up after execution.
        """
        self._unlink_primary_key_path()
        return None


    def _unlink_primary_key_path(self):
        try:
            primary_key_path = self._primary_key_path
        except AttributeError:
            None
        else:
            log.info(f"Removing temporary file at {primary_key_path}")
            os.unlink(primary_key_path)
        return None


    def on_kill(self) -> None:
        if self.slurm_kwargs:
            # Cancel the Slurm job.
            try:
                cancel_slurm_job_given_name(self._slurm_label)
            except AttributeError:
                log.warning(f"Tried to cancel Slurm job but cannot find the Slurm label! Maybe the Slurm job wasn't submitted yet?")

        self._unlink_primary_key_path()
        return None
        

    def execute_by_slurm(
        self, 
        context, 
        bash_command,
        directory=None,
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
        if len(label) > 64:
            log.warning(f"Truncating Slurm label ({label}) to 64 characters: {label[:64]}")
            label = label[:64]

        self._slurm_label = label

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
        q.create(label=label, dir=directory, **slurm_kwargs)
        q.append(bash_command)
        try:
            q.commit(hard=True, submit=True)
        except CalledProcessError as e:
            log.exception(f"Exception occurred when committing Slurm job with output:\n{e.output}")
            raise

        log.info(f"Slurm job submitted with {q.key} and keywords {slurm_kwargs}")
        log.info(f"\tJob directory: {directory or q.job_dir}")

        stdout_path = os.path.join(directory or q.job_dir, f"{label}_01.o")
        stderr_path = os.path.join(directory or q.job_dir, f"{label}_01.e")    

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
        if "Error" in stdout.rstrip().split("\n")[-1] \
        or "Error" in stderr.rstrip().split("\n")[-1]:
            raise RuntimeError(f"detected exception at task end-point")

        # TODO: Get exit codes from squeue
        
        return None
        


