import inspect
import os
import uuid
from time import sleep, time
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.compat.functools import cached_property
from airflow.models import BaseOperator
from airflow.hooks.subprocess import SubprocessHook
from airflow.utils.operator_helpers import context_to_airflow_vars
from datetime import datetime
from subprocess import CalledProcessError

from astra.database.utils import (create_task_instance, serialize_pks_to_path)
from astra.utils import log, get_scratch_dir
from astra.utils.slurm import cancel_slurm_job_given_name

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
    template_fields = ("bash_command_kwargs", )
    template_fields_renderers = {"bash_command_kwargs": "py"}

    def __init__(
        self,
        bash_command=None,
        bash_command_kwargs=None,
        spectrum_callback=None,
        spectrum_callback_kwargs=None,
        slurm_kwargs=None,
        _data_model_identifiers=None,
        **kwargs
    ):
        super(AstraOperator, self).__init__(**kwargs)
        self.bash_command = bash_command
        self.bash_command_kwargs = bash_command_kwargs
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

        if self.bash_command is None:
            raise RuntimeError("No bash_command specified")

        args = (context["dag"].dag_id, context["task"].task_id, context["run_id"])

        parameters = dict(
            bash_command=self.bash_command,
            bash_command_kwargs=self.bash_command_kwargs
        )
        instance = create_task_instance(*args, parameters)
        self.pks = instance.pk
        return None


    @property
    def bash_command_line_arguments(self):
        """
        Return the parameters supplied to this task as a string of command line arguments.
        """

        cla = []
        for key, value in (self.bash_command_kwargs or dict()).items():
            cla.append(f"--{key.replace('_', '-')} {value}")
        return " ".join(cla)


    def get_env(self, context):
        """Builds the set of environment variables to be exposed for the bash command."""
        env = os.environ.copy()

        airflow_context_vars = context_to_airflow_vars(context, in_env_var_format=True)
        env.update(airflow_context_vars)
        return env
        

    @cached_property
    def subprocess_hook(self):
        """Returns hook for running the bash command"""
        return SubprocessHook()


    @cached_property
    def path_to_serialized_pks(self):
        if type(self.pks) == int:
            # TODO: Think of a better way to handle this case where we are using an AstraOperator,
            #       have only one primary key, and don't need any serialized path for pks to
            #       construct a bash command.
            return ""
        
        pks_path = serialize_pks_to_path(
            self.pks,
            dir=get_scratch_dir()
        )
        log.info(f"Serialized {len(self.pks)} primary keys to {pks_path}")
        return pks_path


    def execute(self, context):
        """
        Execute the operator.

        :param context:
            The Airflow DAG execution context.
        """

        # Use bash command. Maybe in Slurm.
        try:
            bash_command = f"{self.bash_command} {self.path_to_serialized_pks} {self.bash_command_line_arguments}"
        except:
            log.exception(f"Cannot construct bash command")
            raise
        else:
            log.info(f"Executing bash command:\n{bash_command}")

        if self.slurm_kwargs:
            self.execute_by_slurm(
                context,
                bash_command,
            )
        else:
            # TODO: 
            output_encoding = "utf-8"
            skip_exit_code = 99

            env = self.get_env(context)
            command = f"bash -c {bash_command}".split()

            result = self.subprocess_hook.run_command(
                command=command,
                env=env,
                output_encoding=output_encoding
            )
            if skip_exit_code is not None and result.exit_code == skip_exit_code:
                raise AirflowSkipException(f"Bash command returned exit code {skip_exit_code}. Skipping.")
            elif result.exit_code != 0:
                raise AirflowException(
                    f"Bash command failed. The command returned a non-zero exit code {result.exit_code}."
                )
            
        # Remove the temporary file.
        if self.path_to_serialized_pks:
            try:
                os.unlink(self.path_to_serialized_pks)
            except:
                log.warning(f"Unable to remove path to primary keys '{self.path_to_serialized_pks}'")

        return self.pks


    def on_kill(self) -> None:
        if self.slurm_kwargs:
            # Cancel the Slurm job.
            cancel_slurm_job_given_name(self._slurm_label)
        else:
            # Kill the subprocess.
            self.subprocess_hook.send_sigterm()

        os.unlink(self.path_to_serialized_pks)
        return None
        

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
        


