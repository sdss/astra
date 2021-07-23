
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from airflow.models import BaseOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from astra.utils import log
from time import sleep, time

class SlurmPythonOperator(PythonOperator):
    
    """
    Submits a Slurm job that will execute a Python callable

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:PythonOperator`

    :param python_callable: A reference to an object that is callable
    :type python_callable: python callable
    :param op_kwargs: a dictionary of keyword arguments that will get unpacked
        in your function
    :type op_kwargs: dict (templated)
    :param op_args: a list of positional arguments that will get unpacked when
        calling your callable
    :type op_args: list (templated)
    :param templates_dict: a dictionary where the values are templates that
        will get templated by the Airflow engine sometime between
        ``__init__`` and ``execute`` takes place and are made available
        in your callable's context after the template has been applied. (templated)
    :type templates_dict: dict[str]
    :param templates_exts: a list of file extensions to resolve while
        processing templated fields, for examples ``['.sql', '.hql']``
    :type templates_exts: list[str]
    :param slurm_kwargs: a dictionary of keyword arguments that will be submitted
        to the slurm queue
    :type slurm_kwargs: dict
    """

    ui_color = "#BDCC94"

    template_fields = ('templates_dict', 'op_args', 'op_kwargs', 'slurm_kwargs')
    template_fields_renderers = {
        "templates_dict": "json", 
        "op_args": "py", 
        "op_kwargs": "py",
        "slurm_kwargs": "py"
    }

    shallow_copy_attrs = (
        'python_callable',
        'op_kwargs',
        'slurm_kwargs'
    )

    def __init__(
        self,
        *,
        python_callable: Callable,
        op_args: Optional[List] = None,
        op_kwargs: Optional[Dict] = None,
        templates_dict: Optional[Dict] = None,
        templates_exts: Optional[List[str]] = None,
        slurm_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            python_callable=python_callable,
            op_args=op_args,
            op_kwargs=op_kwargs,
            templates_dict=templates_dict,
            templates_exts=templates_exts,
            **kwargs
        )
        self.op_kwargs = op_kwargs
        self.slurm_kwargs = slurm_kwargs
        return None


    def execute(self, context):
        # Check whether we are in slurm or not.
        slurm_job_id = os.getenv("SLURM_JOB_ID")

        print(f"SLURM JOB ID IS {slurm_job_id}")

        if slurm_job_id is not None:
            log.info(f"Executing Python callable in Slurm as job ID {slurm_job_id}")
            log.info(self.op_kwargs)
            print(f"EXECUTING PYTHON CALLABLE AS {slurm_job_id}")
            print(self.op_kwargs)
            return super(SlurmPythonOperator, self).execute(context)

        else:
            # TODO: Consider making this a keyword argument.
            poke_interval = 60 # seconds

            cmd = f"airflow tasks run -f -l -i -I -A {context['dag'].dag_id} {context['task'].task_id} {context['execution_date'].strftime('%Y-%m-%d')}"
            log.info(f"Submitting Slurm job with command:\n{cmd}")
            
            log.info(self.op_kwargs)

            uid = str(uuid.uuid4())[:10]
            label = ".".join([
                context["dag"].dag_id,
                context["task"].task_id,
                # run_id is None if triggered by command line
                uid
            ])

            # It's bad practice to import here, but the slurm package is
            # not easily installable outside of Utah, and is not a "must-have"
            # requirement. 
            slurm_kwds = (self.slurm_kwargs or dict())

    
            from slurm import queue
            q = queue(verbose=True)
            q.create(label=label, **slurm_kwds)
            q.append(cmd)
            q.commit(hard=True, submit=True)

            log.info(f"Slurm job submitted with {q.key} and keywords {slurm_kwds}")
            log.info(f"\tJob directory: {q.job_dir}")

            stdout_path = os.path.join(q.job_dir, f"{label}_01.o")
            stderr_path = os.path.join(q.job_dir, f"{label}_01.e")    

            # Now we wait until the Slurm job is complete.
            # TODO: Should we just return None and then have a Sensor?

            t_init, t_to_start = (time(), None)
            while 100 > q.get_percent_complete():

                sleep(poke_interval)

                t = time() - t_init

                if not os.path.exists(stderr_path) and not os.path.exists(stdout_path):
                    log.info(f"Waiting on job {q.key} to start (elapsed: {t / 60:.0f} min)")
                    t_to_start = time() - t_init

                else:
                    if t_to_start is None:
                        t_to_start = time() - t_init
                    log.info(f"Waiting on job {q.key} to finish (elapsed: {t / 60:.0f} min; started: {(t_to_start - time())/60:.0f} min ago)")
            
            log.info(f"Job {q.key} in {q.job_dir} is complete after {(time() - t_init)/60:.0f} minutes.")
            return None
            


class SlurmBashOperator(BashOperator):
    r"""
    Submit a Slurm job that will execute a Bash script, command or set of commands.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:BashOperator`

    If BaseOperator.do_xcom_push is True, the last line written to stdout
    will also be pushed to an XCom when the bash command completes

    :param bash_command: The command, set of commands or reference to a
        bash script (must be '.sh') to be executed. (templated)
    :type bash_command: str
    :param env: If env is not None, it must be a dict that defines the
        environment variables for the new process; these are used instead
        of inheriting the current process environment, which is the default
        behavior. (templated)
    :type env: dict
    :param output_encoding: Output encoding of bash command
    :type output_encoding: str
    :param skip_exit_code: If task exits with this exit code, leave the task
        in ``skipped`` state (default: 99). If set to ``None``, any non-zero
        exit code will be treated as a failure.
    :type skip_exit_code: int

 Airflow will evaluate the exit code of the bash command. In general, a non-zero exit code will result in
    task failure and zero will result in task success. Exit code ``99`` (or another set in ``skip_exit_code``)
    will throw an :class:`airflow.exceptions.AirflowSkipException`, which will leave the task in ``skipped``
    state. You can have all non-zero exit codes be treated as a failure by setting ``skip_exit_code=None``.

    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - Exit code
         - Behavior
       * - 0
         - success
       * - `skip_exit_code` (default: 99)
         - raise :class:`airflow.exceptions.AirflowSkipException`
       * - otherwise
         - raise :class:`airflow.exceptions.AirflowException`

    .. note::

        Airflow will not recognize a non-zero exit code unless the whole shell exit with a non-zero exit
        code.  This can be an issue if the non-zero exit arises from a sub-command.  The easiest way of
        addressing this is to prefix the command with ``set -e;``

        Example:
        .. code-block:: python

            bash_command = "set -e; python3 script.py '{{ next_execution_date }}'"

    .. note::

        Add a space after the script name when directly calling a ``.sh`` script with the
        ``bash_command`` argument -- for example ``bash_command="my_script.sh "``.  This
        is because Airflow tries to apply load this file and process it as a Jinja template to
        it ends with ``.sh``, which will likely not be what most users want.

    .. warning::

        Care should be taken with "user" input or when using Jinja templates in the
        ``bash_command``, as this bash operator does not perform any escaping or
        sanitization of the command.

        This applies mostly to using "dag_run" conf, as that can be submitted via
        users in the Web UI. Most of the default template variables are not at
        risk.

    For example, do **not** do this:

    .. code-block:: python

        bash_task = BashOperator(
            task_id="bash_task",
            bash_command='echo "Here is the message: \'{{ dag_run.conf["message"] if dag_run else "" }}\'"',
        )

    Instead, you should pass this via the ``env`` kwarg and use double-quotes
    inside the bash_command, as below:

    .. code-block:: python

        bash_task = BashOperator(
            task_id="bash_task",
            bash_command='echo "here is the message: \'$message\'"',
            env={'message': '{{ dag_run.conf["message"] if dag_run else "" }}'},
        )

    """

    template_fields = ('bash_command', 'env')
    template_fields_renderers = {'bash_command': 'bash', 'env': 'json'}
    template_ext = (
        '.sh',
        '.bash',
    )
    ui_color = '#f0ede4'

    def __init__(
        self,
        *,
        bash_command: str,
        slurm_kwargs: Optional = None,
        env: Optional[Dict[str, str]] = None,
        output_encoding: str = 'utf-8',
        skip_exit_code: int = 99,
        **kwargs,
    ) -> None:
        super().__init__(
            bash_command=bash_command,
            env=env,
            output_encoding=output_encoding,
            skip_exit_code=skip_exit_code,
            **kwargs
        )
        self.slurm_kwargs = slurm_kwargs


    def execute(self, context):

        # TODO: Consider making this a keyword argument.
        poke_interval = 60 # seconds

        log.info(f"Submitting Slurm job with command:\n{self.bash_command}")
        
        uid = str(uuid.uuid4())[:8]
        label = ".".join([
            context["dag"].dag_id,
            context["task"].task_id,
            # run_id is None if triggered by command line
            uid
        ])

        # It's bad practice to import here, but the slurm package is
        # not easily installable outside of Utah, and is not a "must-have"
        # requirement. 
        slurm_kwds = (self.slurm_kwargs or dict())


        from slurm import queue
        q = queue(verbose=False)
        q.create(label=label, **slurm_kwds)
        q.append(self.bash_command)
        q.commit(hard=True, submit=True)

        log.info(f"Slurm job submitted with {q.key} and keywords {slurm_kwds}")
        log.info(f"\tJob directory: {q.job_dir}")

        stdout_path = os.path.join(q.job_dir, f"{label}_01.o")
        stderr_path = os.path.join(q.job_dir, f"{label}_01.e")    

        # Now we wait until the Slurm job is complete.
        # TODO: Should we just return None and then have a Sensor?

        t_init, t_to_start = (time(), None)
        while 100 > q.get_percent_complete():

            sleep(poke_interval)

            t = time() - t_init

            if not os.path.exists(stderr_path) and not os.path.exists(stdout_path):
                log.info(f"Waiting on job {q.key} to start (elapsed: {t / 60:.0f} min)")
                t_to_start = time() - t_init

            else:
                if t_to_start is None:
                    t_to_start = time() - t_init
                log.info(f"Waiting on job {q.key} to finish (elapsed: {t / 60:.0f} min; started: {(t_to_start - time())/60:.0f} min ago)")
        
        log.info(f"Job {q.key} in {q.job_dir} is complete after {(time() - t_init)/60:.0f} minutes.")
        with open(stderr_path, "r", newline="\n") as fp:
            contents = fp.read()
        
        log.info(f"Contents of {stderr_path}:\n{contents}")

        return None
        
