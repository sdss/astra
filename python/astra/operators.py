
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from airflow.models import BaseOperator
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
        self.slurm_kwargs = slurm_kwargs
        return None


    def execute(self, context):
        # Check whether we are in slurm or not.
        slurm_job_id = os.getenv("SLURM_JOB_ID")

        if slurm_job_id is not None:
            log.info(f"Executing Python callable in Slurm as job ID {slurm_job_id}")
            return super(SlurmPythonOperator, self).execute(context)

        else:
            # TODO: Consider making this a keyword argument.
            poke_interval = 60 # seconds

            cmd = f"airflow tasks run -f -l -i -I -A {context['dag'].dag_id} {context['task'].task_id} {context['execution_date'].strftime('%Y-%m-%d')}"
            log.info(f"Submitting Slurm job with command:\n{cmd}")
            
            label = ".".join([
                context["dag"].dag_id,
                context["task"].task_id,
                # run_id is None if triggered by command line
                context["run_id"] or f"{uuid.uuid1()}"
            ])

            # It's bad practice to import here, but the slurm package is
            # not easily installable outside of Utah, and is not a "must-have"
            # requirement. 
    
            from slurm import queue
            q = queue(verbose=True)
            q.create(label=label, **self.slurm_kwargs)
            q.append(cmd)
            q.commit(hard=True, submit=True)

            log.info(f"Slurm job submitted with {q.key} and keywords {self.slurm_kwargs}")
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
                    log.info(f"Waiting on job {q.key} to finish (elapsed: {t / 60:.0f} min; started: {(t_to_start - time())/60:.0f} min ago)")
            
            log.info(f"Job {q.key} in {q.job_dir} is complete after {(time() - t_init)/60:.0f} minutes.")
            return None
            