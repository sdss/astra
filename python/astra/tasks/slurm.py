
import json
import os
from tempfile import mkstemp
from luigi import (Task, Parameter, IntParameter, BoolParameter, WrapperTask)
from luigi.task_register import load_task
from luigi.mock import MockTarget
from astra.tasks import BaseTask
from astra.utils import log
from time import (sleep, time)

def slurm_mixin_factory(task_namespace):
    
    class SlurmMixin(BaseTask):
        use_slurm = BoolParameter(
            default=False, significant=False,
            config_path=dict(section=task_namespace, name="use_slurm")
        )
        slurm_nodes = IntParameter(
            default=1, significant=False,
            config_path=dict(section=task_namespace, name="slurm_nodes")
        )
        slurm_ppn = IntParameter(
            default=64, significant=False,
            config_path=dict(section=task_namespace, name="slurm_ppn")
        )
        slurm_walltime = Parameter(
            default="24:00:00", significant=False,
            config_path=dict(section=task_namespace, name="slurm_walltime")        
        )
        slurm_alloc = Parameter(
            significant=False, default="sdss-np", # The SDSS-V cluster.
            config_path=dict(section=task_namespace, name="slurm_alloc")
        )
        slurm_partition = Parameter(
            significant=False, default="",
            config_path=dict(section=task_namespace, name="slurm_partition")
        )
        slurm_mem = Parameter(
            significant=False, default="",
            config_path=dict(section=task_namespace, name="slurm_mem")
        )
        slurm_gres = Parameter(
            significant=False, default="",
            config_path=dict(section=task_namespace, name="slurm_gres")
        )

    return SlurmMixin


class SlurmMixin(BaseTask):
    use_slurm = BoolParameter(default=False, significant=False)
    slurm_nodes = IntParameter(default=1, significant=False)
    slurm_ppn = IntParameter(default=64, significant=False)
    slurm_walltime = Parameter(default="24:00:00", significant=False)
    slurm_alloc = Parameter(significant=False, default="sdss-np") # The SDSS-V cluster.
    slurm_partition = Parameter(significant=False, default="")
    slurm_mem = Parameter(significant=False, default="")
    slurm_gres = Parameter(significant=False, default="") # resources.


class SlurmTask(Task):

    """ A wrapper task to execute a task through Slurm. """

    wrap_task_module = Parameter()
    wrap_task_family = Parameter()
    wrap_task_params_path = Parameter()

    def requires(self):

        with open(self.wrap_task_params_path, "r") as fp:
            task_params = json.load(fp)

        return load_task(
            self.wrap_task_module,
            self.wrap_task_family,
            task_params
        )


    def run(self):
        print(f"Running {self}")
        log.info(f"Running {self}")
        with self.output().open("w") as fp:
            fp.write("")
        log.info(f"Finished {self}")
        print(f"Finished {self}")
        

    def output(self):
        return MockTarget(self.task_id)

        



def slurmify(func):
    """
    A decorator to execute `task.run()` commands through Slurm if the
    `task.use_slurm` parameter is True.
    """

    def wrapper(self, *args, **kwargs):
        if not getattr(self, "use_slurm", False):
            return func(self, *args, **kwargs)
        else:
            # Save the task params etc to disk.
            task_params = self.to_str_params()

            # Overwrite slurm parameter so we don't get into a circular loop.
            task_params["use_slurm"] = False

            # Write task parameters to a temporary file
            # (Remember: the temporary file has to be accessible to Slurm systems)
            # TODO: Consider putting this in /scratch/
            cwd = os.getcwd()
            _, task_params_path = mkstemp(dir=cwd)
            with open(task_params_path, "w") as fp:
                json.dump(task_params, fp)

            # TODO: Do we always want to use a local scheduler when executing tasks through Slurm?
            #       (I think we do.)
            cmd = f"luigi --module {SlurmTask.__module__} SlurmTask"\
                  f" --wrap-task-module {self.task_module}"\
                  f" --wrap-task-family {self.task_family}"\
                  f" --wrap-task-params-path {task_params_path}"\
                  f" --local-scheduler"

            # Submit a slurm job to perform the SlurmTask.
            from slurm import queue as SlurmQueue

            kwds = dict(label=self.task_id.replace(".", "_"))
            for key in ("ppn", "nodes", "walltime", "alloc", "partition", "mem", "gres"):
                sk = f"slurm_{key}"
                if getattr(self, sk, None):
                    kwds[key] = getattr(self, sk)

            queue = SlurmQueue(verbose=True)
            queue.create(**kwds)
            #queue.append("module avail cuda")
            queue.append(cmd)

            queue.commit(hard=True, submit=True)
            log.info(f"Slurm job submitted with {queue.key} and keywords {kwds}")
            log.info(f"\tJob directory: {queue.job_dir}")
            log.info(f"\tThere are {self.get_batch_size()} objects to run in batch mode")
                
            # Wait for completion.
            stdout_path = os.path.join(queue.job_dir, f"{self.task_id.replace('.', '_')}_01.o")
            stderr_path = os.path.join(queue.job_dir, f"{self.task_id.replace('.', '_')}_01.e")    

            t_init, seek = (time(), None)
            while 100 > queue.get_percent_complete():

                sleep(60)

                t = time() - t_init

                if not os.path.exists(stderr_path) and not os.path.exists(stdout_path):
                    log.info(f"Waiting on job {queue.key} to start (elapsed: {t / 60:.0f} min)")

                else:
                    if seek is None:
                        log.info(f"Job {queue.key} has started (elapsed: {t / 60:.0f} min)")
                        seek = 0
                    
                    else:
                        log.info(f"Job {queue.key} is {queue.get_percent_complete()}% complete (elapsed: {t / 60:.0f} min)")
                        # TODO: Implement a custom "get_percent_complete() on the *task* and try that first."

                    # Supply newline keyword argument so that tqdm output does not appear as newlines.
                    with open(stderr_path, "r", newline="\n") as fp:
                        contents = fp.read()
                    
                    log.info(f"Contents of {stderr_path}:\n{contents}")
                    seek = len(contents)
            
                # TODO: Check for situation where slurm errors occurred in task and "get_percent_complete" never reaches 100%

            assert self.complete()
            log.info(f"Executed {self} through Slurm")
            
            # Remove the task params path.
            os.unlink(task_params_path)
            return None
    return wrapper
