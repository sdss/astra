
import json
from tempfile import mkstemp
from luigi import (Parameter, IntParameter, BoolParameter, WrapperTask)
from luigi.task_register import load_task
from astra.tasks.base import BaseTask


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

    return SlurmMixin


class SlurmMixin(BaseTask):
    use_slurm = BoolParameter(
        default=False, significant=False,
    )
    slurm_nodes = IntParameter(
        default=1, significant=False,
    )
    slurm_ppn = IntParameter(
        default=64, significant=False,
    )
    slurm_walltime = Parameter(
        default="24:00:00", significant=False,
    )
    slurm_alloc = Parameter(
        significant=False, default="sdss-np", # The SDSS-V cluster.
    )



class SlurmTask(WrapperTask):

    """ A wrapper task to execute a task through Slurm. """

    wrap_task_module = Parameter()
    wrap_task_family = Parameter()
    wrap_task_params_path = Parameter()

    def requires(self):

        with open(self.wrap_task_params_path, "r") as fp:
            task_params = json.load(fp)

        yield load_task(
            self.wrap_task_module,
            self.wrap_task_family,
            task_params
        )
    



def slurmify(func):
    """
    A decorator to execute `task.run()` commands through Slurm if the
    `task.use_slurm` parameter is True.
    """

    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "use_slurm", False):
            return func(self, *args, **kwargs)
        else:
            # Save the task params etc to disk.
            task_params = self.to_str_params()

            # Overwrite slurm parameter so we don't get into a circular loop.
            task_params["use_slurm"] = False

            # Write task parameters to a temporary file.
            _, task_params_path = mkstemp()
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

            kwds = dict(label=self.task_id)
            for key in ("ppn", "nodes", "walltime", "alloc"):
                sk = f"slurm_{key}"
                if hasattr(self, sk):
                    kwds[sk] = getattr(self, sk)

            queue = SlurmQueue(verbose=True)
            queue.create(**kwds)
            queue.append(cmd)
            queue.commit(hard=True, submit=True)
            log.info(f"Slurm job submitted with {queue.key}")
                
            # Wait for completion.
            assert self.complete()
            log.info(f"Executed {self} through Slurm")
            
            # Remove the task params path.
            os.unlink(task_params_path)
            return None
    return wrapper
