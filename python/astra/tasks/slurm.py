
from luigi import (Parameter, IntParameter, BoolParameter)
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