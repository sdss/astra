import pickle
import os
from tqdm import tqdm
from tempfile import mkstemp
from astra.utils import expand_path, log, callable
from astra.utils.slurm import SlurmJob, SlurmTask

try:
    from airflow.models.baseoperator import BaseOperator
except ImportError:
    log.warning(f"Cannot import `airflow`: this functionality will not be available")
    BaseOperator = object


class AstraTaskOperator(BaseOperator):


    def __init__(
        self,
        python_task_name,
        spectra_callable=None,
        task_kwargs=None,
        use_slurm=False,
        use_slurm_gpu=False,
        slurm_kwargs=None,
        mkstemp_kwargs=None,
        **kwargs
    ):
        super(AstraTaskOperator, self).__init__(**kwargs)
        self.python_task_name = python_task_name
        self.spectra_callable = spectra_callable
        self.task_kwargs = task_kwargs or {}
        self.use_slurm = use_slurm
        self.use_slurm_gpu = use_slurm_gpu 
        self.slurm_kwargs = slurm_kwargs or {}
        self.mkstemp_kwargs = mkstemp_kwargs or {}
        return None


    def get_slurm_kwargs(self):
        if self.use_slurm_gpu:
            slurm_kwargs = dict(
                account="notchpeak-gpu", # TODO: account needed?
                partition="notchpeak-gpu",
                nodes=1,
                mem=16000,
                walltime="24:00:00",
                gres="gpu:v100"
            )

        else:
            slurm_kwargs = dict(
                account="sdss-np", 
                partition="sdss-np", 
                walltime="24:00:00",
                ntasks=1
            )

        slurm_kwargs.update(self.slurm_kwargs)
        return slurm_kwargs


    def execute(self, context):
        if self.use_slurm or self.use_slurm_gpu:
            return self.execute_by_slurm(context)
        else:
            return self._execute(context)


    def _execute(self, context):

        log.info(f"Setting up")    

        kwds = self.task_kwargs.copy()
        kwds["spectra"] = self.spectra_callable(context)

        f = callable(self.python_task_name) 
        r = f(**kwds)

        log.info(f"Executing")
        for num, item in enumerate(r, start=1):
            log.debug(f"Result {num}: {item}")

        return num


    def execute_by_slurm(self, context):

        # Resolve the spectra, store IDs
        log.info(f"Getting spectra IDs")

        spectrum_ids = []
        if self.spectra_callable is not None:
            for spectrum in self.spectra_callable(context):
                spectrum_ids.append(spectrum.spectrum_id)
        spectrum_ids = tuple(spectrum_ids)

        # pickle the task kwargs and spectra (by their IDS)
        task_kwds = self.task_kwargs.copy()
        task_kwds["spectra"] = spectrum_ids

        slurm_kwds = self.get_slurm_kwargs()
        slurm_kwds.setdefault("job_name", self.python_task_name.split(".")[-1])
        job_name = slurm_kwds["job_name"]

        dir = expand_path(f"$PBS/{job_name}")
        os.makedirs(dir, exist_ok=True)

        mkstemp_kwds = dict(
            dir=dir,
            prefix="task_kwargs_",
            suffix=".pkl",
        )
        mkstemp_kwds.update(self.mkstemp_kwargs)

        _, path = mkstemp(**self.mkstemp_kwds)
        with open(path, "wb") as fp:
            pickle.dump(task_kwds, fp)
            
        log.info(f"Wrote kwds for {self.python_task_name} to {path}")
        
        # If executing by Slurm, ensure we use the same database path        
        commands = []
        astra_database_path = os.environ.get("ASTRA_DATABASE_PATH", None)
        if astra_database_path is not None:
            commands.append(f"export ASTRA_DATABASE_PATH={astra_database_path}")
        commands.append(f"astra execute {self.python_task_name} --kwargs-path {path} > {dir}/stdout 2> {dir}/stderr")
        
        job = SlurmJob(
            [
                SlurmTask(commands)
            ],
            **slurm_kwds
        )

        job_id = job.submit()
        log.info(f"Job ID: {job_id}")
        return job_id




if __name__ == "__main__":

    def spectra_callable(context):
        from astra.models.apogee import ApogeeVisitSpectrumInApStar
        return ApogeeVisitSpectrumInApStar.select().limit(10_000)


    from astra.operators import AstraTaskOperator

    op = AstraTaskOperator(
        task_id="test",
        python_task_name="astra.pipelines.apogeenet.apogeenet",
        spectra_callable=spectra_callable
    )

    op_slurm = AstraTaskOperator(
        task_id="test",
        python_task_name="astra.pipelines.apogeenet.apogeenet",
        spectra_callable=spectra_callable,
        use_slurm_gpu=True,
        slurm_kwargs=dict(
            job_name="apogeenet/2023-07-16",
            walltime="00:05:00"
        )
    ) 