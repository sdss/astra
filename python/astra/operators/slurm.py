from getpass import getuser
from typing import OrderedDict
from airflow.models.baseoperator import BaseOperator
from airflow.sensors.base import BaseSensorOperator

from astra import log
from astra.database.astradb import Bundle

class SlurmOperator(BaseOperator):

    template_fields = ("bundle", "slurm_kwargs", "bundles_per_slurm_job")

    def __init__(
        self,
        bundle=None,
        slurm_kwargs=None,
        bundles_per_slurm_job=1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.bundle = bundle
        self.slurm_kwargs = slurm_kwargs or {}
        self.bundles_per_slurm_job = bundles_per_slurm_job

    
    def execute(self, context):

        bundle = self.bundle
        if not isinstance(bundle, Bundle):
            bundle = Bundle.get(pk=int(bundle))

        executable = f"astra execute --bundle {bundle.pk}"

        # Set Slurm kwargs for this bundle.
        meta = (bundle.meta or {}).copy()
        meta.update(slurm_kwargs=self.slurm_kwargs)

        # It's bad practice to import here, but the slurm package is
        # not easily installable outside of Utah, and is not a "must-have"
        # requirement. 
        from slurm import queue
        from slurm.models import Job, Member
        from slurm.session import Client, Configuration

        key = None
        verbose = True
        if self.bundles_per_slurm_job > 1:
            # TODO: What's the difference between committed and submitted?
            status = "uncommitted"

            # Who am I?
            member = Member.query.filter(Member.username == getuser()).first()
            log.info(f"I am Slurm member {member} ({member.username} a.k.a '{member.firstname} {member.lastname}')")
            log.info(f"Looking for {status} slurm job matching member_id={member.id}, and {self.slurm_kwargs}")
            
            # See if there are any other jobs like this awaiting submission.
            job = Job.query.filter_by(
                member_id=member.id,
                status=Configuration.status.index(status),
                **self.slurm_kwargs
            ).first()

            log.info(f"Job: {job}")

            if job is None:
                log.info(f"Creating new job")
            else:
                log.info(f"Found existing job {job} with key {job.key}")
                key = job.key
        
        # Get or create a queue.
        q = queue(key=key, verbose=verbose)
        if key is None:
            log.info(f"Creating queue with {self.slurm_kwargs}")
            q.create(**self.slurm_kwargs)
        
        q.append(executable)
        log.info(f"Added '{executable}' to queue {q}")

        # Check if this should be submitted now.
        tasks = q.client.job.all_tasks()
        N = len(tasks)
        log.info(f"There are {N} tasks in queue {q}: {tasks}")
        if N >= self.bundles_per_slurm_job:
            log.info(f"Submitting queue {q}")
            q.commit(hard=True, submit=True)
        
        else:
            log.info(f"Not submitting queue {q} because {N} < {self.bundles_per_slurm_job}")

        meta["slurm_job_key"] = q.key
        bundle.update(meta=meta).execute()

        return q.key



class SlurmSensor(BaseSensorOperator):

    template_fields = ("job_key", )

    def __init__(
        self,
        job_key,
        job_status="complete",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.job_key = job_key
        self.job_status = job_status


    def poke(self, context):

        from slurm import queue
        from slurm.models import Job
        from slurm.session import Configuration

        job = Job.query.filter(Job.key == self.job_key).first()
        if job is None:
            raise ValueError(f"No job with key {self.job_key}")

        statuses = Configuration.status
        index = statuses.index(self.job_status)
        
        q = queue(key=self.job_key)
        log.info(f"Job {job} (from key {self.job_key}) is {statuses[job.status]} ({q.get_percent_complete()}% complete)")
        
        return (job.status >= index)
