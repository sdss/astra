from datetime import timedelta
from getpass import getuser
from typing import OrderedDict
import re
import json
import numpy as np
from subprocess import call, Popen, PIPE
from airflow.exceptions import AirflowRescheduleException, AirflowSkipException
from airflow.models.baseoperator import BaseOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils import timezone
from peewee import fn
from astra.database.astradb import Task, TaskBundle, Bundle


from astra import log, config
from astra.utils import flatten, estimate_relative_cost
from astra.database.astradb import database, Bundle


def get_slurm_queue():
    """Get a list of jobs currently in the Slurm queue."""

    pattern = (
        "(?P<job_id>\d+)+\s+(?P<name>[-\w\d_\.]+)\s+(?P<user>[\w\d]+)\s+(?P<group>\w+)"
        "\s+(?P<account>[-\w]+)\s+(?P<partition>[-\w]+)\s+(?P<time_limit>[-\d\:]+)\s+"
        "(?P<time_left>[-\d\:]+)\s+(?P<status>\w*)\s+(?P<nodelist>[\w\d\(\)]+)"
    )
    popen_args = [
        "/uufs/chpc.utah.edu/sys/installdir/slurm/std/bin/squeue",
        "--account=sdss-np,notchpeak-gpu,sdss-np-fast",
        '--format="%14i %50j %10u %10g %13a %13P %11l %11L %2t %R"',
    ]

    process = Popen(
        popen_args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
    )
    output, error = process.communicate()

    # Parse the output.
    return [match.groupdict() for match in re.finditer(pattern, output)]


def partition(items, K, return_indices=False):
    """
    Partition items into K semi-equal groups.
    """
    groups = [[] for _ in range(K)]
    N = len(items)
    sorter = np.argsort(items)
    if return_indices:
        sizes = dict(zip(range(N), items))
        itemizer = list(np.arange(N)[sorter])
    else:
        itemizer = list(np.array(items)[sorter])

    while itemizer:
        if return_indices:
            group_index = np.argmin(
                [sum([sizes[idx] for idx in group]) for group in groups]
            )
        else:
            group_index = np.argmin(list(map(sum, groups)))
        groups[group_index].append(itemizer.pop(-1))

    return [group for group in groups if len(group) > 0]


class SlurmOperator(BaseOperator):

    template_fields = (
        "bundles",
        "slurm_kwargs",
        "min_bundles_per_slurm_job",
        "max_parallel_tasks_per_slurm_job",
    )

    def __init__(
        self,
        bundles=None,
        slurm_kwargs=None,
        get_uncommitted_queue=False,
        min_bundles_per_slurm_job=1,
        max_parallel_tasks_per_slurm_job=32,
        implicit_node_sharing=False,
        only_incomplete=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.bundles = bundles
        self.slurm_kwargs = slurm_kwargs or {}
        self.get_uncommitted_queue = get_uncommitted_queue
        self.min_bundles_per_slurm_job = min_bundles_per_slurm_job
        self.max_parallel_tasks_per_slurm_job = max_parallel_tasks_per_slurm_job
        self.implicit_node_sharing = implicit_node_sharing
        self.only_incomplete = only_incomplete

    def execute(self, context):

        # Load the bundles, which will be an int or list of ints.
        print(f"Loading bundles {self.bundles}")
        primary_keys = flatten(json.loads(self.bundles))
        bundles = Bundle.select().where(Bundle.id.in_(primary_keys))
        print(f"OK")

        # It's bad practice to import here, but the slurm package is
        # not easily installable outside of Utah, and is not a "must-have"
        # requirement.
        from slurm import queue
        from slurm.models import Job, Member
        from slurm.session import Client, Configuration

        key = None
        verbose = True
        if self.get_uncommitted_queue:
            # TODO: What's the difference between committed and submitted?
            status = "uncommitted"

            # Who am I?
            member = Member.query.filter(Member.username == getuser()).first()
            log.info(
                f"I am Slurm member {member} ({member.username} a.k.a '{member.firstname} {member.lastname}')"
            )
            log.info(
                f"Looking for {status} slurm job matching member_id={member.id}, and {self.slurm_kwargs}"
            )

            # See if there are any other jobs like this awaiting submission.
            job = Job.query.filter_by(
                member_id=member.id,
                status=Configuration.status.index(status),
                **self.slurm_kwargs,
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

        # Load balance the bundles.
        B = len(bundles)
        Q = len(q.client.job.all_tasks())

        Q_free = self.max_parallel_tasks_per_slurm_job - Q

        if 0 >= Q_free:
            # just run together.
            group_bundle_ids = [[bundle.id for bundle in bundles]]

        elif B > Q_free:
            """
            # Estimate the cost of each bundle.
            bundle_costs = (
                Bundle.select(
                        Bundle.id,
                        fn.COUNT(Bundle.id)
                    )
                    .join(TaskBundle)
                    .join(Task)
                    .where(Bundle.id.in_(primary_keys))
                    .group_by(Bundle.id)
                    .order_by(fn.COUNT(Bundle.id).desc())
                    .tuples()
            )
            """
            def estimate_relative_cost(bundle_id):
                return (
                    TaskBundle.select()
                    .join(Task)
                    .where(TaskBundle.bundle == bundle_id)
                    .count()
                )
            bundle_costs = np.array(
                [
                    [bundle_id, estimate_relative_cost(bundle_id)]
                    for bundle_id in primary_keys
                ]
            )
            for primary_key, bundle_cost in zip(primary_keys, bundle_costs):
                log.debug(f"Bundle {primary_key} cost: {bundle_cost}")

            # We need to distribute the bundles approximately evenly.
            # This is known as the 'Partition Problem'.
            # bundle_costs = np.array(bundle_costs)
            group_bundle_ids = []
            group_costs = []
            for indices in partition(bundle_costs.T[1], Q_free, return_indices=True):
                group_bundle_ids.append([bundle_costs[i, 0] for i in indices])
                group_costs.append(np.sum(bundle_costs[indices, 1]))

            log.debug(
                f"Total bundle cost: {np.sum(bundle_costs.T[1])} split across {Q_free} groups, with costs {group_costs} (max diff: {np.ptp(group_costs)})"
            )
            log.debug(f"Number per item: {list(map(len, group_bundle_ids))}")
            

        else:
            # Run all bundles in parallel.
            group_bundle_ids = [[bundle.id] for bundle in bundles]

        # Add executables for each bundle.
        options = ""
        if self.only_incomplete:
            options += "--only-incomplete "

        for group_bundle in group_bundle_ids:
            group_bundle_str = " ".join([f"{id:.0f}" for id in group_bundle])
            executable = f"astra execute bundles {options}{group_bundle_str}"
            q.append(executable)
            log.info(f"Added '{executable}' to queue {q}")

        # Update metadata for all the bundles.
        with database.atomic() as txn:
            for bundle in bundles:
                meta = (bundle.meta or dict()).copy()
                meta.update(slurm_kwargs=self.slurm_kwargs, slurm_job_key=q.key)
                bundle.meta = meta
                bundle.save()

        # Check if this should be submitted now.
        tasks = q.client.job.all_tasks()
        N = len(tasks)
        log.info(f"There are {N} items in queue {q}: {tasks}")
        if N == 0:
            raise AirflowSkipException("No tasks in queue")

        ppn = self.max_parallel_tasks_per_slurm_job
        if N >= self.min_bundles_per_slurm_job:
            if self.implicit_node_sharing:
                if "ppn" not in self.slurm_kwargs and "shared" not in self.slurm_kwargs:
                    log.info(
                        f"Using implicit node sharing behaviour for {self}. Setting ppn = {ppn} and shared = True"
                    )
                    q.client.job.ppn = ppn
                    q.client.job.shared = True
                    q.client.job.commit()
                else:
                    log.info(
                        f"Implicit node sharing behaviour is enabled, but not doing anything because ppn/shared keywords already set in slurm kwargs"
                    )
            else:
                """
                # Increse n_threads proportionally?
                n_threads = int(np.ceil(64 / N))

                q_tasks = (
                    Task.select()
                        .join(TaskBundle)
                        .join(Bundle)
                        .where(Bundle.id.in_(primary_keys))
                )
                for task in q_tasks:
                    if "n_threads" in task.parameters:
                        existing_n_threads = task.parameters.get("n_threads", None)
                        if existing_n_threads != n_threads:
                            log.debug(f"Increasing n_threads = {n_threads} on {task}")
                            parameters = task.parameters.copy()
                            parameters["n_threads"] = n_threads
                            task.parameters = parameters
                            n = task.save()
                            assert n == 1
                """

            log.info(f"Submitting queue {q}")
            q.commit(hard=True, submit=True)

        else:
            log.info(
                f"Not submitting queue {q} because {N} < {self.min_bundles_per_slurm_job}"
            )

        # meta["slurm_job_key"] = q.key
        bundle.meta["slurm_job_key"] = q.key
        bundle.save()

        return q.key


class SlurmQueueSensor(BaseSensorOperator):

    """Prevents overloading the Slurm queue with jobs."""

    def poke(self, context):
        # First check that we don't have too many jobs submitted/running already.
        try:
            max_concurrent_jobs = config["slurm"]["max_concurrent_jobs"]
        except:
            log.warning(f"No Astra configuration set for slurm.max_concurrent_jobs")
            return True
        else:
            me = getuser()
            jobs = [job for job in get_slurm_queue() if job["user"] == me]
            N = len(jobs)

            log.info(f"Found {N} jobs running or queued for user {me}: {jobs}")

            return N < max_concurrent_jobs


class SlurmSensor(BaseSensorOperator):

    template_fields = ("job_key",)

    def __init__(self, job_key, job_status="complete", **kwargs) -> None:
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
        log.info(
            f"Job {job} (from key {self.job_key}) is {statuses[job.status]} ({q.get_percent_complete()}% complete)"
        )

        complete = job.status >= index

        # If not complete, it may have timed out. Let's at least check we have *A* job running.
        """
        if not complete:
            q = get_slurm_queue()
            if len(q) == 0:
                log.warning(f"There are NO slurm jobs running. This job {self.job_key} must have timed out.")
                return True
        """
        # If complete, cat the log location.
        return complete
