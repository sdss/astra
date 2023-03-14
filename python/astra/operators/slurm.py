from datetime import timedelta
from getpass import getuser
from typing import OrderedDict
import re
import json
import inspect
import numpy as np
from tempfile import mkstemp
from subprocess import call, Popen, PIPE
from airflow.exceptions import AirflowRescheduleException, AirflowSkipException
from airflow.models.baseoperator import BaseOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils import timezone
from peewee import fn


from astra import config
from astra.base import _iterable_parameter_names, _get_or_create_database_table_for_task
from astra.utils import log, to_callable, flatten


def get_slurm_queue():
    """Get a list of jobs currently in the Slurm queue."""

    pattern = (
        "(?P<job_id>\d+)+\s+(?P<name>[-\w\d_\.]+)\s+(?P<user>[\w\d]+)\s+(?P<group>\w+)"
        "\s+(?P<account>[-\w]+)\s+(?P<partition>[-\w]+)\s+(?P<time_limit>[-\d\:]+)\s+"
        "(?P<time_left>[-\d\:]+)\s+(?P<status>\w*)\s+(?P<nodelist>[\w\d\(\)]+)"
    )
    popen_args = [
        #"/uufs/chpc.utah.edu/sys/installdir/slurm/std/bin/squeue",
        "/uufs/notchpeak.peaks/sys/installdir/slurm/std/bin/squeue",
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

def generate_dict_chunks(iterable_keys, iterable_values, N):
    M = len(iterable_values[0])
    K, m = divmod(M, N)
    for n in range(N):
        d = dict()
        for k, v in zip(iterable_keys, iterable_values):
            si = n * K
            ei = None if n == N - 1 else (n + 1) * K
            d[k] = v[si:ei]
        yield d





class SlurmTaskOperator(BaseOperator):

    template_fields = ("task_kwargs", )

    def __init__(self, task_callable=None, task_kwargs=None, slurm_kwargs=None, num_slurm_tasks=1, mkstemp_kwargs=None, continue_from_previous=False, **kwargs):
        super(SlurmTaskOperator, self).__init__(**kwargs)
        self.task_callable = task_callable
        self.mkstemp_kwargs = mkstemp_kwargs or {}
        self.task_kwargs = task_kwargs or {}
        self.slurm_kwargs = slurm_kwargs or {}
        self.num_slurm_tasks = num_slurm_tasks
        self.continue_from_previous = continue_from_previous
        return None

    def execute(self, context):
        from slurm import queue

        if isinstance(self.task_callable, str):
            log.info(f"Task callable is string: {self.task_callable}")
            task_callable = to_callable(self.task_callable)
        else:
            task_callable = self.task_callable
            
        signature = inspect.signature(task_callable)
        iterable_parameter_names = _iterable_parameter_names(signature)

        serialized_task_callable = f"{task_callable.__module__}.{task_callable.__name__}"

        slurm_kwargs = self.slurm_kwargs
        for k in ("cpus", "ppn"):
            try:
                slurm_kwargs[k] = int(slurm_kwargs[k])
            except:
                None
        for k, v in slurm_kwargs.items():
            print(f"Slurm keyword {k}: {v} ({type(v)})")

        q = queue(verbose=True)
        q.create(label=task_callable.__name__, **slurm_kwargs)
        print(f"Created slurm queue with {slurm_kwargs}")


        task_kwargs = self.task_kwargs
        if self.continue_from_previous:
            print(f"Continuing from previous.. ")
            
            # It'll be either data product or source.
            # Need to know where we read out to.
            model = _get_or_create_database_table_for_task(task_callable)
            if "data_product" in task_kwargs:

                data_product_ids = task_kwargs["data_product"]
                if isinstance(data_product_ids, str):
                    data_product_ids = json.loads(data_product_ids)

                print(f"Found {len(data_product_ids)} upstream: {data_product_ids}")

                sq = (
                    model
                    .select(model.data_product_id)
                    .tuples()                
                )
                done = flatten(sq)
                print(f"Found {len(done)} done")
                remaining = list(set(data_product_ids) - set(done))
                print(f"There are {len(remaining)} remaining")
                task_kwargs["data_product"] = remaining

            elif "source" in task_kwargs:
                source_ids = task_kwargs["source"]
                if isinstance(source_ids, str):
                    source_ids = json.loads(source_ids)                

                # TODO: should match on all other task_kwargs too, but CBF
                log.warning("NOT MATCHING ON ALL THINGS!!!")
                sq = (
                    model
                    .select(model.source_id)
                    .tuples()
                )
                done = flatten(sq)
                print(f"Found {len(done)} done")
                remaining = list(set(source_ids) - set(done))
                print(f"There are {len(remaining)} remaining")

                task_kwargs["source"] = remaining

            else:
                raise ValueError("don't know what iterable")

            '''
            data_product = task_kwargs["data_product"]
            if isinstance(data_product, str):
                data_product = json.loads(data_product)

            index = data_product.index(self.continue_from_data_product)
            print(f"That's at index {index}")
            task_kwargs["data_product"] = data_product[1 + index:]
            '''

        if self.num_slurm_tasks == 1 or len(iterable_parameter_names) == 0:
            # Write to a JSON file. 
            # TODO: If task_kwargs is a SQL query, flatten it.
            content = {
                "task_callable": serialized_task_callable,
                "task_kwargs": task_kwargs,
            }
            _, path = mkstemp(**self.mkstemp_kwargs)
            with open(path, "w") as fp:
                json.dump(content, fp)
            
            log.info(f"Written to {path}")

            q.append(f"astra run {path}")

        else:
            # Break up the task_kwargs into num_slurm_tasks chunks.
            # First we need to find what parameters are iterable.
            common_kwargs = task_kwargs.copy()
            iterable_values = []
            for k in iterable_parameter_names:
                v = common_kwargs.pop(k)
                if isinstance(v, str):
                    v = json.loads(v)
                iterable_values.append(v)
            

            N = len(flatten(iterable_values[0]))

            # chunk up the list of items in iterable values into num_slurm_tasks chunks.

            for iterable_kwargs in generate_dict_chunks(iterable_parameter_names, iterable_values, self.num_slurm_tasks):
                kwargs = common_kwargs.copy()
                kwargs.update(iterable_kwargs)
                content = {
                    "task_callable": serialized_task_callable,
                    "task_kwargs": kwargs,
                    }
                _, path = mkstemp(**self.mkstemp_kwargs)
                with open(path, "w") as fp:
                    json.dump(content, fp)
                
                q.append(f"astra run {path}")

        q.commit(hard=True, submit=True)
        return q.key




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
        from astra.database.astradb import Bundle, database, Task, TaskBundle, Bundle, Status


        # Load the bundles, which will be an int or list of ints.
        print(f"Loading bundles {self.bundles}")
        primary_keys = flatten(json.loads(self.bundles))
        bundles = Bundle.select().distinct(Bundle).join(TaskBundle).join(Task).where(Bundle.id.in_(primary_keys))
        if self.only_incomplete:
            #bundles = bundles.where(Bundle.status_id != Status.get(description="completed").id)
            completed_id = Status.get(description="completed").id
            bundles = bundles.where((Task.status_id != completed_id) | (Bundle.status_id != completed_id))

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
            raise NotImplementedError
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

    def __init__(self, max_concurrent_jobs=None, **kwargs):
        super(SlurmQueueSensor, self).__init__(**kwargs)
        self.max_concurrent_jobs = max_concurrent_jobs
        return None

    def poke(self, context):
        # First check that we don't have too many jobs submitted/running already.
        if self.max_concurrent_jobs is None:
            try:
                max_concurrent_jobs = config["slurm"]["max_concurrent_jobs"]
            except:
                log.warning(f"No Astra configuration set for slurm.max_concurrent_jobs")
                return True
        else:
            max_concurrent_jobs = self.max_concurrent_jobs

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


