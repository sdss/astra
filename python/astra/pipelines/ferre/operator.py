import os
import re
import numpy as np
import warnings
from glob import glob
from subprocess import check_output
from time import sleep, time
from tqdm import tqdm
from astra.utils import log, expand_path, flatten
from astra.pipelines.ferre.utils import wc, read_ferre_headers, format_ferre_input_parameters, execute_ferre
from shutil import copyfile
from peewee import chunked

DEFAULT_SLURM_KWDS = dict(account="sdss-np", partition="sdss-np", time="24:00:00")

def parse_control_kwds(input_nml_path):
    with open(expand_path(input_nml_path), "r") as fp:
        contents = fp.read()
    matches = re.findall("(?P<key>[A-Z\(\)0-9]+)\s*=\s*['\"]?\s*(?P<value>.+)\s*['\"]?\s*$", contents, re.MULTILINE)
    control_kwds = {}
    for key, value in matches:
        value = value.rstrip('"\'')
        try:
            value = int(value)
        except:
            None
        finally:
            control_kwds[key] = value
    return control_kwds


def update_control_kwds(input_nml_path, key, value):
    path = expand_path(input_nml_path)
    with open(path, "r") as fp:
        contents = fp.read()
    si = contents.index(key.upper())
    ei = si + contents[si:].index("\n")
    new_contents = f"{contents[:si]}{key.upper()} = {value}{contents[ei:]}"
    with open(path, "w") as fp:
        fp.write(new_contents)


CORE_TIME_COEFFICIENTS = {
    'sBA': np.array([-0.11225854,  0.91822257,  0.        ]),
    'sgGK': np.array([2.11366749, 0.94762965, 0.0215653 ]),
    'sgM': np.array([ 2.07628319,  0.83709908, -0.00974312]),
    'sdF': np.array([ 2.36062644e+00,  7.88815732e-01, -1.47683420e-03]),
    'sdGK': np.array([ 2.84815701,  0.88390584, -0.05891258]),
    'sdM': np.array([ 2.83218967,  0.76324445, -0.05566659])
}

def predict_ferre_core_time(grid, N, nov, conservatism=2):
    intercept, N_coef, nov_coef = CORE_TIME_COEFFICIENTS[grid]
    return conservatism * 10**(N_coef * np.log10(N) + nov_coef * np.log10(nov) + intercept)
    


def has_partial_results(pwd, control_kwds=None):
    """
    Check whether a FERRE-executable directory has partial FERRE results.

    :returns:
        A two-length tuple containing:
        - a bool indicating whether partial results exist already;
        - a tuple of paths identified which indicate there are partial results.
    """

    pwd = expand_path(pwd)
    if control_kwds is None:
        control_kwds = parse_control_kwds(f"{pwd}/input.nml")
    check_basenames = [
        "stdout",
        "stderr",
        control_kwds["OPFILE"],
        control_kwds["OFFILE"],
        control_kwds["SFFILE"]
    ]
    
    paths = [f"{pwd}/{basename}" for basename in check_basenames]
    paths = tuple(filter(os.path.exists, paths))
    has_partial_results = len(paths) > 0

    return (has_partial_results, paths)
    

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



class SlurmTask:

    def __init__(self, commands):
        self.commands = commands
        return None
    
    def set_meta(self, directory, node_index, task_index):
        self.directory = directory
        self.task_index = task_index
        self.node_index = node_index or 1
        return None

    def write(self):
        contents = []
        for command in self.commands:
            if isinstance(command, (list, tuple)):
                command, pwd = command
                contents.append(f"cd {pwd}")
            contents.append(f"{command} > stdout 2> stderr")

        path = expand_path(f"{self.directory}/node{self.node_index:0>2.0f}_task{self.task_index:0>2.0f}.slurm")
        with open(path, "w") as fp:
            fp.write("\n".join(contents))
        return path
    

class SlurmJob:

    def __init__(self, tasks, job_name, account, partition, time, node_index=None):
        self.account = account
        self.partition = partition
        self.time = time
        self.job_name = job_name
        self.tasks = tasks
        self.node_index = node_index
        for j, task in enumerate(self.tasks, start=1):
            task.set_meta(self.directory, self.node_index or 1, j)
        return None

    @property
    def directory(self):
        return expand_path(f"$PBS/{self.job_name}")
    
    def write(self):
        if self.node_index is None:
            node_index = 1
            node_index_suffix = ""
        else:
            node_index = self.node_index
            node_index_suffix = f"_{self.node_index:0>2.0f}"

        contents = [
            "#!/bin/bash",
            f"#SBATCH --account={self.account}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks=64",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --job-name={self.job_name}{node_index_suffix}",
            f"# ------------------------------------------------------------------------------",
            "export CLUSTER=1"
        ]
        for task in self.tasks:
            contents.append(f"source {task.write()} &")
        contents.extend(["wait", "echo \"Done\""])

        node_path = expand_path(f"{self.directory}/node{node_index:0>2.0f}.slurm")
        with open(node_path, "w") as fp:
            fp.write("\n".join(contents))

        return node_path


def post_execution_interpolation(pwd, n_threads=128, f_access=1, epsilon=0.001):
    """
    Run a single FERRE process to perform post-execution interpolation of model grids.
    """

    input_path = expand_path(f"{pwd}/input.nml")
    control_kwds = parse_control_kwds(input_path)

    # parse synthfile headers to get the edges
    synthfile = control_kwds["SYNTHFILE(1)"]
    headers = read_ferre_headers(synthfile)

    output_parameter_path = os.path.join(f"{pwd}/{os.path.basename(control_kwds['OPFILE'])}")
    output_names = np.atleast_1d(np.loadtxt(output_parameter_path, usecols=(0, ), dtype=str))
    output_parameters = np.atleast_2d(np.loadtxt(output_parameter_path, usecols=range(1, 1 + int(control_kwds["NDIM"]))))

    clipped_parameters = np.clip(
        output_parameters, 
        headers[0]["LLIMITS"] + epsilon,
        headers[0]["ULIMITS"] - epsilon
    )

    clipped_parameter_path = f"{output_parameter_path}.clipped"
    with open(clipped_parameter_path, "w") as fp:
        for name, point in zip(output_names, clipped_parameters):
            fp.write(format_ferre_input_parameters(*point, name=name))

    contents = f"""
    &LISTA
    SYNTHFILE(1) = '{synthfile}'
    NOV = 0
    OFFILE = 'model_flux.output'
    PFILE = '{os.path.basename(clipped_parameter_path)}'
    INTER = {control_kwds['INTER']}
    F_FORMAT = 1
    F_ACCESS = {f_access}
    F_SORT = 0
    NTHREADS = {n_threads}
    NDIM = {control_kwds['NDIM']}
    /
    """
    contents = "\n".join(map(str.strip, contents.split("\n"))).lstrip()
    input_nml_path = f"{pwd}/post_execution_interpolation.nml"
    with open(input_nml_path, "w") as fp:
        fp.write(contents)
    
    execute_ferre(input_nml_path)
    return None


class FerreOperator:


    def __init__(
        self,
        stage_dir,
        input_nml_wildmask="*/input*.nml",
        job_name="ferre",
        slurm_kwds=None,
        post_interpolate_model_flux=True,
        overwrite=False,
        max_nodes=10,
        max_tasks_per_node=4,
        cpus_per_node=128,
        refresh_interval=1,
        show_progress=True,
    ):
        """
        :param stage_dir:
            The stage directory for this operator to execute within. For example, if the stage directory
            is `~/ferre/coarse`, then this operator will look for FERRE-executable folders following the
            mask `~/ferre/coarse/*/input.nml`

        :param post_interpolate_model_flux:
            Perform a second FERRE step to post-interpolate the model fluxes without any FERRE-applied continuum.
        
        :param max_tasks_per_node:
            The maximum number of FERRE tasks to submit per node.
        
        :param max_nodes:
            The maximum number of nodes to use.
        
        :param n_threads:
            The number of FERRE threads to use in standard load balancing. The actual number of threads
            requested per FERRE process might actually be changed by this operator in order to minimise
            the expected walltime for all jobs.

            For example, if you request `max_nodes` of 10 and you have 13 FERRE executions, where 4 of those
            are very small executions and the rest are very large, then this operator might send 4 of those
            small processes to one node, each with `n_threads` threads, and the other 9 processes to the
            other 9 nodes, where the number of threads requested will be adjusted to 32 * 4.
        """

        self.job_name = job_name
        self.stage_dir = stage_dir
        self.max_nodes = int(max_nodes)
        if self.max_nodes <= 0:
            raise NotImplementedError("must use slurm nodes for now")
        self.max_tasks_per_node = int(max_tasks_per_node)
        self.cpus_per_node = int(cpus_per_node)
        self.post_interpolate_model_flux = post_interpolate_model_flux
        self.overwrite = overwrite
        self.slurm_kwds = slurm_kwds or DEFAULT_SLURM_KWDS
        self.refresh_interval = refresh_interval
        self.show_progress = show_progress
        self.input_nml_wildmask = input_nml_wildmask
        return None


    def execute(self, context=None):

        t_load_estimate = 5 * 60 # 5 minutes est to load grid

        input_basename = "input.nml"

        # Find all executable directories.
        all_input_paths = glob(
            os.path.join(
            expand_path(self.stage_dir),
            self.input_nml_wildmask
            )
        )

        is_input_list = lambda p: os.path.basename(p).lower() == "input_list.nml"

        input_paths, spectra, core_seconds = ([], [], [])
        for input_path in all_input_paths:

            if is_input_list(input_path):
                warnings.warn(f"No clever load balancing implemented for the abundance stage yet!")
                log.info(f"Found list of executable FERRE input files: {input_path}")
                with open(input_path, "r") as fp:
                    content = fp.readlines()
                    A = len(content)
                    rel_input_path = content[0].strip()                    
                
                pwd = os.path.dirname(input_path)
                rel_input_path = os.path.join(pwd, rel_input_path)
                control_kwds = parse_control_kwds(rel_input_path)

                N = A * wc(f"{pwd}/{control_kwds['FFILE']}")
                nov, synthfile = (control_kwds["NOV"], control_kwds["SYNTHFILE(1)"])
                grid = synthfile.split("/")[-2].split("_")[0]
                t = predict_ferre_core_time(grid, N, nov)

            else:
                log.info(f"Found executable FERRE input file: {input_path}")    
                control_kwds = parse_control_kwds(input_path)

                # Check whether there are partial results
                pwd = os.path.dirname(input_path)
                partial_results, partial_result_paths = has_partial_results(pwd, control_kwds)
                if partial_results:
                    if self.overwrite:
                        log.info(f"Partial results exist in {pwd}. Moving existing result files:")
                        for path in partial_result_paths:
                            os.rename(path, f"{path}.backup")
                            log.info(f"\t{path} -> {path}.backup")
                    else:
                        log.info(f"Skipping over {pwd} because partial results exist in: {', '.join(partial_result_paths)}")
                        continue
                
                N = wc(f"{pwd}/{control_kwds['PFILE']}")
                nov, synthfile = (control_kwds["NOV"], control_kwds["SYNTHFILE(1)"])
                grid = synthfile.split("/")[-2].split("_")[0]
                t = predict_ferre_core_time(grid, N, nov)
            
            input_paths.append(input_path)
            spectra.append(N)
            core_seconds.append(t)

        # Spread the approximate core-seconds over N nodes
        core_seconds = np.array(core_seconds)
        core_seconds_per_task = int(np.sum(core_seconds) / (self.max_nodes * self.max_tasks_per_node))

        # Limit nodes to the number of spectra
        tasks_needed = np.clip(np.round(core_seconds / core_seconds_per_task), 1, spectra).astype(int)

        # if things have n_nodes_per_process > 1, split them:
        total_spectra = np.sum(spectra)
        if total_spectra == 0:
            log.info(f"Found no spectra needing execution.")
            log.info(f"Operator out.")
            return None

        log.info(f"Found {total_spectra} spectra total for {self.max_nodes} nodes ({core_seconds_per_task/60:.0f} min/task)")

        parent_partitions, partitioned_input_paths, partitioned_core_seconds = ({}, [], [])
        for n_tasks, input_path, n_spectra, n_core_seconds in zip(tasks_needed, input_paths, spectra, core_seconds):

            if n_tasks == 1 or is_input_list(input_path): # don't partition the abundances.. too complex
                log.info(f"Keeping FERRE job in {input_path} (with {n_spectra} spectra) as is")
                partitioned_input_paths.append(input_path)
                partitioned_core_seconds.append(n_core_seconds)

            else:
                # This is where we check if we can split by spectra, or just split by input nml files for abundances
                log.info(f"Partitioning FERRE job in {input_path} into {n_tasks} tasks")
                pwd = os.path.dirname(input_path)

                # Partition it.
                spectra_per_task = int(np.ceil(n_spectra / n_tasks))

                # Split up each of the files.
                basenames = ("flux.input", "e_flux.input", "parameter.input")
                for basename in basenames:
                    check_output(["split", "-l", f"{spectra_per_task}", "-d", f"{pwd}/{basename}", f"{pwd}/{basename}"])
                
                actual_n_tasks = int(n_spectra / spectra_per_task)
                for k in range(actual_n_tasks):
                    partitioned_pwd = os.path.join(f"{pwd}/partition_{k:0>2.0f}")
                    os.makedirs(partitioned_pwd, exist_ok=True)

                    # copy file
                    partitioned_input_path = f"{partitioned_pwd}/{input_basename}"
                    copyfile(f"{pwd}/{input_basename}", partitioned_input_path)

                    # move the relevant basename files
                    for basename in basenames:
                        os.rename(f"{pwd}/{basename}{k:0>2.0f}", f"{partitioned_pwd}/{basename}")

                    if k == (actual_n_tasks - 1):
                        f = (n_spectra % spectra_per_task) / n_spectra
                    else:
                        f = spectra_per_task / n_spectra
                    
                    parent_partitions.setdefault(pwd, [])
                    parent_partitions[pwd].append(partitioned_pwd)
                    partitioned_input_paths.append(partitioned_input_path)
                    partitioned_core_seconds.append(f * n_core_seconds)
        
        # Partition by tasks, but chunk by node.
        partitioned_core_seconds = np.array(partitioned_core_seconds)        
        
        chunks = chunked(
            partition(
                partitioned_core_seconds, 
                self.max_nodes * self.max_tasks_per_node, 
                return_indices=True,
            ),
            self.max_tasks_per_node
        )

        slurm_jobs, executions, t_longest = ([], [], 0)
        for i, node_indices in enumerate(chunks, start=1):
            
            log.info(f"Node {i}: {len(node_indices)} tasks")
            
            # TODO: set the number of threads by proportion of how many spectra in this task compared to others on this node
            denominator = np.sum(partitioned_core_seconds[flatten(node_indices)])
            
            slurm_tasks, est_times = ([], [])
            for j, task_indices in enumerate(node_indices, start=1):

                n_threads = 32 # TODO
                #int(max(
                #    1,
                #    (self.cpus_per_node * np.sum(partitioned_core_seconds[st_indices]) / denominator)
                #))
                est_time = np.sum(partitioned_core_seconds[task_indices]) / n_threads + t_load_estimate

                log.info(f"  {i}.{j}: {len(task_indices)} executions with {n_threads} threads. Estimated time is {est_time/60:.0f} min")

                task_commands = []
                for k, index in enumerate(task_indices, start=1):
                    input_path = partitioned_input_paths[index]
                    pwd = os.path.dirname(input_path)

                    if is_input_list(input_path):
                        flags = "-l "
                        N = wc(f"{pwd}/flux.input")
                        with open(input_path, "r") as fp:
                            for rel_input_path in fp.readlines():
                                rel_pwd = os.path.dirname(f"{pwd}/{rel_input_path}")
                                executions.append([
                                    f"{rel_pwd}/parameter.output",
                                    N, 
                                    0
                                ])
                    else:
                        flags = ""
                        executions.append([
                            f"{pwd}/parameter.output",
                            wc(f"{pwd}/parameter.input"),
                            0
                        ])
                        # Be sure to update the NTHREADS, just in case it was set at some dumb value (eg 1)
                        update_control_kwds(f"{pwd}/input.nml", "NTHREADS", n_threads)
                    
                    command = f"ferre.x {flags}{os.path.basename(input_path)}"
                    task_commands.append((command, pwd))
            
                    log.info(f"    {i}.{j}.{k}: cd {pwd} | {command}")
                    
                slurm_tasks.append(SlurmTask(task_commands))
                est_times.append(est_time)

            # Time for the node is the worst of individual tasks
            t_node = np.max(est_times)
            log.info(f"Node {i} estimated time is {t_node/60:.0f} min")
            if t_node > t_longest:
                t_longest = t_node

            slurm_job = SlurmJob(
                slurm_tasks,
                self.job_name,
                node_index=i,
                **self.slurm_kwds
            )                
            slurm_job.write()
            slurm_jobs.append(slurm_job)


        t_longest_no_balance = (t_load_estimate + np.max(core_seconds) / 32) # TODO
        speedup = t_longest_no_balance/t_longest
        log.info(f"Estimated time without load balancing: {t_longest_no_balance/60:.0f} min")
        log.info(f"Estimated time for all jobs to complete: {t_longest/60:.0f} min (speedup {speedup:.0f}x), excluding wait time for jobs to start")


        # Submit the slurm jobs.
        log.info(f"Submitting jobs")
        job_ids = {}
        for slurm_job in slurm_jobs:
            slurm_path = slurm_job.write()
            output = check_output(["sbatch", slurm_path]).decode("ascii")
            job_id = int(output.split()[3])
            job_ids[job_id] = slurm_path
            log.info(f"Slurm job {job_id} for {slurm_path}")

            
        # Check progress.
        log.info(f"Monitoring progress..")

        n_executions_done, n_executions, n_spectra, last_updated = (0, len(executions), np.sum(spectra), time())
        desc = lambda n_executions_done, n_executions, last_updated: f"FERRE ({n_executions_done}/{n_executions}; {(time() - last_updated)/60:.0f} min ago)"

        tqdm_kwds = dict(
            desc=desc(n_executions_done, n_executions, last_updated),
            total=n_spectra, 
            unit=" spectra", 
            mininterval=self.refresh_interval
        )
        if not self.show_progress:
            tqdm_kwds.update(disable=True)

        segmentation_fault_detected = []
        with tqdm(**tqdm_kwds) as pb:
            while True:
                for i, (output_path, n_input, n_done) in enumerate(executions):
                    if n_done >= n_input or output_path in segmentation_fault_detected:
                        continue
                    
                    if os.path.exists(output_path):
                        n_now_done = wc(output_path)
                    else:
                        n_now_done = 0
                    n_new = n_now_done - n_done
                    
                    if n_new > 0:
                        last_updated = time()
                        pb.update(n_new)
                        executions[i][-1] = n_now_done
                        pb.set_description(desc(n_executions_done, n_executions, last_updated))
                        pb.refresh()
                    
                    else:
                        # Check for stderr.
                        stderr_path = os.path.join(os.path.basename(output_path), "stderr")
                        try:
                            with open(stderr_path, "r") as fp:
                                stderr = fp.read()
                        except:
                            None
                        else:
                            if "error" in stderr or "Segmentation fault" in stderr:
                                segmentation_fault_detected.append(output_path)
                                n_executions_done += 1
                                last_updated = time()
                                pb.set_description(desc(n_executions_done, n_executions, last_updated))
                                pb.refresh()
                                if self.post_interpolate_model_flux:
                                    post_execution_interpolation(os.path.dirname(output_path))

                    if n_now_done >= n_input:
                        last_updated = time()
                        n_executions_done += 1
                        pb.set_description(desc(n_executions_done, n_executions, last_updated))
                        pb.refresh()
                        if self.post_interpolate_model_flux:
                            post_execution_interpolation(os.path.dirname(output_path))
                                    
                if n_executions_done == n_executions:
                    pb.refresh()
                    break
                
                sleep(self.refresh_interval)

        if segmentation_fault_detected:
            log.warning(f"Segmentation faults detected in following executions:")
            for output_path in segmentation_fault_detected:
                log.warning(f"\t{os.path.dirname(output_path)}")

        # Bring back the partitions into the parent folder
        basenames = ["stdout", "stderr", "rectified_model_flux.output", "parameter.output", "rectified_flux.output"]
        if self.post_interpolate_model_flux:
            basenames.append("model_flux.output")

        for parent_dir, partitioned_dirs in parent_partitions.items():
            log.info(f"Merging partitions under {parent_dir}")
            for basename in basenames:
                content = ""
                for path in sorted([f"{pd}/{basename}" for pd in partitioned_dirs]):
                    with open(path, "r") as fp:
                        content += fp.read()
                with open(f"{parent_dir}/{basename}", "w") as fp:
                    fp.write(content)                    
                log.info(f"\tMerged {basename} partitions")
        
                # TODO: Remove the partition directory unless it had a segmentation fault

        log.info(f"Operator out.")
        return None
    

        # prepare slurm jobs
        # - slurm jobs need to run a monitoring process too
        # - slurm jobs need to run a post-interpolation step if asked

        # monitoring:
        # -> status in queue
        # -> RAM + CPU usage of nodes that it is deployed on
        # -> how many FERRE processes running per node
        # -> which directories have anything in their stdout/stderr
        # -> number of output parameters by folder compared to expected
        # -> time-averaged rate of spectra analysed per minute, for each process, to monitor for timeouts
        # -> monitor any post-process step

        # TODO: spot partial executions, don't clobber, just do unfinished ones and use a merge script after?
