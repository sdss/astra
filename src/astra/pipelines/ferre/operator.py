import os
import re
import numpy as np
import warnings
import json
from subprocess import Popen, PIPE
from glob import glob
from subprocess import check_output
from time import sleep, time
from tqdm import tqdm
from astra.utils import log, expand_path, flatten
from astra.utils.slurm import SlurmJob, SlurmTask, get_queue
from astra.pipelines.ferre.utils import parse_control_kwds, wc, read_ferre_headers, format_ferre_input_parameters, execute_ferre
from shutil import copyfile
from peewee import chunked

DEFAULT_SLURM_KWDS = dict(
    account="sdss-np", 
    partition="sdss-np", 
    walltime="24:00:00",
    mem=256_000, # needed to be able to do `srun`
)

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
    'sBA': np.array([-0.11225854,  0.91822257,  0.        ,             1]),
    'sgGK': np.array([2.11366749, 0.94762965, 0.0215653,                1]),
    'sgM': np.array([ 2.07628319,  0.83709908, -0.00974312,             1]),
    'sdF': np.array([ 2.36062644e+00, 7.88815732e-01, -1.47683420e-03,  4]),
    'sdGK': np.array([ 2.84815701,  0.88390584, -0.05891258,            5]),
    'sdM': np.array([ 2.83218967,  0.76324445, -0.05566659,             4]),
    
    # copied for turbospectrum #TODO
    'tBA': np.array([-0.11225854,  0.91822257,  0.        ,             1]),
    'tgGK': np.array([2.11366749, 0.94762965, 0.0215653 ,               1]),
    'tgM': np.array([ 2.07628319,  0.83709908, -0.00974312,             1]),
    'tdF': np.array([ 2.36062644e+00,  7.88815732e-01, -1.47683420e-03, 4]),
    'tdGK': np.array([ 2.84815701,  0.88390584, -0.05891258,            5]),
    'tdM': np.array([ 2.83218967,  0.76324445, -0.05566659,             4])    
}


def predict_ferre_core_time(grid, N, nov, pre_factor=1):
    """
    Predict the core-seconds required to analyze $N$ spectra with FERRE using the given grid.

    Note that these predictions do not include the grid load time (1-5 minutes).

    :param grid:
        The short-hand grid name (e.g., `sBA`, `sdF`, `sdM`).
    
    :param N:
        The number of spectra to analyze.
    
    :param nov:
        The number of free parameters at fitting time. In ASPCAP this value depends on the
        grid, and on which stage of analysis (e.g., coarse, parameter, or abundance stage).

        Below is a guide on what numbers you might want to use, depending on the grid and/or
        analysis stage.

        In the coarse stage, most grids use `nov=4`. No dimensions are frozen in the stellar
        parameter stage, so `nov` would be set to the number of dimensions in the grid. You
        can check this yourself, but as a guide you might expect:

        - `sBA`: 4 dimensions
        - `sgGK`: 6 dimensions
        - `sgM`: 6 dimensions
        - `sdF`: 7 dimensions
        - `sdGK`: 7 dimensions
        - `sdM`: 7 dimensions
    
    :pre_factor: [optional]
        An optional scaling term to use for time estimates. The true time can vary depending on
        which nodes FERRE is executed on, and which directories are used to read or write to.
        
    :returns:
        The estimated core-seconds needed to analyze the spectra.
    """
    intercept, N_coef, nov_coef, this_pre_factor = CORE_TIME_COEFFICIENTS[grid]
    return pre_factor * this_pre_factor * 10**(N_coef * np.log10(N) + nov_coef * np.log10(nov) + intercept)
    


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
    

def partition_items(items, K, return_indices=False):
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



def post_execution_interpolation(input_nml_path, pwd, n_threads=128, f_access=1, epsilon=0.001):
    """
    Run a single FERRE process to perform post-execution interpolation of model grids.
    """

    control_kwds = parse_control_kwds(input_nml_path)

    # parse synthfile headers to get the edges
    # TODO: so hacky. just give post_execution_interpolation an input nml path and a reference dir.
    
    synthfile = control_kwds["SYNTHFILE(1)"]
    for check in (synthfile, f"{pwd}/{synthfile}"):
        if os.path.exists(check):
            synthfile = check
            break
    headers = read_ferre_headers(synthfile)

    output_parameter_path = os.path.join(f"{pwd}/{control_kwds['OPFILE']}")
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

    relative_dir = os.path.relpath(os.path.dirname(input_nml_path), pwd)

    contents = f"""
    &LISTA
    SYNTHFILE(1) = '{control_kwds["SYNTHFILE(1)"]}'
    NOV = 0
    OFFILE = '{os.path.join(relative_dir, "model_flux.output")}'
    PFILE = '{os.path.join(relative_dir, os.path.basename(clipped_parameter_path))}'
    INTER = {control_kwds['INTER']}
    F_FORMAT = 1
    F_ACCESS = {f_access}
    F_SORT = 0
    NTHREADS = {n_threads}
    NDIM = {control_kwds['NDIM']}
    /
    """
    contents = "\n".join(map(str.strip, contents.split("\n"))).lstrip()
    new_input_nml_path = os.path.join(pwd, relative_dir, "post_execution_interpolation.nml")
    with open(new_input_nml_path, "w") as fp:
        fp.write(contents)
    
    execute_ferre(new_input_nml_path, pwd)

    os.system(f"vaffoff {os.path.join(pwd, relative_dir, 'parameter.output.clipped')} {os.path.join(pwd, relative_dir, 'model_flux.output')}")

    return None


def setup_ferre_for_re_execution():
    """
    Take an existing FERRE execution that has partial results and set it up for a new execution with only the things not yet finished.
    """
    raise NotImplementedError

    input_nml_path = process_args[-1]
    
    if input_nml_path.endswith(".nml"):
        restart_number, existing_suffix = (1, "")
    else:
        num = input_nml_path.split(".nml.")[1]
        existing_suffix = "." + num
        restart_number = 1 + int(num)

    if restart_number > max_attempts:
        click.echo(f"reached max max_attempts for {pwd}")
    else:
        # Just get the input/output names from this execution.
        input_names = np.loadtxt(f"{pwd}/parameter.input{existing_suffix}", usecols=(0, ), dtype=str)
        try:
            output_names = np.loadtxt(f"{pwd}/parameter.output{existing_suffix}", usecols=(0, ), dtype=str)
        except:
            output_names = []
        
        incomplete_names = np.setdiff1d(input_names, output_names)

        intersect, input_indices, incomplete_indices = np.intersect1d(input_names, incomplete_names, return_indices=True)

        # Only restart the process if the number of spectra is more than the number of threads
        with open(f"{pwd}/input.nml", "r") as fp:
            input_nml_contents = fp.readlines()
        
        nthreads = -1
        for line in input_nml_contents:
            if line.startswith("NTHREADS"):
                nthreads = int(line.split("=")[-1])
                break
            
        if len(input_indices) < nthreads:
            click.echo(f"not enough incomplete spectra to restart {pid} in {pwd}")

        else:                                
            # Randomize
            np.random.shuffle(input_indices)

            # Create a new parameter.input file, new flux.input file, and new e_flux.input file.
            for prefix in ("flux.input", "e_flux.input", "parameter.input"):
                with open(f"{pwd}/{prefix}{existing_suffix}", "r") as fp:
                    content = fp.readlines()
                    new_content = [content[index] for index in input_indices]
                    with open(f"{pwd}/{prefix}.{restart_number}", "w") as fp_new:
                        fp_new.write("".join(new_content))
                
            with open(f"{pwd}/input.nml", "r") as fp:
                new_content = []
                for line in fp.readlines():
                    if line.startswith(("PFILE", "ERFILE", "FFILE", "OPFILE", "OFFILE", "SFFILE")):
                        key, value = line.split(" = ")
                        value = value.strip("'\n")
                        new_content.append(f"{key} = '{value}.{restart_number}'\n")
                    else:
                        new_content.append(line)
                
                with open(f"{pwd}/input.nml.{restart_number}", "w") as fp_new:
                    fp_new.write("".join(new_content))                                

            click.echo(f"retrying process {pid} in {pwd} as restart number {restart_number}")

            # Re-start the process.
            proc = subprocess.Popen(
                ["ferre.x", f"input.nml.{restart_number}"],
                stdout=open(f"{pwd}/stdout.{restart_number}", "w"),
                stderr=open(f"{pwd}/stderr.{restart_number}", "w"),
                cwd=pwd
            )
            lookup_stub_by_pid[proc.pid] = stub
            states[stub]["pids"].append(proc.pid)


def load_balancer(
    stage_dir,
    job_name="ferre",
    input_nml_wildmask="*/input*.nml",
    slurm_kwds=None,
    post_interpolate_model_flux=True,
    partition=True,
    overwrite=True,
    n_threads=32, # 42
    max_nodes=0,
    max_tasks_per_node=4, # 3
    balance_threads=False,
    cpus_per_node=128,
    t_load_estimate=300, # 5 minutes est to load grid
    chaos_monkey=True,
    full_output=False,
    experimental_abundances=False
):
    stage_dir = expand_path(stage_dir)

    # Find all executable directories.
    input_nml_paths = glob(
        os.path.join(
        stage_dir,
        input_nml_wildmask
        )
    )
    return _load_balancer(
        stage_dir,
        input_nml_paths,
        job_name=job_name,
        slurm_kwds=slurm_kwds,
        post_interpolate_model_flux=post_interpolate_model_flux,
        partition=partition,
        overwrite=overwrite,
        n_threads=n_threads,
        max_nodes=max_nodes,
        max_tasks_per_node=max_tasks_per_node,
        balance_threads=balance_threads,
        cpus_per_node=cpus_per_node,
        t_load_estimate=t_load_estimate,
        chaos_monkey=chaos_monkey,
        full_output=full_output,
        experimental_abundances=experimental_abundances
    )
    

def _load_balancer(
    stage_dir,
    input_nml_paths,
    job_name="ferre",
    slurm_kwds=None,
    post_interpolate_model_flux=True,
    partition=True,
    overwrite=True,
    n_threads=32, # 42
    max_nodes=0,
    max_tasks_per_node=4, # 3
    balance_threads=False,
    cpus_per_node=128,
    t_load_estimate=300, # 5 minutes est to load grid
    chaos_monkey=True,
    full_output=False,
    experimental_abundances=False
):

    slurm_kwds = slurm_kwds or DEFAULT_SLURM_KWDS

    input_basename = "input.nml"
    stage_dir = expand_path(stage_dir)

    nodes = max_nodes if max_nodes > 0 else 1

    is_input_list = lambda p: os.path.basename(p).lower().startswith("input_list")

    input_paths, spectra, core_seconds = ([], [], [])
    for input_path in input_nml_paths:

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
            # Check whether it has FULL results.
            
            
            if partial_results:
                if overwrite:
                    log.warning(f"Partial results exist in {pwd}. Moving existing result files:")
                    for path in partial_result_paths:
                        os.rename(path, f"{path}.backup")
                        log.info(f"\t{path} -> {path}.backup")
                else:
                    log.warning(f"Skipping over {pwd} because partial results exist in: {', '.join(partial_result_paths)}")
                    continue
            
            if experimental_abundances:
                N = wc(f"{pwd}/{os.path.basename(control_kwds['PFILE'])}")
            else:
                N = wc(f"{pwd}/{control_kwds['PFILE']}")
            nov, synthfile = (control_kwds["NOV"], control_kwds["SYNTHFILE(1)"])
            grid = synthfile.split("/")[-2].split("_")[0]
            t = predict_ferre_core_time(grid, N, nov)
        
        input_paths.append(input_path)
        spectra.append(N)
        # Set the grid load time as a minimum estimate so that we don't get all small jobs partitioned to one node
        core_seconds.append(max(t, t_load_estimate))

    # Spread the approximate core-seconds over N nodes
    core_seconds = np.array(core_seconds)
    core_seconds_per_task = int(np.sum(core_seconds) / (nodes * max_tasks_per_node))

    # Limit nodes to the number of spectra
    tasks_needed = np.clip(np.round(core_seconds / core_seconds_per_task), 1, spectra).astype(int)

    # if things have n_nodes_per_process > 1, split them:
    total_spectra = np.sum(spectra)
    if total_spectra == 0:
        log.info(f"Found no spectra needing execution.")
        # TODO: I don't like this here, but it's the least bad hack I could think of right now.
        if os.getenv("AIRFLOW_CTX_DAG_OWNER", None) is not None:
            from airflow.exceptions import AirflowSkipException
            raise AirflowSkipException("No spectra to execute")
        log.info(f"Operator out.")
        return None

    log.info(f"Found {total_spectra} spectra total for {nodes} nodes ({core_seconds_per_task/60:.0f} min/task)")

    
    parent_partitions, partitioned_input_paths, partitioned_core_seconds = ({}, [], [])
    for n_tasks, input_path, n_spectra, n_core_seconds in zip(tasks_needed, input_paths, spectra, core_seconds):

        if not partition or n_tasks == 1 or is_input_list(input_path): # don't partition the abundances.. too complex
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
            
            actual_n_tasks = int(np.ceil(n_spectra / spectra_per_task))
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
    
    if balance_threads:
        longest_job_index = np.argmax(partitioned_core_seconds)
        fractional_core_seconds = partitioned_core_seconds / np.sum(partitioned_core_seconds)
        use_n_nodes = int(np.ceil(1/fractional_core_seconds[longest_job_index]))
        if max_nodes > 0:
            use_n_nodes = min(max_nodes, use_n_nodes)

        node_indices = partition_items(
            partitioned_core_seconds,
            use_n_nodes,
            return_indices=True
        )
        # Now sort each node chunk into tasks.
        chunks = []
        for node_index in node_indices:
            task_indices = partition_items(
                partitioned_core_seconds[node_index],
                max_tasks_per_node,
                return_indices=True
            )
            chunks.append([])
            for task_index in task_indices:
                chunks[-1].append(np.array(node_index)[task_index])

    else:
        chunks = chunked(
            partition_items(
                partitioned_core_seconds, 
                nodes * max_tasks_per_node, 
                return_indices=True,
            ),
            max_tasks_per_node
        )

    # For merging partitions afterwards
    partitioned_pwds = flatten(parent_partitions.values())
    output_basenames = ["stdout", "stderr", "rectified_model_flux.output", "parameter.output", "rectified_flux.output", "timing.csv"]
    if post_interpolate_model_flux:
        output_basenames.append("model_flux.output")

    slurm_jobs, executions, t_longest = ([], [], 0)
    for i, node_indices in enumerate(chunks, start=1):
        
        log.info(f"Node {i}: {len(node_indices)} tasks")
        
        # TODO: set the number of threads by proportion of how many spectra in this task compared to others on this node
        denominator = np.sum(partitioned_core_seconds[flatten(node_indices)])
        
        if balance_threads:
            t_tasks = np.array([np.sum(partitioned_core_seconds[ni]) for ni in node_indices])
            use_n_threads = np.ceil(np.clip(cpus_per_node * t_tasks / np.sum(t_tasks), 1, cpus_per_node)).astype(int)
        else:
            use_n_threads = [n_threads] * len(node_indices)

        expect_ferre_executions_in_these_pwds, slurm_tasks, est_times = ([], [], [])
        for j, (task_indices, n_threads) in enumerate(zip(node_indices, use_n_threads), start=1):

            #int(max(
            #    1,
            #    (cpus_per_node * np.sum(partitioned_core_seconds[st_indices]) / denominator)
            #))

            est_time = np.clip(np.sum(partitioned_core_seconds[task_indices]) / n_threads, t_load_estimate, np.inf)

            log.info(f"  {i}.{j}: {len(task_indices)} executions with {n_threads} threads. Estimated time is {est_time/60:.0f} min")

            task_commands = []
            for k, index in enumerate(task_indices, start=1):
                
                execution_commands = []

                input_path = partitioned_input_paths[index]
                cwd = os.path.dirname(input_path)
                    
                if is_input_list(input_path):
                    flags = "-l "
                    N = wc(f"{cwd}/flux.input")
                    with open(input_path, "r") as fp:
                        for rel_input_path in fp.readlines():
                            if balance_threads:
                                update_control_kwds(f"{cwd}/{rel_input_path}".rstrip(), "NTHREADS", n_threads)
                                
                            rel_pwd = os.path.dirname(f"{cwd}/{rel_input_path}")
                            executions.append([
                                f"{rel_pwd}/parameter.output",
                                N, 
                                0
                            ])
                else:
                    # if we are in the abundances stage, we should execute it from the parent directory to avoid duplicates of the flux/e_flux file
                    
                    flags = ""
                    executions.append([
                        f"{cwd}/parameter.output",
                        wc(f"{cwd}/parameter.input"),
                        0
                    ])

                    # Be sure to update the NTHREADS, just in case it was set at some dumb value (eg 1)
                    update_control_kwds(f"{cwd}/input.nml", "NTHREADS", n_threads)
                
                command = f"ferre.x {flags}{os.path.basename(input_path)}"
                expect_ferre_executions_in_these_pwds.append(cwd)
                
                if experimental_abundances:
                    execution_commands.append(f"cd {cwd}/../")
                    rel = "/".join(input_path.split("/")[-2:])
                    command = f"ferre.x {rel}"
                else:
                    execution_commands.append(f"cd {cwd}")
                    
                execution_commands.append(f"{command} > stdout 2> stderr")
                execution_commands.append(f"ferre_wait_for_clean_up .")
                execution_commands.append(f"ferre_timing . > timing.csv")
                
                if post_interpolate_model_flux:
                    execution_commands.append(f"ferre_interpolate_unnormalized_model_flux {cwd} > post_execute_stdout 2> post_execute_stderr")
                
                # If it's a partition result, then append the results to the parent directory
                if cwd in partitioned_pwds:
                    # the parent directory is probably the ../, but let's check for sure
                    for parent_dir, partition_dirs in parent_partitions.items():
                        if cwd in partition_dirs:
                            break
                    else:
                        raise RuntimeError(f"no parent dir found for {cwd}?")
                    
                    for basename in output_basenames:
                        execution_commands.append(f"cat {cwd}/{basename} >> {parent_dir}/{basename}")

                for command in execution_commands:
                    log.info(f"    {i}.{j}.{k}: {command}")
                task_commands.extend(execution_commands)
                            
            slurm_tasks.append(SlurmTask(task_commands))
            est_times.append(est_time)


        if chaos_monkey:
            command = f"ferre_chaos_monkey"
            command += f" > {stage_dir}/monkey_{i:0>2.0f}.out 2> {stage_dir}/monkey_{i:0>2.0f}.err"
            
            slurm_tasks.append(SlurmTask([command]))
            log.info(f"  {i}.{j+1}: 1 execution")
            log.info(f"    {i}.{j+1}.1: {command}")


        # Time for the node is the worst of individual tasks
        t_node = np.max(est_times)
        log.info(f"Node {i} estimated time is {t_node/60:.0f} min")
        if t_node > t_longest:
            t_longest = t_node
            

        slurm_job = SlurmJob(
            slurm_tasks,
            job_name,
            node_index=i,
            dir=stage_dir,
            **slurm_kwds
        )                
        slurm_job.write()
        slurm_jobs.append(slurm_job)


    t_longest_no_balance = (t_load_estimate + np.max(core_seconds) / 32) # TODO
    speedup = t_longest_no_balance/t_longest
    log.info(f"Estimated time without load balancing: {t_longest_no_balance/60:.0f} min")
    log.info(f"Estimated time for all jobs to complete: {t_longest/60:.0f} min (speedup {speedup:.0f}x), excluding wait time for jobs to start")

    # Submit the slurm jobs.
    if max_nodes == 0:
        log.warning(f"Not using Slurm job submission system because `max_nodes` is 0!")
        log.warning(f"Waiting 5 seconds before starting,...")
        sleep(5)

    log.info(f"Submitting jobs")
    job_ids = []
    for i, slurm_job in enumerate(slurm_jobs, start=1):
        slurm_path = slurm_job.write()
        if max_nodes == 0:
            pid = Popen(["sh", slurm_path])
            log.info(f"Started job {i} (process={pid}) at {slurm_path}")
            job_ids.append(pid)
        else:
            output = check_output(["sbatch", slurm_path]).decode("ascii")
            job_id = int(output.split()[3])
            log.info(f"Submitted slurm job {i} (jobid={job_id}) for {slurm_path}")
            job_ids.append(job_id)

    if not chaos_monkey:
        log.warning(f"FERRE chaos monkey not set up to run on this node. Please run it yourself.")            

    if full_output:
        return (tuple(job_ids), executions)
    else:
        return tuple(job_ids)



def monitor(
    job_ids,
    planned_executions,
    show_progress=True,
    refresh_interval=10,
    chaos_monkey=False,    
):

    # Check progress.
    log.info(f"Monitoring progress of the following FERRE execution directories:")

    n_spectra, executions = (0, [])
    for n_executions, e in enumerate(planned_executions, start=1):
        executions.append(e)
        n_spectra += e[1]
        log.info(f"\t{os.path.dirname(e[0])}")

    n_executions_done, last_updated = (0, time())
    desc = lambda n_executions_done, n_executions, last_updated: f"FERRE ({n_executions_done}/{n_executions}; {(time() - last_updated)/60:.0f} min ago)"

    tqdm_kwds = dict(
        desc=desc(n_executions_done, n_executions, last_updated),
        total=n_spectra, 
        unit=" spectra", 
        mininterval=refresh_interval
    )
    if not show_progress:
        tqdm_kwds.update(disable=True)

    # Get a parent folder from the executions so that we can put the FERRE chaos monkey logs somewhere.

    dead_processes, warn_on_dead_processes, chaos_monkey_processes = ([], [], {})
    with tqdm(**tqdm_kwds) as pb:
        while True:
            
            # FLY MY PRETTIES!!!1
            if chaos_monkey and len(chaos_monkey_processes) < len(job_ids):
                queue = get_queue()
                for job_id in set(job_ids).difference(chaos_monkey_processes):
                    try:
                        job = queue[job_id]                        
                    except:
                        log.warning(f"Job {job_id} does not appear in queue! Cannot launch FERRE chaos monkey.")
                    else:
                        if job["status"] == "R":
                            log.info(f"Slurm job {job_id} has started. Launching FERRE chaos monkey on job {job_id}")
                            # The `nohup` here *should* make sure the chaos monkey keeps running, even if the SSH
                            # connection between the operator and the Slurm job is disconnected. But we need to
                            # test this a bit more!
                            chaos_monkey_processes[job_id] = Popen(
                                ["srun", f"--jobid={job_id}", "nohup", "ferre_chaos_monkey"],
                                stdin=PIPE,
                                stdout=PIPE,
                                stderr=PIPE
                            )                        
            
            n_done_this_iteration = 0
            for i, (output_path, n_input, n_done) in enumerate(executions):
                if n_done >= n_input or output_path in dead_processes:
                    continue
                
                absolute_path = expand_path(output_path)
                if os.path.exists(absolute_path):
                    n_now_done = wc(absolute_path)
                else:
                    n_now_done = 0
                n_new = n_now_done - n_done
                n_done_this_iteration += n_new
                
                if n_new > 0:
                    last_updated = time()
                    pb.update(n_new)
                    executions[i][-1] = n_now_done
                    pb.set_description(desc(n_executions_done, n_executions, last_updated))
                    pb.refresh()
                
                else:
                    # Check for evidence of the chaos monkey
                    if os.path.exists(os.path.join(os.path.dirname(absolute_path), "killed")):
                        dead_processes.append(output_path)
                        n_executions_done += 1
                        last_updated = time()
                        pb.set_description(desc(n_executions_done, n_executions, last_updated))
                        pb.refresh()
                    else:
                        # Check stderr for segmentation faults.
                        stderr_path = os.path.join(os.path.dirname(absolute_path), "stderr")
                        try:
                            with open(stderr_path, "r") as fp:
                                stderr = fp.read()
                        except:
                            None
                        else:
                            # Don't die on deaths.. we expect the chaos monkey to restart it.
                            if ("error" in stderr or "Segmentation fault" in stderr):
                                warn_on_dead_processes.append(output_path)
                            '''
                                dead_processes.append(output_path)
                                n_executions_done += 1
                                last_updated = time()
                                pb.set_description(desc(n_executions_done, n_executions, last_updated))
                                pb.refresh()
                            '''

                if n_now_done >= n_input:
                    last_updated = time()
                    n_executions_done += 1
                    pb.set_description(desc(n_executions_done, n_executions, last_updated))
                    pb.refresh()
                                
            if n_executions_done == n_executions:
                pb.refresh()
                break

            # If FERRE is being executed by Slurm jobs, then they should already be in the queue before we start.
            if n_done_this_iteration == 0 and job_ids:
                jobs_in_queue = set(job_ids).intersection(get_queue())
                if len(jobs_in_queue) == 0:
                    log.warning(f"No Slurm jobs left in queue ({job_ids}). Finishing.")
                    break                

            sleep(refresh_interval)

    if warn_on_dead_processes:
        log.warning(f"Segmentation faults or chaos monkey deaths detected in following executions (these may have been restarted):")
        for output_path in warn_on_dead_processes:
            log.warning(f"\t{os.path.dirname(output_path)}")

    if job_ids:# and pb.n >= n_spectra:
        try:
            int(job_ids[0])
        except:
            log.info("Waiting until processes are complete")
            while True:
                done = 0
                for proc in job_ids:
                    returncode = proc.poll()
                    if returncode is None:
                        continue
                    else:
                        done += 1
                if done == len(job_ids):
                    log.info(f"All processes are complete.")
                    break                        
                sleep(5)
            
        else:
            log.info(f"Checking that all Slurm jobs are complete")
            while True:
                queue = get_queue()
                for job_id in job_ids:
                    if job_id in queue:
                        log.info(f"\tJob {job_id} has status {queue[job_id]['status']}")
                        break
                else:
                    log.info("All Slurm jobs complete.")
                    break

                sleep(5)
        
    log.info(f"Operator out.")
    return None
    

class FerreOperator:
    def __init__(
        self,
        stage_dir,
        input_nml_wildmask="*/input*.nml",
        job_name="ferre",
        slurm_kwds=None,
        post_interpolate_model_flux=True,
        overwrite=True,
        n_threads=32,
        max_nodes=0,
        max_tasks_per_node=4,
        cpus_per_node=128,
        experimental_abundances=False,
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

        self.n_threads = int(n_threads)
        self.job_name = job_name
        self.stage_dir = stage_dir
        self.max_nodes = int(max_nodes)
        self.max_tasks_per_node = int(max_tasks_per_node)
        self.cpus_per_node = int(cpus_per_node)
        self.post_interpolate_model_flux = post_interpolate_model_flux
        self.overwrite = overwrite
        self.slurm_kwds = slurm_kwds or DEFAULT_SLURM_KWDS
        self.experimental_abundances = experimental_abundances
        self.input_nml_wildmask = input_nml_wildmask
        return None


    def execute(self, context=None):
        return load_balancer(
            self.stage_dir,
            slurm_kwds=self.slurm_kwds,
            job_name=self.job_name,
            input_nml_wildmask=self.input_nml_wildmask,
            post_interpolate_model_flux=self.post_interpolate_model_flux,
            overwrite=self.overwrite,
            n_threads=self.n_threads,
            max_nodes=self.max_nodes,
            partition= not self.experimental_abundances,
            max_tasks_per_node=self.max_tasks_per_node,
            cpus_per_node=self.cpus_per_node,
            experimental_abundances=self.experimental_abundances,
            full_output=True         
        )


class FerreMonitoringOperator:

    def __init__(
        self,
        job_ids,
        executions,
        show_progress=True,
        refresh_interval=10,
        chaos_monkey=False,
        **kwargs
    ):
        self.job_ids = job_ids
        self.executions = executions
        self.refresh_interval = refresh_interval
        self.show_progress = show_progress
        self.chaos_monkey = chaos_monkey
        return None

    def execute(self, context=None):

        return monitor(self.job_ids, self.executions, self.show_progress, self.refresh_interval, self.chaos_monkey)