
import click
import os
import multiprocessing as mp
import subprocess
import threading
from tempfile import mktemp
from astra import log
from astra.db.connection import session
from astra.db.models import Component
from astra.core.component import _get_likely_component
from astra.core import subset, task





def _non_blocking_pipe_read(pipe, queue):
    f""" A non-blocking and non-destructive pipe reader for long-running interactive jobs. """
    thread = threading.currentThread()
    while getattr(thread, "needed", True):
        queue.put(pipe.readline())


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("component", nargs=1, required=True)
@click.option("-i", "--from-file", is_flag=True,
              help="specifies that the INPUT_PATH is a text file that contains a list of input "
                   "paths that are separated by new lines")
@click.argument("input_path", nargs=1, required=True)
@click.argument("output_dir", nargs=1, required=True)
@click.option("--timeout", nargs=1, default=3600,
              help="Time out in seconds before killing the task.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def execute(context, component, input_path, output_dir, from_file, timeout, args, **kwargs):
    r"""
    Execute a component on a data product.
    """
    #context["ignore_unknown_options"] = True

    # TODO: THIS SHOULD BE MOVED TO task.execute!

    log.debug("execute")

    this_component = _get_likely_component(component)

    log.info(f"Identified component: {this_component}")

    # Load as a module
    command = f"module load {this_component.product}/{this_component.version}"
    log.info(f"Using modules to load component:\n{command}")

    try:
        process = subprocess.Popen(command.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, encoding="utf-8")

    except (subprocess.CalledProcessError, FileNotFoundError):
        log.exception(f"Exception when loading {this_component} as module. Continuing anyways!")
        None

    else:
        stdout, stderr = process.communicate()
        log.info(f"Stdout:\n{stdout}\n\nStderr:\n{stderr}")


    # Parse the input paths.
    if from_file:
        log.info(f"Reading input paths from {input_path}")
        with open(input_path, "r") as fp:
            input_paths = list(map(str.strip, fp.readlines()))

    else:
        log.info(f"Assuming a single input path: {input_path}")
        input_paths = [input_path]

    # Create a subset.
    log.info(f"Creating subset with {len(input_paths)} entries")
    this_subset = subset.create_from_data_paths(input_paths, add_unrecognised=True)

    log.info(f"Subset: {this_subset}")

    # Create task.
    log.info(f"Creating task for {this_component} to run on {this_subset} with additional arguments"
             f" {args}")

    # TODO: How to deal with default args and extra args? Overwrite them??

    this_task = task.create(this_component, this_subset, args=args, output_dir=output_dir)

    log.info(f"Created task: {this_task}")

    # Create a temporary file that has all the data paths in it, which we will pass to the command
    # line function.
    temp_data_paths = mktemp()
    with open(temp_data_paths, "w") as fp:
        fp.write("\n".join(input_paths))
    log.info(f"Created temporary file {temp_data_paths} for input data paths")

    # Execute the task
    command = [this_component.command, "-i", temp_data_paths, output_dir]

    if this_task.args is not None:
        if isinstance(this_task.args, str):
            command.extend(this_task.args.split())
        else:
            command.extend(this_task.args)

    log.info(f"Executing command:\n{command}")

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, encoding="utf-8", bufsize=1)

    except subprocess.CalledProcessError:
        log.exception(f"Exception raised when executing task")

        # Update task status.
        task.update(this_task, status="FAILED")

    else:
        try:
            stdout, stderr = process.communicate(timeout=timeout)

        except subprocess.TimeoutExpired:
            log.exception(f"Time out occurred ({timeout}) when executing task")

            # Update task status.
            task.update(this_task, status="TIMEOUT")

        else:
            log.info(f"Standard output:\n{stdout}")
            log.error(f"Standard error:\n{stderr}")

            log.info("Task completed")
            task.update(this_task, status="COMPLETE")


    return None
