
import click
import os
import subprocess
from astra import log
from astra.db.connection import session
from astra.db.models import Component
from astra.core.component import _get_likely_component
from astra.core import subset, task


# Thanks https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
class OptionEatAll(click.Option):

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


@click.command()
@click.argument("component", nargs=1, required=True)
@click.option("-i", "--from-file", is_flag=True,
              help="specifies that the INPUT_PATH is a text file that contains a list of input "
                   "paths that are separated by new lines")
@click.argument("input_path", nargs=1, required=True)
@click.argument("output_dir", nargs=1, required=True)
@click.option("--args", default=None, cls=OptionEatAll,
              help="Supply additional arguments to this component.")
@click.pass_context
def execute(context, component, input_path, output_dir, from_file, args, **kwargs):
    r"""
    Execute a component on a data product.
    """

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

    except subprocess.CalledProcessError:
        log.exception(f"Exception when loading {this_component} as module. Continuing anyways!")
        None

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

    # Execute the task.
    args = " ".join(args)
    command = [this_component.product, "-i", tempfile, output_dir, args]
    log.info(f"Executin command:\n{command}")
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, encoding="utf-8")

    except subprocess.CalledProcessError:
        log.exception(f"Exception raised when executing task")

        # Update task status.
        task.update(this_task, status="FAILED")


    else:
        log.info("Task completed")
        task.update(this_task, status="COMPLETE")

    return None
    