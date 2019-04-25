
import argparse

from astra.utils import log

class SetVerboseLogging(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        """ Enable verbose logging """
        log.set_level(0)
        log.debug("Verbose logging enabled")
        setattr(namespace, self.dest, values)


class ParseInputPaths(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        """ Parse input paths from a file if instructed. """

        if namespace.from_file:
            log.debug(f"Reading input paths from {values}")

            with open(values, "r") as fp:
                values = list(map(str.strip, fp.readlines()))

        else:
            values = [values]

        setattr(namespace, "input_paths", values)


class PrepareOutputDirectory(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        """ Create the output directory """
        log.debug(f"Creating output directory {values}")
        os.makedirs(values, exist_ok=True)
        setattr(namespace, self.des, values)


def component_parser(prog, description):
    r"""
    A common parser for Astra components.

    :param prog:
        The name of your program. 

    :param description:
        A description for your program.

    :returns:
        A :class:`argparse.ArgumentParser` with top-level (required) inputs by Astra, and some
        basic processing functionality. For example, this parser will enable verbose logging if
        specified, parse input paths either from a text file or a single path, and prepare the
        output directory.
    """

    parser = argparse.ArgumentParser(prog=prog, description=description)

    # Required arguments by Astra.
    parser.add_argument("-v", "--verbose", dest="verbose", action=SetVerboseLogging,
                        help="verbose logging", nargs=0)
    parser.add_argument("-i", "--from-file", action="store_true", default=False,
                        help="specifies that the INPUT_PATH is a text file that contains a list of "
                             "input paths that are separated by new lines")
    parser.add_argument("input_path", action=ParseInputPaths,
                        help="local path to a reduced data product, or a file that contains a list "
                             "of paths to reduced data products if the -i flag is used")
    parser.add_argument("output_dir",
                        help="directory for analysis outputs")

    return parser




if __name__ == "__main__":
    example = component_parser(prog="test", description="bar")
    example_args = example.parse_args()
