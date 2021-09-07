import click
import sys

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("primary_keys", nargs=1, required=True)
@click.option("--processes", nargs=1, default=1, type=int, required=False)
@click.pass_context
def thepayne_che(context, primary_keys, processes, **kwargs):
    """
    Estimate stellar labels using ThePayne-Che
    """

    from astra.contrib.thepayne_che.operators import estimate_stellar_labels
    
    try:
        estimate_stellar_labels(
            primary_keys,
            processes=processes
        )
    except:
        from astra.utils import log
        log.exception(f"Exception occurred:")
        # Click always returns 0 exit code, even if an exception occurred.
        # See https://github.com/pallets/click/issues/747
        # Instead we will forcibly exit with a non-zero code if there is an exception.
        sys.exit(1)