import click
import sys

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("primary_keys", nargs=1, required=True)
@click.option("--model-path", nargs=1, required=True)
@click.option("--num-uncertainty-draws", default=100, show_default=True)
@click.option("--large-error", default=1e10)
@click.pass_context
def apogeenet(context, primary_keys, model_path, num_uncertainty_draws, large_error, **kwargs):
    """
    Estimate stellar labels using APOGEENet II.
    """

    from astra.contrib.apogeenet.operators import estimate_stellar_labels
    
    try:
        estimate_stellar_labels(
            primary_keys,
            model_path,
            num_uncertainty_draws=num_uncertainty_draws,
            large_error=large_error
        )
    except:
        from astra.utils import log
        log.exception(f"Exception occurred:")
        # Click always returns 0 exit code, even if an exception occurred.
        # See https://github.com/pallets/click/issues/747
        # Instead we will forcibly exit with a non-zero code if there is an exception.
        sys.exit(1)