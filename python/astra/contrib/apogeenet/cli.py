import click

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pks", nargs=1, required=True)
@click.argument("model_path", nargs=1, required=True)
@click.option("--num-uncertainty-draws", default=100, show_default=True)
@click.option("--large-error", default=1e10)
@click.pass_context
def apogeenet(context, model_path, pks, num_uncertainty_draws, large_error, **kwargs):
    """
    Estimate stellar labels using APOGEENet II.
    """

    from astra.contrib.apogeenet.operators import estimate_stellar_labels
    
    estimate_stellar_labels(
        pks,
        model_path,
        num_uncertainty_draws=num_uncertainty_draws,
        large_error=large_error
    )
