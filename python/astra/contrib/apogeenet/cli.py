
import click

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("model_path", nargs=1, required=True)
@click.argument("pk", nargs=1, required=True)
@click.option('--all-visits', is_flag=True)
@click.option("--num-uncertainty-draws", default=100, show_default=True)
@click.option("--large-error", default=1e10)
@click.pass_context
def apogeenet(context, model_path, pk, all_visits, num_uncertainty_draws, large_error, **kwargs):
    """
    Estimate stellar labels using APOGEENet.
    """

    from astra.contrib.apogeenet.operators import estimate_stellar_labels

    estimate_stellar_labels(
        pk,
        model_path,
        analyze_individual_visits=all_visits,
        num_uncertainty_draws=num_uncertainty_draws,
        large_error=large_error
    )
