
import click

@click.group("thepayne")
@click.pass_context
def thepayne(context):
    """ Estimate stellar labels using a single layer neural network """
    pass

@thepayne.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("output_model_path", nargs=1, required=True)
@click.argument("training_set_path", nargs=1, required=True)
@click.option("--num-neurons", default=100, show_default=True)
@click.option("--num-epochs", default=100_000, show_default=True)
@click.option("--learning-rate", default=1e-3, show_default=True)
@click.option("--weight-decay", default=0.0, show_default=True)
@click.pass_context
def train(context, **kwargs):
    """
    Train a single layer neural network
    """
    from astra.contrib.thepayne.operators import train_model
    return train_model(**kwargs)


@thepayne.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pk", nargs=1, required=True)
@click.argument("model_path", nargs=1, required=True)
@click.pass_context
def test(context, model_path, pk, **kwargs):
    """
    Estimate stellar labels given a single layer neural network
    """

    from astra.contrib.thepayne.operators import estimate_stellar_labels
    return estimate_stellar_labels(
        pks=pk,
        model_path=model_path,
    )
