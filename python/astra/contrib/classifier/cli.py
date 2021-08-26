
import click

@click.group("classifier")
@click.pass_context
def classifier(context):
    """ Classify sources using a convolutional neural network """
    pass

@classifier.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--output-model-path", nargs=1, required=True)
@click.option("--network-factory", nargs=1, required=True)
@click.option("--training-spectra-path", nargs=1, required=True)
@click.option("--training-labels-path", nargs=1, required=True)
@click.option("--validation-spectra-path", nargs=1, required=True)
@click.option("--validation-labels-path", nargs=1, required=True)
@click.option("--test-spectra-path", nargs=1, required=True)
@click.option("--test-labels-path", nargs=1, required=True)
@click.option("--learning-rate", default=1e-5, show_default=True)
@click.option("--weight-decay", default=1e-5, show_default=True)
@click.option("--num-epochs", default=200, show_default=True)
@click.option("--batch-size", default=100, show_default=True)
@click.pass_context
def train(context, **kwargs):
    """
    Train a classifier.
    """
    from astra.contrib.classifier.operators import train_model
    return train_model(**kwargs)


@classifier.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("primary_keys", nargs=1, required=True)
@click.option("--model-path", nargs=1, required=True)
@click.pass_context
def test(context, primary_keys, model_path, **kwargs):
    """
    Classify some SDSS data products.
    """

    from astra.contrib.classifier.operators import classify
    return classify(primary_keys, model_path)
