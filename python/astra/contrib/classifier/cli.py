
import click

@click.group("classifier")
@click.pass_context
def classifier(context):
    """ Classify sources using a convolutional neural network """
    pass

@classifier.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("output_model_path", nargs=1, required=True)
@click.argument("network_factory", nargs=1, required=True)
@click.argument("training_spectra_path", nargs=1, required=True)
@click.argument("training_labels_path", nargs=1, required=True)
@click.argument("validation_spectra_path", nargs=1, required=True)
@click.argument("validation_labels_path", nargs=1, required=True)
@click.argument("test_spectra_path", nargs=1, required=True)
@click.argument("test_labels_path", nargs=1, required=True)
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
@click.argument("pk", nargs=1, required=True)
@click.argument("model_path", nargs=1, required=True)
@click.pass_context
def test(context, model_path, pk, **kwargs):
    """
    Classify some SDSS data products.
    """

    from astra.contrib.classifier.operators import classify
    return classify(pk, model_path)
