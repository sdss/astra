import click

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pks", nargs=1, required=True)
@click.pass_context
def doppler(context, pks, **kwargs):
    """
    Estimate radial velocity using Doppler.
    """

    from astra.contrib.doppler.operators import estimate_radial_velocity

    # TODO: Allow arguments to be passed through to Doppler   
    estimate_radial_velocity(pks)
