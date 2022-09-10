import click
import sys

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("primary_keys", nargs=1, required=True)
@click.option("--model-path", nargs=1, required=True)
@click.option("--dwave", default=10., show_default=True)
@click.option("--p", default=(1E-8, 1E-7), show_default=True)
@click.option("--q", default=0.7, show_default=True)
@click.option("--ivar_block", default=None, show_default=True)
@click.option("--eps", default=1E-19, show_default=True)
@click.option("--rsv_frac", default=2., show_default=True)
@click.option("--n_jobs", default=1, show_default=True)
@click.option("--verbose", default=5, show_default=True)
@click.pass_context
def slam(context, primary_keys, model_path, dwave, p, q, ivar_block, eps, rsv_frac, n_jobs, verbose, **kwargs):
    """
    Estimate stellar labels using APOGEENet II.
    """

    from astra.contrib.slam.operators import estimate_stellar_labels
    
    try:
        estimate_stellar_labels(
            primary_keys,
            model_path,
            dwave_slam=dwave,
            p_slam=p,
            q_slam=q,
            ivar_block_slam=None,
            eps_slam=eps,
            rsv_frac_slam=rsv_frac,
            n_jobs_slam=n_jobs,
            verbose_slam=verbose
        )
    except:
        from astra import log
        log.exception(f"Exception occurred:")
        # Click always returns 0 exit code, even if an exception occurred.
        # See https://github.com/pallets/click/issues/747
        # Instead we will forcibly exit with a non-zero code if there is an exception.
        sys.exit(1)
