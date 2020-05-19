import luigi
from astra.tasks.base import BaseTask
from astra.tasks.io import ApStarFile
from astra.tasks.continuum import Sinusoidal
#from astra_ferre.tasks import Ferre
from astra_thepayne.tasks import Train, Test


class ContinuumNormalize(Sinusoidal, ApStarFile):

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))




class StellarParameters(Test, ApStarFile):

    def requires(self):
        return {
            "model": Train(**self.get_common_param_kwargs(Train)),
            "observations": [
                ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
            ]
        }

    def output(self):
        return luigi.LocalTarget("foo.pkl")
    
    

if __name__ == "__main__":
        
    # Do single star.
    file_params = dict(
        apred="r12",
        apstar="stars",
        telescope="apo25m",
        field="000+14",
        prefix="ap",
        obj="2M16505794-2118004",
    )

    additional_params = dict(
        n_steps=10,
        training_set_path="/Users/arc/research/projects/astra_components/data/the-payne/kurucz_training_spectra.npz"
    )

    params = {**file_params, **additional_params}

    workflow = StellarParameters(**params).run()

    raise a
    luigi.build(
        [
            StellarParameters(**params)
        ],
        local_scheduler=True,
        detailed_summary=True
    )

    raise a

    # Do all stars.
    import os
    import luigi
    from sdss_access import SDSSPath
    from glob import glob

    path = SDSSPath()
    dirname = os.path.dirname(path.full("apStar", **file_params))

    paths = glob(os.path.join(dirname, "*.fits"))

    luigi.build([
        StellarParameters(**{**path.extract("apStar", p), **additional_params}) for p in paths
    ])
    
