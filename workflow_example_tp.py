import luigi
from astra.tasks.base import BaseTask
from astra.tasks.io import ApStarFile
from astra.tasks.continuum import Sinusoidal
from astra_thepayne.tasks import Train, Test


class ContinuumNormalize(Sinusoidal, ApStarFile):

    sum_axis = 0 # Stack multiple visits.

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


class StellarParameters(Test, ApStarFile):

    def requires(self):
        return {
            "model": Train(**self.get_common_param_kwargs(Train)),
            "observation": ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
        }

    
    

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
        n_steps=1000,
        training_set_path="/Users/arc/research/projects/astra_components/data/the-payne/kurucz_data.pkl"
    )

    params = {**file_params, **additional_params}
    
    task = StellarParameters(**params)

    luigi.build(
        [task],
        local_scheduler=True,
        detailed_summary=True
    )
