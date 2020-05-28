import luigi
from astra.tasks.base import BaseTask
from astra.tasks.io import ApStarFile
from astra.tasks.continuum import Sinusoidal

from astra_thecannon.tasks import Train, Test


class ContinuumNormalize(Sinusoidal, ApStarFile):

    sum_axis = 0 # Stack multiple visits.

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


class StellarParameters(Test, ApStarFile):

    def requires(self):
        return dict(
            model=Train(**self.get_common_param_kwargs(Train)),
            observation=ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
        )




if __name__ == "__main__":
        
    # Do single star.
    file_params = dict(
        apred="r12",
        apstar="stars",
        telescope="apo25m",
        field="000+14",
        prefix="ap",
        obj="2M16505794-2118004",
        use_remote=True
    )

    additional_params = dict(
        order=2,
        label_names=("TEFF", "LOGG", "FE_H"),
        training_set_path="/Users/arc/research/projects/astra_components/data/the-cannon/dr14-apogee-giant-training-set.pkl",
    )

    params = {**file_params, **additional_params}

    
    task = StellarParameters(**params)

    luigi.build(
        [task],
        local_scheduler=True,
        detailed_summary=True
    )
