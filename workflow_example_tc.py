
import pickle
from sqlalchemy import Column, Float

import astra
from astra.tasks.base import BaseTask
from astra.tasks.targets import (LocalTarget, DatabaseTarget)
from astra.tasks.io import ApStarFile
from astra.tasks.continuum import Sinusoidal


from astra_thecannon.tasks import TrainTheCannon, TestTheCannon


class ContinuumNormalize(Sinusoidal, ApStarFile):

    # Since we are using ApStarFiles, we only want to use the combined spectrum.
    spectrum_kwds = dict(data_slice=(slice(0, 1), slice(None)))

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))





class CannonResult(DatabaseTarget):
    results_schema = [
        Column("TEFF", Float()),
        Column("LOGG", Float()),
        Column("FE_H", Float())
    ]


class StellarParameters(TestTheCannon, ContinuumNormalize):

    def requires(self):
        return {
            "model": TrainTheCannon(**self.get_common_param_kwargs(TrainTheCannon)),
            "observation": ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
        }
    

    def output(self):
        return {
            "database": CannonResult(self),
            "extras": LocalTarget(f"{self.task_id}.pkl")
        }


    def run(self):

        model = self.read_model()        

        # Since there is some overhead to load The Cannon model, we can run this in batch mode.
        for task in self.get_batch_tasks():
            # Make sure observations match the model dispersion.
            flux, ivar = task.resample_observations(model.dispersion)
            labels, cov, metadata = model.test(
                flux,
                ivar,
                initialisations=task.N_initialisations,
                use_derivatives=task.use_derivatives
            )
            
            # Write the results to Astra's database.
            self.output()["database"].write(dict(zip(model.vectorizer.label_names, labels[0])))

            # Save the extra results to disk.
            with open(task.output()["extras"].path, "wb") as fp:
                pickle.dump((labels, cov, metadata), fp)




if __name__ == "__main__":
        
    # Do a couple of  stars
    file_params = dict(
        apred=("r12", "r12"),
        apstar=("stars", "stars"),
        telescope=("apo25m", "apo25m"),
        field=("000+14", "000+14"),
        prefix=("ap", "ap"),
        obj=("2M16505794-2118004", "2M16575794-2042048"),
        use_remote=True
    )

    additional_params = dict(
        order=2,
        label_names=("TEFF", "LOGG", "FE_H"),
        training_set_path="dr14-apogee-giant-training-set.pkl",
        continuum_regions_path="/home/ubuntu/data/sdss/astra-components/astra_thecannon/python/astra_thecannon/etc/continuum-regions.list",
    )

    params = {**file_params, **additional_params}

    task = StellarParameters(**params)

    astra.build(
        [task],
        local_scheduler=True,
        detailed_summary=True
    )
