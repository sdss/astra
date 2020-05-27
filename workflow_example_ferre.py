from astra.tasks.io import ApStarFile
from astra.tasks.continuum import Sinusoidal
from astra_ferre.tasks import Ferre


class ContinuumNormalize(Sinusoidal, ApStarFile):

    sum_axis = 0

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


class StellarParameters(Ferre, ApStarFile):
    
    def requires(self):
        return ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
    
    

if __name__ == "__main__":
        
    import luigi

    BUILD = False
    SINGLE_STAR = True

    if SINGLE_STAR:
        
        import matplotlib.pyplot as plt

        # Do single star.
        file_params = dict(
            use_remote=True, # Download remote paths if they don't exist.
            apred="r12",
            apstar="stars",
            telescope="apo25m",
            field="000+14",
            prefix="ap",
            obj="2M16505794-2118004",
        )

        #https://data.sdss.org/sas/dr16/apogee/spectro/redux/r12/stars/apo25m/000+14/apStar-r12-2M16544175-2148453.fits

        additional_params = dict(
            initial_teff=5000,
            initial_logg=4.0,
            initial_m_h=-1,
            initial_alpha_m=0.0,
            initial_n_m=0.0,
            initial_c_m=0.0,
            synthfile_paths="/Users/arc/research/projects/astra_components/data/ferre/asGK_131216_lsfcombo5v6/p6_apsasGK_131216_lsfcombo5v6_w123.hdr"
        )

        params = {**file_params, **additional_params}

        task = StellarParameters(**params)


        if not BUILD:
            # This will execute the task even if it has alredy been completed.
            # (But it requires that the tasks it depends on have been executed! Use luigi.build to control the workflow)
            spectrum, result = task.run()

            params, params_err, model_flux, meta = result

            fig, ax = plt.subplots()
            ax.plot(spectrum.wavelength, spectrum.flux[0], c='k')
            ax.plot(meta["dispersion"][0], model_flux[0], c='r')

            plt.show()

        else:
            # Use Luigi' Build functionality to run the pipeline (do this the first time you use this script).
            result = luigi.build(
                [task],
                local_scheduler=True,
                detailed_summary=True
            )

    else:
        # Do all stars.
        # (This will only run tasks if those tasks have not been run before, 
        #  because we are using luigi.build() instead of task.run())
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
        
