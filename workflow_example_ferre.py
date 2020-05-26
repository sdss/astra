

from astra.tasks.io import ApStarFile
from astra.tasks.continuum import Sinusoidal
from astra_ferre.tasks import Ferre


class ContinuumNormalize(Sinusoidal, ApStarFile):

    # We extend the ContinuumNormalize class with ApStarFile because this
    # continuum normalization task could be run with many kinds of inputs.
    sum_axis = 0

    def requires(self):
        return ApStarFile(**self.get_common_param_kwargs(ApStarFile))


class StellarParameters(Ferre, ApStarFile):
    
    # We extend the FerreStellarParameters class with the ApStarFile class because
    # FERRE could be run with an ApStar file, an ApVisit file, or something else!

    def requires(self):
        return ContinuumNormalize(**self.get_common_param_kwargs(ContinuumNormalize))
    




    

if __name__ == "__main__":
        
    SINGLE_STAR = True

    if SINGLE_STAR:
        
        import matplotlib.pyplot as plt

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
            initial_teff=5000,
            initial_logg=4.0,
            initial_m_h=-1,
            initial_alpha_m=0.0,
            initial_n_m=0.0,
            initial_c_m=0.0,
            synthfile_paths="/Users/arc/research/projects/astra_components/data/ferre/asGK_131216_lsfcombo5v6/p6_apsasGK_131216_lsfcombo5v6_w123.hdr"
        )

        params = {**file_params, **additional_params}

        spectrum, result = StellarParameters(**params).run()

        params, params_err, model_flux, meta = result

        fig, ax = plt.subplots()
        ax.plot(spectrum.wavelength, spectrum.flux[0], c='k')
        ax.plot(meta["dispersion"][0], model_flux[0], c='r')

        plt.show()


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
        
