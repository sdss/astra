import luigi
import os
import numpy as np
import pickle

from astra.utils import log
        
# TODO: use sdss_tree instead.
data_path = 'sandbox_apo25m_2257'


class BaseApStarTask(luigi.Task):

    apstar_version = luigi.IntParameter()
    starname = luigi.Parameter()




class ApStarFile(luigi.ExternalTask, BaseApStarTask):

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                data_path,
                f"apStar-r{self.apstar_version}-{self.starname}.fits"
            )
        )



class ContinuumNormalizeApStarBySinesAndCosines(BaseApStarTask):


    L = luigi.FloatParameter(default=1400)
    order = luigi.IntParameter(default=3)
    continuum_regions_path = luigi.Parameter()

    def requires(self):
        return ApStarFile(
            starname=self.starname,
            apstar_version=self.apstar_version
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                data_path,
                f"apStar-r{self.apstar_version}-{self.starname}-normalised.pkl",
            )
        )
    

    def run(self):

        from astropy.nddata import InverseVariance

        from python.astra.tools.spectrum import Spectrum1D
        from python.astra.utils.continuum.sines_and_cosines import normalize

        continuum_regions = np.loadtxt(self.continuum_regions_path)
        
        spectrum = Spectrum1D.read(self.input().path)

        # TODO: Return a Spectrum1D object and write that instead.
        normalized_flux, normalized_ivar, continuum, metadata = normalize(
            spectrum.wavelength.value,
            spectrum.flux.value,
            spectrum.uncertainty.quantity.value,
            continuum_regions=continuum_regions,
            L=self.L,
            order=self.order,
        )


        output_path = self.output().path
        with open(output_path, "wb") as fp:
            pickle.dump(
                (
                    spectrum.wavelength.value,
                    normalized_flux[0], # TODO: Don't just dom first spcetrum
                    normalized_ivar[0], # TODO: Don't just dom first spcetrum
                ), 
                fp, 
                -1
            )


class StellarParametersWithFerreFromApStar(BaseApStarTask):

    """ Use FERRE to estimate stellar parameters given some apStar normalized spectrum. """
    
    # Current FERRE version at time of writing is 4.6.6.
    ferre_version_major = luigi.IntParameter(default=4)
    ferre_version_minor = luigi.IntParameter(default=6)
    ferre_version_patch = luigi.IntParameter(default=6)

    synthfile_paths = luigi.Parameter()
    interpolation_order = luigi.IntParameter(default=1)

    error_algorithm_flag = luigi.IntParameter(default=0)
    optimization_algorithm_flag = luigi.IntParameter(default=1)
    wavelength_interpolation_flag = luigi.IntParameter(default=0)

    initial_teff = luigi.FloatParameter()
    initial_logg = luigi.FloatParameter()
    initial_m_h = luigi.FloatParameter()
    initial_alpha_m = luigi.FloatParameter()
    initial_n_m = luigi.FloatParameter()
    initial_c_m = luigi.FloatParameter()

    def requires(self):
        """ Requires a continuum-normalized apStar file. """
        return ContinuumNormalizeApStarBySinesAndCosines(
            self.apstar_version,
            self.starname
        )

    def output(self):
        # TODO: Consider FERRE versionining in path, synthfile, or "experiment" name.
        return luigi.LocalTarget(
            os.path.join(
                data_path,
                f"apStar-r{self.apstar_version}-{self.starname}-normalised-ferre.pkl",
            )
        )
    
    def run(self):

        from astra_ferre.core import ferre
        from astropy import units as u
        from astropy.nddata import InverseVariance
        from python.astra.tools.spectrum import Spectrum1D

        initial_parameters = [
            self.initial_teff,
            self.initial_logg,
            self.initial_m_h,
            self.initial_alpha_m,
            self.initial_n_m,
            self.initial_c_m
        ]
        frozen_parameters = None

        # alpha_m, n_m, c_m,

        control_kwds = dict(
            synthfile_paths=self.synthfile_paths,
            interpolation_order=self.interpolation_order,
            optimization_algorithm_flag=self.optimization_algorithm_flag,
            error_algorithm_flag=self.error_algorithm_flag,
            wavelength_interpolation_flag=self.wavelength_interpolation_flag,
            use_direct_access=False
        )
        
        with open(self.input().path, "rb") as fp:
            wavelength, normalized_flux, normalized_ivar = pickle.load(fp)
        
        unit = u.Unit("1e-17 erg / (Angstrom cm2 s)")
        spectrum = Spectrum1D(
            spectral_axis=wavelength * u.Angstrom,
            flux=normalized_flux * unit,
            uncertainty=InverseVariance(normalized_ivar * unit**-2),
        )

        print("Loading FERRE")


        result = ferre.fit(
            [spectrum],
            initial_parameters=initial_parameters,
            frozen_parameters=frozen_parameters,
            control_kwds=control_kwds
        )

        params, param_errs, model_flux, meta = result

        print(f"FERRE output:\n{meta['stdout']}")
        print(f"FERRE stderr:\n{meta['stdout']}")
        for k, v in params.items():
            print(f"Estimated {k}: {v}")
            



if __name__ == "__main__":

    from glob import glob
    starnames = [p.split("-r8-")[-1][:-5] for p in glob(os.path.join(data_path, "*.fits"))]

    #ContinuumNormalizeApStarBySinesAndCosines(apstar_version=8, starname=starnames[0]).run()

    """
    StellarParametersWithFerreFromApStar(
        apstar_version=8,
        starname=starnames[0],
    ).run()
    """

    other = dict(
        initial_teff=5000,
        initial_logg=2.0,
        initial_m_h=0.0,
        initial_alpha_m=0.0,
        initial_n_m=0.0,
        initial_c_m=0.0,
        synthfile_paths="/Users/arc/research/projects/astra_components/data/ferre/asGK_131216_lsfcombo5v6/p6_apsasGK_131216_lsfcombo5v6_w123.hdr"
    )

    #StellarParametersWithFerreFromApStar(apstar_version=8, starname=starnames[0], **other).run()
    #raise a
    luigi.build([
        StellarParametersWithFerreFromApStar(apstar_version=8, starname=starname, **other) for starname in starnames
    ])


