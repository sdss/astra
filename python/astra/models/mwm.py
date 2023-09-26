from peewee import (
    AutoField,
    FloatField,
    BooleanField,
    DateTimeField,
    BigIntegerField,
    IntegerField,
    TextField,
    ForeignKeyField,
    DeferredForeignKey,
    fn,
)
import numpy as np
import os
from astropy.io import fits
from astra import __version__
from astra.models.source import Source
from astra.models.fields import PixelArray
from astra.models.spectrum import Spectrum
from astra.models.apogee import ApogeeVisitSpectrum

class MwmApogeeVisitSpectrum(ApogeeVisitSpectrum):

    """An APOGEE visit spectrum resampled to a common wavelength array in the rest frame."""

    source = ForeignKeyField(
        Source, 
        # We want to allow for spectra to be unassociated with a source so that 
        # we can test with fake spectra, etc, but any pipeline should run their
        # own checks to make sure that spectra and sources are linked.
        null=True, 
        index=True,
        backref="mwm_apogee_visit_spectra",
    )

    #> Spectrum Identifiers
    spectrum_id = ForeignKeyField(
        Spectrum,
        index=True,
        lazy_load=False,
        primary_key=True,
        default=Spectrum.create
    )
    upstream_spectrum = ForeignKeyField(ApogeeVisitSpectrum, field=ApogeeVisitSpectrum.spectrum_id)    
    #TODO: rename to drp_spectrum_id



    wavelength = PixelArray(ext=1)
    flux = PixelArray(ext=2)
    ivar = PixelArray(ext=3)
    pixel_flags = PixelArray(ext=4)

    v_astra = TextField(default=__version__)

    # This is the velocity used to shift the spectrum to rest frame.
    v_shift = FloatField(null=True, index=True)

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "apred",
                    "mjd",
                    "plate",
                    "telescope",
                    "field",
                    "fiber",
                    "prefix",
                    "reduction",
                    "v_shift",
                ),
                True,
            ),
        )


    @property
    def path(self):
        k = 100
        folders = f"{(self.spectrum_id // k) % k:0>2.0f}/{self.spectrum_id % k:0>2.0f}"
        return f"$MWM_ASTRA/{self.v_astra}/apogee/visit/{self.apred}/{folders}/mwmApogeeVisit-{self.spectrum_id}.fits"


    @classmethod
    def get_v_shift(cls, spectrum, v_shift=None):
        if v_shift is None:
            if spectrum.v_rel is not None:
                v_shift = -spectrum.v_rel
            elif spectrum.xcorr_v_rel is not None:
                v_shift = -spectrum.xcorr_v_rel
            else:
                v_shift = None
        return v_shift
    

    @classmethod
    def get_from_apogee_visit_spectrum(cls, spectrum, v_shift=None, tol=0.01):
        v_shift = cls.get_v_shift(spectrum, v_shift)

        q = (
            cls
            .select()
            .where(
                (cls.upstream_spectrum_id == spectrum.spectrum_id)
            )        
        )
        if v_shift is None:
            q = q.where(cls.v_shift.is_null())
        else:
            q = (
                q.where(
                    fn.ABS(cls.v_shift - v_shift) < tol
                )
            )

        return q.get()
        
    @classmethod
    def create_from_apogee_visit_spectrum(cls, spectrum, v_shift=None):
        v_shift = cls.get_v_shift(spectrum, v_shift)

        kwds = spectrum.__data__.copy()
        kwds.update(dict(
            spectrum_id=Spectrum.create().spectrum_id,
            upstream_spectrum_id=spectrum.spectrum_id,
            v_shift=v_shift
        ))

        spec = cls.create(**kwds)

        C_KM_S = 299792.458

        n_res = (5, 4.25, 3.5)

        new_wavelength = 10**(4.179 + 6e-6 * np.arange(8575))
        flux, ivar, pixel_flags = spectrum.resample(
            new_wavelength * (1 - (v_shift or 0.0) / C_KM_S),
            n_res,
        )
        
        path = spec.absolute_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Store flux, ivar, and pixel array
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=new_wavelength),
            fits.ImageHDU(data=flux),
            fits.ImageHDU(data=ivar),
            fits.ImageHDU(data=pixel_flags)
        ])
        hdul.writeto(path)        
        return spec
    

    @classmethod
    def get_or_create_from_apogee_visit_spectrum(cls, spectrum, v_shift=None):
        """
        Create a `MwmApogeeVisitSpectrum` product from the given spectrum, which is usually an `ApogeeVisitSpectrum`.
        
        :param spectrum:
            The upstream spectrum.

        :param v_shift: [optional]
            A velocity shift to apply to place to rest frame. If `None` is given then it will default to the
            following resolution order:

            - `v_shift`: `-v_rel` if `v_rel` is not `None`
            - `v_shift`: `-xcorr_v_rel` if `xcorr_v_rel` is not `None`
            - `v_shift`: `None`
        """
        try:
            spec = cls.get_from_apogee_visit_spectrum(spectrum, v_shift)
        except:
            spec = cls.create_from_apogee_visit_spectrum(spectrum, v_shift)
            return (spec, True)
        else:
            return (spec, False)


        
