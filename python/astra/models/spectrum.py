from peewee import AutoField
from astra.models.base import BaseModel
from astra.models.fields import BitField


class Spectrum(BaseModel):

    """ A one dimensional spectrum. """

    pk = AutoField()
    spectrum_type_flags = BitField(default=0)

    def resolve(self):
        for expression, field in self.dependencies():
            if SpectrumMixin not in field.model.__mro__:
                continue
            try:
                q = field.model.select().where(expression)
            except:
                continue
            else:
                if q.exists():
                    return q.first()
                
        raise self.model.DoesNotExist(f"Cannot resolve spectrum with identifier {self.pk}")

                
class SpectrumMixin:

    '''
    @property
    def ivar(self):
        """
        Inverse variance of flux, computed from the `e_flux` attribute.
        """
        ivar = np.copy(self.e_flux**-2)
        ivar[~np.isfinite(ivar)] = 0
        return ivar

    @property
    def e_flux(self):
        """
        Uncertainty in flux (1-sigma).
        """
        e_flux = self.ivar**-0.5
        bad_pixel = (e_flux == 0) | (~np.isfinite(e_flux))
        e_flux[bad_pixel] = LARGE
        return e_flux
    '''

    def plot(self, rectified=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = (self.wavelength, self.flux)
        c = self.continuum if rectified else 1
        ax.plot(x, y / c, c='k')

        #ax.plot(x, self.model_flux)
        return fig


    def resample(self, wavelength, n_res, **kwargs):
        """
        Re-sample the spectrum at the given wavelengths.

        :param wavelength:
            A new wavelength array to sample the spectrum on.
        
        :param n_res:
            The number of resolution elements to use. This can be a float or a list-like
            of floats where the length i
        """
        return _resample(
            self.wavelength,
            wavelength,
            self.flux,
            self.ivar,
            n_res,
            pixel_flags=self.pixel_flags,
            **kwargs
        )
