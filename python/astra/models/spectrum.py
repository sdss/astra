from peewee import AutoField
from astra.models.base import BaseModel
from astra.models.fields import BitField
from functools import cached_property
import numpy as np

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

    @cached_property
    def ref(self):
        return self.resolve()
    
    def __getattr__(self, attr):
        # Resolve to reference attribute
        return getattr(self.ref, attr)
    
    def __repr__(self):
        return f"<Spectrum pointer -> ({self.ref.__repr__().strip('<>')})>"
    
    
    def plot(self, figsize=(8, 3), ylim_percentile=(1, 99)):
        
        import matplotlib.pyplot as plt        
        # to gracefully handle apVisits
        x = np.atleast_2d(self.wavelength)
        y = np.atleast_2d(self.flux) 
        try:
            y_err = np.atleast_2d(self.e_flux)
        except:
            try:
                y_err = np.atleast_2d(self.ivar)**-0.5
            except:
                y_err = np.nan * np.ones_like(self.flux)
        
        N, P = y.shape
        
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(N):
            ax.plot(
                x[i],
                y[i],
                c='k',
                label=f"spectrum_pk={self.pk}" if i == 0 else None,
                drawstyle="steps-mid"
            )
            ax.fill_between(
                x[i],
                y[i] - y_err[i],
                y[i] + y_err[i],
                step="mid",
                facecolor="#cccccc",
                zorder=-1                
            )
        
        # discern some useful limits
        ax.set_xlim(*x.flatten()[[0, -1]])
        ax.set_ylim(np.clip(np.nanpercentile(y, ylim_percentile), 0, np.inf))        
        ax.set_xlabel(r"$\lambda$ $(\mathrm{\AA})$")
        ax.set_ylabel(r"$f_\lambda$ $(10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^2\,\mathrm{\AA}^{-1})$")
        fig.tight_layout()
        return fig
    
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
