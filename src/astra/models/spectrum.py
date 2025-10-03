import warnings
import numpy as np
from astra.fields import AutoField, BitField
from astra.utils import log
from astra.models.base import BaseModel
from functools import cached_property

class SpectrumMixin(object):

    @property
    def e_flux(self):
        e_flux = self.ivar**-0.5
        e_flux[~np.isfinite(e_flux)] = 1e10
        return e_flux


    def plot(self, rectified=True, plot_model=True, figsize=(8, 3), ylim_percentile=(1, 99)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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

            continuum, is_rectified = (1, False)
            if rectified:
                try:
                    for key in ("continuum", ):
                        continuum = getattr(self, key, None)
                        if continuum is not None:
                            is_rectified = True
                            break
                    else:
                        log.warning(f"Cannot find continuum for spectrum {self}")
                        continuum = 1
                except:
                    log.exception(f"Exception when trying to get continum for spectrum {self}")

            has_model_flux = False
            if plot_model:
                try:
                    model_flux = self.model_flux
                except AttributeError:
                    try:
                        model_flux = self.nmf_rectified_model_flux * continuum
                    except:
                        None
                    else:
                        has_model_flux = True
                else:
                    has_model_flux = True

            N, P = y.shape


            fig, ax = plt.subplots(figsize=figsize)
            for i in range(N):
                label = None
                if i == 0:
                    try:
                        for k in ("spectrum_pk", "pk", "task_pk"):
                            v = getattr(self, k, None)
                            if v is not None:
                                label = f"{k}={v}"
                    except:
                        None

                ax.plot(
                    x[i],
                    y[i] / continuum,
                    c='k',
                    label=label,
                    drawstyle="steps-mid",
                )
                ax.fill_between(
                    x[i],
                    (y[i] - y_err[i])/continuum,
                    (y[i] + y_err[i])/continuum,
                    step="mid",
                    facecolor="#cccccc",
                    zorder=-1
                )

            if has_model_flux:
                try:
                    ax.plot(
                        self.wavelength,
                        model_flux / continuum,
                        c="tab:red",
                        label="model",
                    )
                except:
                    log.exception(f"Exception when trying to plot model flux for {self}")

            # discern some useful limits
            ax.set_xlim(*x.flatten()[[0, -1]])
            if is_rectified:
                ax.set_ylim(0, 1.2)
                ax.axhline(1.0, c='#666666', ls=":", lw=0.5, zorder=-1)
                ax.set_ylabel(r"$f_\lambda$ $(\mathrm{rectified})$")
            else:
                ylim = np.clip(np.nanpercentile(y, ylim_percentile), 0, np.inf)
                offset = np.ptp(ylim) * 0.10
                ylim = (ylim[0] - offset, ylim[1] + offset)
                ax.set_ylim(ylim)
                ax.set_ylabel(r"$f_\lambda$ $(10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^2\,\mathrm{\AA}^{-1})$")

            ax.set_xlabel(r"$\lambda$ $(\mathrm{\AA})$")
            fig.tight_layout()
            return fig




class Spectrum(BaseModel, SpectrumMixin):

    """ A one dimensional spectrum. """

    pk = AutoField(primary_key=True)
    spectrum_flags = BitField(default=0)


    def resolve(self):
        try:
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
        except:
            raise self.model.DoesNotExist(f"Cannot resolve spectrum with identifier {self.pk}")

    @cached_property
    def ref(self):
        return self.resolve()

    def __getattr__(self, attr):
        # Resolve to reference attribute
        return getattr(self.ref, attr)

    def __repr__(self):
        return f"<Spectrum pk={self.pk} -> ({self.ref.__repr__().strip('<>')})>"
