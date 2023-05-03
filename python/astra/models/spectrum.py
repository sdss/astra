from peewee import AutoField
from astra.models.base import BaseModel


class Spectrum(BaseModel):

    """ A one dimensional spectrum. """

    spectrum_id = AutoField()
    

class SpectrumMixin:

    def plot(self, rectified=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = (self.wavelength, self.flux)
        c = self.continuum if rectified else 1
        ax.plot(x, y / c, c='k')

        #ax.plot(x, self.model_flux)
        return fig