
from astropy.io import registry
from astropy.table import Table
from .connect import PhotosphereRead, PhotosphereWrite


class Photosphere(Table):
    """A class to represent a model photosphere."""

    read = registry.UnifiedReadWriteMethod(PhotosphereRead)
    write = registry.UnifiedReadWriteMethod(PhotosphereWrite)

    @property
    def is_spherical_geometry(self):
        # TODO: we're assuming the radius units are in cm.
        # MARCs models give radius of 1 cm for plane-parallel models
        return self.meta.get("radius", 0) > 1
    
    @property
    def is_plane_parallel_geometry(self):
        return not self.is_spherical_geometry

    def plot(self, x, y, xlabel=None, ylabel=None, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(self[x], self[y], **kwargs)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        fig.tight_layout()
        return fig