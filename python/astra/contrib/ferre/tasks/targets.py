import os
from astra.tasks.io import SDSS4ApStarFile as ApStarFile
from astra.tasks.targets import (LocalTarget, DatabaseTarget)
from astra.contrib.ferre.tasks.mixin import GridHeaderFileMixin
from sqlalchemy import (Column, Float)

from astra.contrib.ferre.tasks.mixin import SPECLIB_DIR

class GridHeaderFile(GridHeaderFileMixin):

    """
    A task to represent a grid header file.

    :param radiative_transfer_code:
        A string description of the radiative transfer code used (e.g., turbospectrum).

    :param model_photospheres:
        A string description of the model photospheres used (e.g., marcs).
    
    :param isotopes:
        A description of the isotope ratios used (e.g., giantisotopes).
    
    :param gd:
        A (legacy) character indicating whether this is a giant grid or a dwarf grid.
    
    :param spectral_type:
        The spectral type of the grid.
    
    :param grid_creation_date:
        The date that the grid was created.
    
    :param lsf:
        A single character describing the LSF of the grid (e.g., 'a', 'b', 'c', or 'd').
    
    :param aspcap:
        The ASPCAP reduction version. (TODO: Legacy keywords.)

    """

    def output(self):
        date_str = self.grid_creation_date.strftime("%y%m%d")
        return LocalTarget(
            os.path.join(
                SPECLIB_DIR,
                self.radiative_transfer_code,
                self.model_photospheres,
                self.isotopes,
                # TODO: Check with HoltZ what the format is here. Is it always t prefix or is that for turbospectrum?
                f"t{self.gd}{self.spectral_type}_{date_str}_lsf{self.lsf}_{self.aspcap}",
                # TODO: Check with HoltZ on what 012_075 means
                f"p_apst{self.gd}{self.spectral_type}_{date_str}_lsf{self.lsf}_{self.aspcap}_012_075.hdr"
            )
        )


class FerreResult(DatabaseTarget):

    """ A database row representing an output target from FERRE. """

    # TODO: We should consider generating this results schema from the grid header file.    
    teff = Column("TEFF", Float)
    logg = Column("LOGG", Float)
    metals = Column("METALS", Float)
    alpha_m = Column("O Mg Si S Ca Ti", Float)
    n_m = Column("N", Float)
    c_m = Column("C", Float)
    log10vdop = Column("LOG10VDOP", Float)
    # Not all grids have LGVSINI.
    lgvsini = Column("LGVSINI", Float, nullable=True)
    log_snr_sq = Column("log_snr_sq", Float)
    log_chisq_fit = Column("log_chisq_fit", Float)
    

    