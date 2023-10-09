import numpy as np
from typing import Optional, Tuple, Union, List
from astra.specutils.continuum.base import Continuum
from astra.pipelines.ferre.utils import get_apogee_pixel_mask
from scipy.ndimage import median_filter


class MedianFilter(Continuum):

    def __init__(
        self,
        median_filter_width: Optional[int] = 151,
        bad_minimum_flux: Optional[float] = 0.01,
        non_finite_err_value: Optional[float] = 1e10,
        valid_continuum_correction_range: Optional[Tuple[float]] = (0.1, 10.0),
        mode: Optional[str] = "mirror",
        regions: Optional[Tuple[Tuple[float, float]]] = (
            (15152, 15800),
            (15867, 16424),
            (16484, 16944),
        ),
        fill_value: Optional[Union[int, float]] = np.nan,
        **kwargs
    ) -> None:
        (
        """
        :param median_filter_width: [optional]
            The width (int) for the median filter (default: 151) in units of pixels.

        :param bad_minimum_flux: [optional]
            The value at which to set pixels as bad and median filter over them. This should be a float,
            or `None` to set no low-flux filtering (default: 0.01).

        :param non_finite_err_value: [optional]
            The error value to set for pixels with non-finite fluxes (default: 1e10).

        :param valid_continuum_correction_range: [optional]
            A (min, max) tuple of the bounds that the final correction can have. Values outside this range will be set
            as 1.
        """
            + Continuum.__init__.__doc__
        )
        super(MedianFilter, self).__init__(regions=regions, fill_value=fill_value, **kwargs)
        self.median_filter_width = median_filter_width
        self.bad_minimum_flux = bad_minimum_flux
        self.non_finite_err_value = non_finite_err_value
        self.valid_continuum_correction_range = valid_continuum_correction_range
        self.mode = mode
        self.mask = get_apogee_pixel_mask()
        return None


    def fit(self, spectrum, coarse_result) -> np.ndarray:
        """
        Fit a median continuum given an upstream coarse FERRE result.

        :param spectrum:
            The input spectrum.
        
        :param coarse_result:
            The database entry for the coarse FERRE result (from `astra.models.pipelines.FerreCoarse`).
        """
        

        flux = np.nan * np.ones(self.mask.shape, dtype=float)
        model_flux = np.nan * np.ones(self.mask.shape, dtype=float)

        # We will use FERRE-rectified fluxes here (model + data)
        flux[self.mask] = coarse_result.rectified_flux
        model_flux[self.mask] = coarse_result.rectified_model_flux
        
        continuum = self.fill_value * np.ones_like(spectrum.flux)
        for si, ei in self._get_region_slices(spectrum):
            
            flux_region = flux[si:ei].copy()
            ratio = flux_region / model_flux[si:ei]

            is_bad_pixel = (
                (flux_region < self.bad_minimum_flux)
            |   (flux_region > (np.nanmedian(flux_region) + 3 * np.nanstd(flux_region)))
            |   (~np.isfinite(ratio))
            |   (ratio < self.valid_continuum_correction_range[0])
            |   (ratio > self.valid_continuum_correction_range[1])
            )

            if not np.all(is_bad_pixel):
                x = np.arange(ratio.size)
                ratio[is_bad_pixel] = np.interp(x[is_bad_pixel], x[~is_bad_pixel], ratio[~is_bad_pixel])            
                continuum[si:ei] = median_filter(ratio, size=self.median_filter_width, mode=self.mode)

        return continuum
