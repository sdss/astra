from scipy import optimize  # if you remove this, everything at Utah breaks. seriously.
import numpy as np
import torch
from typing import Optional, Iterable
from functools import cache
from astropy.nddata import StdDevUncertainty
from astropy import units as u
from peewee import FloatField, IntegerField

from astra.base import task
from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.utils import get_metadata, create_bitmask
from astra.database.astradb import DataProduct, SDSSOutput
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.utils import expand_path, flatten

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ApogeeNetOutput(SDSSOutput):
    teff = FloatField()
    logg = FloatField()
    fe_h = FloatField()
    e_teff = FloatField()
    e_logg = FloatField()
    e_fe_h = FloatField()
    teff_sample_median = FloatField()
    logg_sample_median = FloatField()
    fe_h_sample_median = FloatField()
    bitmask_flag = IntegerField()


@cache
def read_model(model_path, device):
    return Model(expand_path(model_path), device)    


@task
def APOGEENet(
    data_product: Iterable[DataProduct],
    model_path: str = "$MWM_ASTRA/component_data/APOGEENet/model.pt",
    large_error: Optional[float] = 1e10,
    num_uncertainty_draws: Optional[int] = 100,
) -> Iterable[ApogeeNetOutput]:

    model = read_model(model_path, DEVICE)

    for data_product_ in flatten(data_product):
        for spectrum in SpectrumList.read(data_product_.path):
            if not spectrum_overlaps(spectrum, 16_500 * u.Angstrom):
                # Skip non-APOGEE spectra.
                continue

            N, P = np.atleast_2d(spectrum.flux).shape
            flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, P))
            e_flux = np.nan_to_num(
                    spectrum.uncertainty.represent_as(StdDevUncertainty).array
                ).astype(np.float32).reshape((N, P))
            
            meta_dict, metadata_norm = get_metadata(spectrum)
            meta = np.tile(metadata_norm, N).reshape((N, -1))

            N, P = flux.shape
            flux = flux.reshape((N, 1, P))
            e_flux = e_flux.reshape((N, 1, P))
            median_error = 5 * np.median(e_flux, axis=(1, 2))
            for j, value in enumerate(median_error):
                bad_pixel = (e_flux[j] == large_error) | (e_flux[j] >= value)
                e_flux[j][bad_pixel] = value

            flux = torch.from_numpy(flux).to(DEVICE)
            e_flux = torch.from_numpy(e_flux).to(DEVICE)
            meta = torch.from_numpy(meta).to(DEVICE)

            with torch.set_grad_enabled(False):
                predictions = model.predict_spectra(flux, meta)
                predictions = predictions.cpu().data.numpy()

            # Replace infinites with non-finite.
            predictions[~np.isfinite(predictions)] = np.nan

            inputs = (
                torch.randn(
                    (num_uncertainty_draws, N, 1, P), device=DEVICE
                )
                * e_flux
                + flux
            )
            inputs = inputs.reshape((num_uncertainty_draws * N, 1, P))

            meta_draws = meta.repeat(num_uncertainty_draws, 1)
            with torch.set_grad_enabled(False):
                draws = model.predict_spectra(inputs, meta_draws)
                draws = draws.cpu().data.numpy()

            draws = draws.reshape((num_uncertainty_draws, N, -1))

            # un-log10-ify the draws before calculating summary statistics
            predictions[:, 1] = 10 ** predictions[:, 1]
            draws[:, :, 1] = 10 ** draws[:, :, 1]

            median_draw_predictions = np.nanmedian(draws, axis=0)
            std_draw_predictions = np.nanstd(draws, axis=0)

            logg_median, teff_median, fe_h_median = median_draw_predictions[0]
            logg_std, teff_std, fe_h_std = std_draw_predictions[0]
            logg, teff, fe_h = predictions[0]
            bitmask_flag, = create_bitmask(
                predictions,
                meta_dict,
                median_draw_predictions=median_draw_predictions,
                std_draw_predictions=std_draw_predictions,
            )

            yield ApogeeNetOutput(
                data_product=data_product_,
                spectrum=spectrum,
                teff=teff,
                logg=logg,
                fe_h=fe_h,
                e_teff=teff_std,
                e_logg=logg_std,
                e_fe_h=fe_h_std,
                teff_sample_median=teff_median,
                logg_sample_median=logg_median,
                fe_h_sample_median=fe_h_median,
                bitmask_flag=bitmask_flag
            )
