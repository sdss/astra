import os
import io
import sys
import torch
from tqdm import tqdm
from astra.pipelines.bnet.boss_net import BossNet
from typing import Optional
from dataclasses import dataclass
import tempfile
from collections import OrderedDict, namedtuple
import numpy as np
from functools import partial
from astropy.io import fits

@dataclass(frozen=True)
class DataStats:
    """
    A dataclass that holds stellar parameter statistics related to a single value.

    Attributes:
    - MEAN: float, the mean value of the value.
    - STD: float, the standard deviation of the value.
    - PNMAX: float, the post-normalization maximum value of the value.
    - PNMIN: float, the post-normalization minimum value of the value.
    """
    MEAN: float
    STD: float
    PNMAX: float
    PNMIN: float

@dataclass(frozen=True)
class StellarParameters:
    """
    The StellarParameters class is a dataclass that represents the statistical properties of 
    three stellar parameters: effective temperature (LOGTEFF), surface gravity (LOGG), and 
    metallicity (FEH). The class contains three attributes, each of which is an instance of 
    the DataStats class, representing the mean, standard deviation, post-normalization minimum, 
    and post-normalization maximum values of each parameter.

    Attributes:
    - LOGTEFF: DataStats, representing the statistical properties of the effective temperature.
    - LOGG: DataStats, representing the statistical properties of the surface gravity.
    - FEH: DataStats, representing the statistical properties of the metallicity.
    - RV: DataStats, representing the statistical properties of the radial velocity.
    """
    LOGTEFF: DataStats
    LOGG: DataStats
    FEH: DataStats
    RV: DataStats

# Unnormalization values for the stellar parameters.
stellar_parameter_stats = StellarParameters(
    LOGTEFF=DataStats(
        MEAN=3.8,
        PNMAX=12.000000000000002,
        PNMIN=-6.324908332532,
        STD=0.1,
    ),
    LOGG=DataStats(
        MEAN=3.9,
        PNMAX=6.584444444444444,
        PNMIN=-4.403565883333333,
        STD=0.9,
    ),
    FEH=DataStats(
        MEAN=-0.4,
        PNMAX=4.4496842,
        PNMIN=-7.496200000000001,
        STD=0.5,
    ),
    RV=DataStats(
        MEAN=-7.4,
        PNMAX=9.074319773706897,
        PNMIN=-9.116415791127874,
        STD=60.9
    ),
)

# Data structure for the output of the model.
PredictionOutput = namedtuple('PredictionOutput', ['log_G', 'log_Teff', 'FeH', 'rv'])

# Data structure for uncertainty predictions.
UncertaintyOutput = namedtuple('UncertaintyOutput', [
    'log_G_median', 'log_Teff_median', 'Feh_median', 'rv_median',
    'log_G_std', 'log_Teff_std', 'Feh_std', 'rv_std'
])

def unnormalize(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    This function takes in a PyTorch tensor X, and two scalar values mean and std, and returns the unnormalized tensor.

    Args:
    - X: torch.Tensor, input tensor to be unnormalized.
    - mean: float, mean value used for normalization.
    - std: float, standard deviation used for normalization.

    Returns:
    - torch.Tensor: Unnormalized tensor with the same shape as the input tensor X.
    """
    return X * std + mean

def unnormalize_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """
    The unnormalize_predictions function takes a tensor X of shape (batch_size, 3) and unnormalizes 
    each of its three columns using the mean and standard deviation of the corresponding DataStats 
    objects. Specifically, the first column corresponds to LOGG, the second to LOGTEFF, the third to FEH,
    and the fourth to RV.

    Args:
    - predictions: torch.Tensor, Input tensor of shape (batch_size, 4).
    - stellar_parameter_stats: StellarParameters, an object containing the mean and standard deviation of 
      the three columns of X.

    Returns:
    - torch.Tensor: Output tensor of shape (batch_size, 4) where each column has been unnormalized using 
      the mean and standard deviation stored in stellar_parameter_stats.
    """
    predictions[:, 0] = unnormalize(predictions[:, 0], stellar_parameter_stats.LOGG.MEAN, stellar_parameter_stats.LOGG.STD)
    predictions[:, 1] = unnormalize(predictions[:, 1], stellar_parameter_stats.LOGTEFF.MEAN, stellar_parameter_stats.LOGTEFF.STD)
    predictions[:, 2] = unnormalize(predictions[:, 2], stellar_parameter_stats.FEH.MEAN, stellar_parameter_stats.FEH.STD)
    predictions[:, 3] = unnormalize(predictions[:, 3], stellar_parameter_stats.RV.MEAN, stellar_parameter_stats.RV.STD)

    return predictions

def franken_load(load_path: str, chunks: int) -> OrderedDict:
    """
    Loads a PyTorch model from multiple binary files that were previously split.

    Args:
        load_path: str, The directory where the model chunks are located.
        chunks: int, The number of model chunks to load.

    Returns:
        A ordered dictionary containing the PyTorch model state.
    """

    def load_member(load_path: str, file_out: io.BufferedReader, file_i: int) -> None:
        """
        Reads a single chunk of the model from disk and writes it to a buffer.

        Args:
            load_path: str, The directory where the model chunk files are located.
            file_out: io.BufferedReader, The buffer where the model chunks are written.
            file_i: int, The index of the model chunk to read.

        """
        load_name = os.path.join(load_path, f"model_chunk_{file_i}")
        with open(load_name, "rb") as file_in:
            file_out.write(file_in.read())

    with tempfile.TemporaryDirectory() as tempdir:
        # Create a temporary file to write the model chunks.
        model_path = os.path.join(tempdir, "model.pt")
        with open(model_path, "wb") as file_out:
            # Load each model chunk and write it to the buffer.
            for i in range(chunks):
                load_member(load_path, file_out, i)
        
        # Load the PyTorch model from the buffer.
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    return state_dict

def create_uncertainties_batch(flux: torch.Tensor, error: torch.Tensor, num_uncertainty_draws: int) -> torch.Tensor:
    """
    Creates a batch of flux tensors with added noise from the specified error tensors.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.
    - num_uncertainty_draws: int, The number of times to draw noise samples to create a batch of flux tensors.

    Returns:
    - flux_with_noise: torch.Tensor, A torch.Tensor representing the batch of flux tensors with added noise from the 
      specified error tensors.
    """
    normal_sample = torch.randn((num_uncertainty_draws, *error.shape[-2:]))
    return flux + error * normal_sample

def interpolate_flux(
    flux: torch.Tensor, wavelen: torch.Tensor, linear_grid: torch.Tensor
) -> torch.Tensor:
    """
    The function interpolate_flux takes in the flux, wavelength, and linear grid of a spectrum,
    interpolates the flux onto a new linear wavelength grid, and returns the interpolated flux as
    a torch.Tensor.

    Args:
    - flux_batch: torch.Tensor, A torch.Tensor representing the flux values of the spectrum.
    - wavelen: torch.Tensor, A torch.Tensor representing the wavelength values of the spectrum.
    - linear_grid: torch.Tensor, A torch.Tensor representing the new linear wavelength grid to
      interpolate the flux onto.

    Returns:
    - interpolated_flux: torch.Tensor, A torch.Tensor representing the interpolated flux values
      of the spectrum on the new linear wavelength grid.
    """
    interpolated_flux = torch.zeros(1,1, len(linear_grid))
    _wavelen = wavelen[~torch.isnan(flux)]
    _flux = flux[~torch.isnan(flux)]
    _flux = np.interp(linear_grid, _wavelen, _flux)
    _flux = torch.from_numpy(_flux)
    interpolated_flux[0] = _flux
    return interpolated_flux

def interpolate_flux_err(
    flux_batch: torch.Tensor, wavelen: torch.Tensor, linear_grid: torch.Tensor
) -> torch.Tensor:
    """
    The function interpolate_flux takes in the flux, wavelength, and linear grid of a spectrum,
    interpolates the flux onto a new linear wavelength grid, and returns the interpolated flux as
    a torch.Tensor.

    Args:
    - flux_batch: torch.Tensor, A torch.Tensor representing the flux values of the spectrum.
    - wavelen: torch.Tensor, A torch.Tensor representing the wavelength values of the spectrum.
    - linear_grid: torch.Tensor, A torch.Tensor representing the new linear wavelength grid to
      interpolate the flux onto.

    Returns:
    - interpolated_flux: torch.Tensor, A torch.Tensor representing the interpolated flux values
      of the spectrum on the new linear wavelength grid.
    """
    interpolated_flux = torch.zeros(*flux_batch.shape[:-1],1, len(linear_grid))
    for i, flux in enumerate(flux_batch):
        _wavelen = wavelen[~torch.isnan(flux)]
        _flux = flux[~torch.isnan(flux)]
        _flux = np.interp(linear_grid, _wavelen, _flux)
        _flux = torch.from_numpy(_flux)
        interpolated_flux[i] = _flux
    return interpolated_flux

def log_scale_flux(flux: torch.Tensor) -> torch.Tensor:
    """
    The function log_scale_flux applies a logarithmic scaling to the input flux tensor and clips the values
    to remove outliers.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.

    Returns:
    - flux: torch.Tensor, A torch.Tensor representing the logarithmically scaled flux values of the data.
    The values are clipped at the 95th percentile plus one to remove outliers.
    """
    s = 0.000001
    flux = torch.clip(flux, min=s, max=None)
    flux = torch.log(flux)
    perc_95 = torch.quantile(flux, 0.95)
    flux = torch.clip(flux, min=None, max=perc_95 + 1)
    return flux

def reverse_inverse_error(inverse_error: np.array, default_error: int) -> np.array:
    """
    A function that calculates error values from inverse errors.

    Args:
    - inverse_error: np.array, a numpy array containing inverse error values.
    - default_error: int, an integer to use for error values that cannot be calculated.

    Returns:
    - error: np.array, a numpy array containing calculated error values.

    The function calculates error values from inverse error values by taking the square root of the reciprocal of each
    value in the input `inverse_error` array. The resulting array is then processed to replace any infinite or NaN values
    with a default error value or a multiple of the median non-infinite value in the array. The resulting array is returned
    as a numpy array.
    """
    np.seterr(all="ignore")
    inverse_error = np.nan_to_num(inverse_error)
    error = np.divide(1, inverse_error) ** 0.5
    if np.isinf(error).all():
        error = np.ones(*error.shape) * default_error
        error = error.astype(inverse_error.dtype)
    median_error = np.nanmedian(error[error != np.inf])
    error = np.clip(error, a_min=None, a_max=5 * median_error)
    error = np.where(np.isnan(error), 5 * median_error, error)
    return error

def open_boss_fits(file_path):
    """ 
    The function open_boss_fits opens a BOSS FITS file and returns three torch.Tensors
    representing the flux, error, and wavelength of the data.

    Args:
    - file_path: str, The path to the BOSS FITS file to be opened.

    Returns:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.
    - wavlen: torch.Tensor, A torch.Tensor representing the wavelength values of the data.
    """
    with fits.open(file_path) as hdul:
        spec = hdul[1].data
        flux = spec["flux"].astype(np.float32)
        inverse_error = spec["ivar"].astype(np.float32)
        error = reverse_inverse_error(inverse_error, np.median(flux) * 0.1)
        wavlen = 10 ** spec["loglam"].astype(np.float32)

    flux = torch.from_numpy(flux).float()
    error = torch.from_numpy(error).float()
    wavlen = torch.from_numpy(wavlen).float()

    return flux, error, wavlen

def make_prediction(spectra, error, wavlen,num_uncertainty_draws,model,device):

    # Interpolate and log scale spectra
    interp_spectra = interpolate_flux(spectra, wavlen)
    normalized_spectra = log_scale_flux(interp_spectra).float()

    # Calculate and unnormalize steller parameter predictions
    normalized_prediction = model(normalized_spectra.to(device))
    prediction = unnormalize_predictions(normalized_prediction)
    prediction = prediction.squeeze()

    # Unpack stellar parameters
    log_G = prediction[0].item()
    log_Teff = prediction[1].item()
    FeH = prediction[2].item()
    rv = prediction[3].item()

    uncertainties_batch = create_uncertainties_batch(spectra, error, num_uncertainty_draws)
    # Interpolate and log scale sprectra
    interp_uncertainties_batch = interpolate_flux_err(uncertainties_batch, wavlen)
    normalized_uncertainties_batch = log_scale_flux(interp_uncertainties_batch).float()
    # Calculate and unnormalize stellar parameters predictions
    normalized_predictions_batch = model(normalized_uncertainties_batch.to(device))
    prediction = unnormalize_predictions(normalized_predictions_batch)
    # Calculate the median and std for each stellar parameter
    median = torch.median(prediction, axis=0)[0]
    std = torch.std(prediction, axis=0)
    # Unpack medians
    log_G_median = median[0].item()
    log_Teff_median = median[1].item()
    Feh_median = median[2].item()
    rv_median = median[3].item()
    # Unpack stds
    log_G_std = std[0].item()
    log_Teff_std = std[1].item()
    Feh_std = std[2].item()
    rv_std = std[3].item()

    return log_G,log_Teff,FeH,rv,log_G_std,log_Teff_std,Feh_std,rv_std
    
    
    

from astra import task
from astra.utils import log, expand_path

from astra.models import BossVisitSpectrum
from astra.models.bnet import BNet
from peewee import JOIN
from typing import Optional, Iterable

MIN_WL, MAX_WL, FLUX_LEN = 3800, 8900, 3900
linear_grid = torch.linspace(MIN_WL, MAX_WL, steps=FLUX_LEN)
interpolate_flux = partial(interpolate_flux, linear_grid=linear_grid)
interpolate_flux_err = partial(interpolate_flux_err, linear_grid=linear_grid)   

@task
def bnet(
    spectra: Optional[Iterable[BossVisitSpectrum]] = (
        BossVisitSpectrum
        .select()
        .join(BNet, JOIN.LEFT_OUTER, on=(BossVisitSpectrum.spectrum_pk == BNet.spectrum_pk))
        .where(BNet.spectrum_pk.is_null())
    ),
    num_uncertainty_draws: Optional[int] = 20
) -> Iterable[BNet]:
    
    model = BossNet()
    model_path = expand_path("$MWM_ASTRA/pipelines/BNet/deconstructed_model")
    state_dict = franken_load(model_path, 10)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # As per https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
    if torch.cuda.is_available():
        model.cuda()

    if isinstance(spectra, ModelSelect):
        # Note: if you don't use the `.iterator()` you may get out-of-memory issues from the GPU nodes 
        spectra = spectra.iterator()         
    
    for spectrum in tqdm(spectra, total=0):
        
        try:
            flux = np.nan_to_num(spectrum.flux, nan=0.0).astype(np.float32)
            e_flux = reverse_inverse_error(spectrum.ivar.astype(np.float32), np.median(flux) * 0.1).astype(np.float32)
            wavelen = spectrum.wavelength.astype(np.float32)
            
            flux = torch.from_numpy(flux).float()
            e_flux = torch.from_numpy(e_flux).float()
            wavelen = torch.from_numpy(wavelen).float()
            
            log_G,log_Teff,FeH,rv,log_G_std,log_Teff_std,Feh_std,rv_std = make_prediction(flux, e_flux, wavelen, num_uncertainty_draws,model,device)
        except:
            log.exception(f"Exception when running ANet on {spectrum}")    
            yield ANet(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_runtime_exception=True
            )            
        else:
            yield BNet(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                fe_h=FeH,
                e_fe_h=Feh_std,
                logg=log_G,
                e_logg=log_G_std,
                teff=10**log_Teff,
                e_teff=10**log_Teff_std, # check
                v_rad=rv,
                e_v_rad=rv_std
            )
            


'''
model = BossNet()
model_path = "deconstructed_model"
state_dict = franken_load(model_path, 10)
model.load_state_dict(state_dict, strict=False)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_uncertainty_draws=20




####### to run BOSS Net
flux, error, wavlen = open_boss_fits('spec-015223-59265-4515432683.fits')
log_G,log_Teff,FeH,rv,log_G_std,log_Teff_std,Feh_std,rv_std=make_prediction(flux, error, wavlen,num_uncertainty_draws,model,device)
'''