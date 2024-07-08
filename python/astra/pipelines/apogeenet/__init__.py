import os
import io
import sys
import torch
from tqdm import tqdm
from astra.pipelines.apogeenet.boss_net import BossNet
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

# Unnormalization values for the stellar parameters.
stellar_parameter_stats = StellarParameters(
    FEH=DataStats(
        MEAN=-0.2,
        PNMAX=4.568167,
        PNMIN=-8.926531221275827,
        STD=0.3,
    ),
    LOGG=DataStats(
        MEAN=3.2,
        PNMAX=3.3389749999999996,
        PNMIN=-3.2758333384990697,
        STD=1.2,
    ),
    LOGTEFF=DataStats(
        MEAN=3.7,
        PNMAX=9.387230328989702,
        PNMIN=-5.2989908487604165,
        STD=0.1,
    ),
)
# Data structure for the output of the model.
PredictionOutput = namedtuple('PredictionOutput', ['log_G', 'log_Teff', 'FeH'])

# Data structure for uncertainty predictions.
UncertaintyOutput = namedtuple('UncertaintyOutput', [
    'log_G_median', 'log_Teff_median', 'Feh_median', 
    'log_G_std', 'log_Teff_std', 'Feh_std', 
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


def open_apogee_fits(file_path: str):
    with fits.open(file_path) as hdul:
        # We use the first index to specify that we are using the coadded spectrum
        flux = hdul[1].data.astype(np.float32)[0]
        error = hdul[2].data.astype(np.float32)[0]
        wavlen = np.zeros_like(flux)
        flux = np.nan_to_num(flux, nan=0.0)
    flux = torch.from_numpy(flux).float()
    error = torch.from_numpy(error).float()

    return flux, error, wavlen

def make_prediction(spectra, error, wavlen,num_uncertainty_draws,model,device):

    # log scale spectra
    normalized_spectra = log_scale_flux(spectra).float().reshape((1,1,spectra.size()[0]))
    # Calculate and unnormalize steller parameter predictions
    normalized_prediction = model(normalized_spectra.to(device))
    prediction = unnormalize_predictions(normalized_prediction)
    prediction = prediction.squeeze()
    # Unpack stellar parameters
    log_G = prediction[0].item()
    log_Teff = prediction[1].item()
    FeH = prediction[2].item()

    # Get batch of noised spectra
    uncertainties_batch = create_uncertainties_batch(spectra,error, num_uncertainty_draws)
    # scale sprectra
    normalized_uncertainties_batch = log_scale_flux(uncertainties_batch).float()
    x=normalized_uncertainties_batch.shape
    normalized_uncertainties_batch=normalized_uncertainties_batch.reshape((x[0],1,x[1]))
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
    # Unpack stds
    log_G_std = std[0].item()
    log_Teff_std = std[1].item()
    Feh_std = std[2].item()

    return log_G,log_Teff,FeH,log_G_std,log_Teff_std,Feh_std


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

from astra import task
from astra.utils import log, expand_path

from astra.models import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar
from astra.models.apogeenet import ApogeeNet
from peewee import JOIN, ModelSelect
from typing import Optional, Iterable, Union



@task
def apogeenet(
    spectra: Optional[Iterable[Union[ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar]]] = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .join(ApogeeNet, JOIN.LEFT_OUTER, on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == ApogeeNet.spectrum_pk))
        .where(ApogeeNet.spectrum_pk.is_null())
    ),
    num_uncertainty_draws: Optional[int] = 20,
    limit=None,
    **kwargs
) -> Iterable[ApogeeNet]:
    """
    Run the ANet (APOGEENet III) pipeline.
    
    """
    
    
    model = BossNet()
    model_path = expand_path("$MWM_ASTRA/pipelines/ANet/deconstructed_model")
    state_dict = franken_load(model_path, 10)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # As per https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
    if torch.cuda.is_available():
        model.cuda()
    
    if isinstance(spectra, ModelSelect):
        if limit is not None:
            spectra = spectra.limit(limit)
        # Note: if you don't use the `.iterator()` you may get out-of-memory issues from the GPU nodes 
        spectra = spectra.iterator() 
    
    for spectrum in tqdm(spectra, total=0):
        
        try:        
            flux = np.nan_to_num(spectrum.flux, nan=0.0).astype(np.float32)
            e_flux = reverse_inverse_error(spectrum.ivar, 0.1 * np.median(flux)).astype(np.float32) # TODO: do as per how BOSSNet does it?
            
            flux = torch.from_numpy(flux).float()
            e_flux = torch.from_numpy(e_flux).float()
            
            log_G,log_Teff,FeH,log_G_std,log_Teff_std,Feh_std = make_prediction(flux, e_flux, None, num_uncertainty_draws,model,device)
        except:
            log.exception(f"Exception when running ApogeeNet on {spectrum}")    
            yield ApogeeNet(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                flag_runtime_exception=True
            )
        else:
            yield ApogeeNet(
                spectrum_pk=spectrum.spectrum_pk,
                source_pk=spectrum.source_pk,
                fe_h=FeH,
                e_fe_h=Feh_std,
                logg=log_G,
                e_logg=log_G_std,
                teff=10**log_Teff,
                e_teff=10**log_Teff * log_Teff_std * np.log(10)                
            )
            

'''
path = expand_path(
    "$SAS_BASE_DIR/sdsswork/mwm/apogee/spectro/redux/{apred}/{apstar}/{telescope}/{healpix_group}/{healpix}/apStar-{apred}-{telescope}-{obj}.fits".format(
        apred="daily",
        apstar="stars",
        telescope="apo25m",
        obj="2M04482524+4456383",
        healpix_group="28",
        healpix="28656",
    )
)


####### to run APOGEE Net
flux, error, wavlen=open_apogee_fits(path)
log_G,log_Teff,FeH,log_G_std,log_Teff_std,Feh_std=make_prediction(flux, error, wavlen,num_uncertainty_draws,model,device)
'''
