import numpy as np
import torch
from peewee import JOIN
from typing import Optional, Iterable
from functools import cache
from astra import task
from astra.utils import log, expand_path
from astra.models import ApogeeVisitSpectrum, BossVisitSpectrum
from astra.models.classifier import SpectrumClassification
from astra.pipelines.classifier.utils import read_network, classification_result
from astra.pipelines.classifier import networks

SMALL = -1e+20

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0") if CUDA_AVAILABLE else torch.device("cpu")

@cache
def read_model(model_path):
    model_path = expand_path(model_path)
    factory = getattr(networks, model_path.split("_")[-2])
    model = read_network(factory, model_path)
    model.to(DEVICE)
    model.eval()
    return model


@task
def classify_apogee_visit_spectrum(
    spectra: Optional[Iterable[ApogeeVisitSpectrum]] = (
        ApogeeVisitSpectrum
        .select()
        .join(SpectrumClassification, JOIN.LEFT_OUTER, on=(SpectrumClassification.spectrum_pk == ApogeeVisitSpectrum.spectrum_pk))
        .where(SpectrumClassification.spectrum_pk.is_null())
        .iterator()
    ),
    model_path: str = "$MWM_ASTRA/pipelines/classifier/classifier_NIRCNN_77804646.pt",
) -> Iterable[SpectrumClassification]:
    """
    Classify a source, given an APOGEE visit spectrum (an apVisit data product).
    """

    expected_shape = (3, 4096)

    model = read_model(model_path)
    log.info(f"Making predictions..")

    for spectrum in spectra:
        try:                
            if not spectrum.dithered:
                existing_flux = spectrum.flux.reshape((3, -1))
                flux = np.empty(expected_shape)
                for j in range(3):
                    flux[j, ::2] = existing_flux[j]
                    flux[j, 1::2] = existing_flux[j]   
            else:
                flux = spectrum.flux.reshape(expected_shape)
            
            continuum = np.nanmedian(flux, axis=1)
            batch = flux / continuum.reshape((-1, 1))
            batch = batch.reshape((-1, *expected_shape)).astype(np.float32)
            batch = torch.from_numpy(batch).to(DEVICE)

            with torch.no_grad():
                prediction = model.forward(batch)

            # Should be only one result with apVisit, but whatever..
            for log_probs in prediction.cpu().numpy():
                result = classification_result(log_probs, model.class_names)
                yield SpectrumClassification(
                    spectrum_pk=spectrum.spectrum_pk,
                    source_pk=spectrum.source_pk,
                    **result
                )
        except:
            # TODO: yield a record with no result?
            None


@task
def classify_boss_visit_spectrum(
    spectra: Optional[Iterable[BossVisitSpectrum]] = (
        BossVisitSpectrum
        .select()
        .join(SpectrumClassification, JOIN.LEFT_OUTER, on=(SpectrumClassification.spectrum_pk == BossVisitSpectrum.spectrum_pk))
        .where(SpectrumClassification.spectrum_pk.is_null())
        .iterator()
    ),
    model_path: str = "$MWM_ASTRA/component_data/classifier/classifier_OpticalCNN_40bb9164.pt",
) -> Iterable[SpectrumClassification]:
    """
    Classify a source, given a BOSS visit spectrum.
    """
    

    model = read_model(model_path)
    si, ei = (0, 3800)  # MAGIC: same done in training
    log.info(f"Making predictions")

    for spectrum in spectra:
        try:
            flux = spectrum.flux[si:ei]
            continuum = np.nanmedian(flux)
            batch = flux / continuum
            # remove nans
            finite = np.isfinite(batch)
            if not any(finite):
                log.warning(f"Skipping {spectrum} because all values are NaN")
                continue

            if any(~finite):
                batch[~finite] = np.interp(
                    spectrum.wavelength.value[si:ei][~finite],
                    spectrum.wavelength.value[si:ei][finite],
                    batch[finite],
                )        
            batch = batch.reshape((1, 1, -1)).astype(np.float32)
            batch = torch.from_numpy(batch).to(DEVICE)

            with torch.no_grad():
                prediction = model.forward(batch)

            # Should be only one result with specFull, but whatever..
            for log_probs in prediction.cpu().numpy():
                result = classification_result(log_probs, model.class_names)
                yield SpectrumClassification(
                    spectrum_pk=spectrum.spectrum_pk,
                    source_pk=spectrum.source_pk,
                    **result
                )
        except:
            # TODO: yield a record with no result?
            None

    log.info("Done")

