

import numpy as np
import warnings
from tqdm import tqdm

from model import CannonModel


def grid_search(
    label_names,
    training_labels,
    training_flux,
    training_ivar,
    validation_labels,
    validation_flux,
    validation_ivar,
    log10_alpha_min=-10,
    log10_alpha_max=-1,
    log10_alpha_delta=0.25,
    **kwargs
):
    """
    Perform a grid search to set the regularization strength.

    Returns a three length tuple of:
    - the best regularization strength
    - the trained model with the best regularization strength,
    - a metadata dictionary
    """

    models = []

    alphas = 10**np.arange(log10_alpha_min, log10_alpha_max, log10_alpha_delta)
    validation_chi_sqs = np.zeros_like(alphas)

    for i, alpha in enumerate(tqdm(alphas, desc="Regularization grid search")):
        model = CannonModel(
            training_labels,
            training_flux,
            training_ivar,
            label_names,
            regularization=alpha,
        )
        model.train(**kwargs)
        validation_chi_sqs[i] = model.chi_sq(validation_labels, validation_flux, validation_ivar)

        models.append(model)
    
    # Select model with lowest validation \chi^2
    index = np.argmin(validation_chi_sqs)
    if index == 0 or index == (validation_chi_sqs.size - 1):
        warnings.warn(f"Regularization strength with lowest validation \chi^2 is on the edge of the grid: {alphas[index]:.2e}")

    meta = dict(
        models=models,
        alphas=alphas,
        validation_chi_sqs=validation_chi_sqs,
    )
    return (alphas[index], models[index], meta)
