import numpy as np
import os
import torch

from tqdm import tqdm

from sdss_access import SDSSPath
from astra.utils import log
from astra.contrib.apogeenet.model import (Net, predict, create_flux_tensor)
from astra.database import astradb, session
from astra.database.utils import (
    deserialize_pks,
    create_task_output
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def estimate_stellar_labels(
        pks,
        model_path,
        analyze_individual_visits=True,
        num_uncertainty_draws=100,
        large_error=1e10
    ):
    """
    Estimate the stellar parameters for APOGEE ApStar observations,
    where task instances have been created with the given primary keys (`pks`).

    :param pks:
        The primary keys of task instances that include information about what
        ApStar observation to load.
    
    :param model_path:
        The disk path of the pre-trained model.
    
    :param analyze_individual_visits: [optional]
        Analyze individual visits stored in the ApStar object. If `False` then it
        will only analyze the stacked (zero-th index) observation (default: `True`).
    
    :param num_uncertainty_draws: [optional]
        The number of random draws to make of the flux uncertainties, which will be
        propagated into the estimate of the stellar parameter uncertainties (default: 100).
    
    :param large_error: [optional]
        An arbitrarily large error value to assign to bad pixels (default: 1e10).
    """
    
    log.info(f"Running APOGEENet on device {device}")

    # Load the model.
    model = Net()
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        ),
        strict=False
    )
    model.to(device)
    # Disable any dropout for inference.
    model.eval()

    log.info(f"Loaded model from {model_path}")

    # Get the task instances.
    pks = deserialize_pks(pks)
    q = session.query(astra.TaskInstance)\
               .filter(astra.TaskInstance.pk.in_(pks))
    
    trees = {}

    for instance in tqdm(q.yield_per(1), total=len(pks)):

        parameters = instance.parameters
        tree = trees.get(parameters["release"], None)
        if tree is None:
            trees[parameters["release"]] = tree = SDSSPath(release=parameters["release"])

        path = tree.full(**parameters)

        # Load the spectrum.
        try:
            spectrum = Spectrum1D.read(path)
        except:
            log.exception(f"Unable to load Spectrum1D from path {path} on task instance {instance}")
            continue

        N, P = spectrum.flux.shape

        if analyze_individual_visits:
            K = N
            results = dict(snr=spectrum.meta["snr"])
        else:
            K = 1
            results = dict(snr=[spectrum.meta["snr"][0]])
        
        for i in range(K):
            # Buld the flux tensor as required.
            flux_tensor = create_flux_tensor(
                flux=spectrum.flux.value[i],
                error=spectrum.uncertainty.array[i]**-0.5,
                device=device,
                num_uncertainty_draws=num_uncertainty_draws,
                large_error=large_error
            )

            # Predict the quantities.
            result = predict(model, flux_tensor)
            for key, value in result.items():
                if i == 0:
                    results.setdefault(key, [])
                
                results[key].append(value)

        # Write the database output.
        create_task_output(
            instance,
            astradb.ApogeeNet,
            **results
        )
        
