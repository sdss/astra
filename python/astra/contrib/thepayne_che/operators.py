import multiprocessing as mp
import numpy as np

from astra.database import astradb, session
from astra.utils import log

from astra.operators import ApStarOperator, BossSpecOperator

from astra.contrib.thepayne_che.network import Network
from astra.contrib.thepayne_che.fitting import Fit
from astra.contrib.thepayne_che.uncertfit import UncertFit
from astra.contrib.thepayne_che.lsf import LSF_Fixed_R

from astra.database.utils import create_task_output, deserialize_pks
from astra.operators.utils import prepare_data

import json


class ThePayneCheOperator(ApStarOperator):

    template_fields = ("network_path", "psf_path")

    def __init__(
        self,
        network_path: str,
        N_chebyshev: int,
        spectral_resolution: int,
        N_pre_search_iter: int,
        N_pre_search: int,
        constraints=None,
        **kwargs,
    ) -> None:
        super().__init__(bash_command="astra run thepayne-che", **kwargs)
        self.network_path = network_path
        self.N_chebyshev = N_chebyshev
        self.spectral_resolution = spectral_resolution
        self.N_pre_search_iter = N_pre_search_iter
        self.N_pre_search = N_pre_search
        self.constraints = constraints


def _estimate_stellar_labels(pk):

    # TODO: It would be great if these were stored with the network,
    #       instead of being hard-coded.
    label_names = ["teff", "logg", "vsini", "v_micro", "m_h"]
    # Translate:
    _t = {
        "teff": "T_eff",
        "logg": "log(g)",
        "m_h": "[M/H]",
        "vsini": "v*sin(i)",
    }

    # TODO: This implicitly assumes that the same constraints and network path are used by all the
    #       primary keys given. This is the usual case, but we should check this, and code around it.

    # TODO: This implementation requires knowing the observed spectrum before loading data.
    #       This is fine for ApStar objects since they all have the same dispersion sampling,
    #       but will not be fine for dispersion sampling that differs in each observation.

    # Let's peak ahead at the first valid spectrum we can find.
    instance, _, spectrum = next(prepare_data([pk]))
    if spectrum is None:
        # No valid spectrum.
        log.warning(
            f"Cannot build LSF for fitter because no spectrum found for primary key {pk}"
        )
        return None

    network = Network()
    network.read_in(instance.parameters["network_path"])

    constraints = json.loads(instance.parameters.get("constraints", "{}"))
    fitted_label_names = [
        ln
        for ln in label_names
        if network.grid[_t.get(ln, ln)][0] != network.grid[_t.get(ln, ln)][1]
    ]
    L = len(fitted_label_names)

    bounds_unscaled = np.zeros((2, L))
    for i, ln in enumerate(fitted_label_names):
        bounds_unscaled[:, i] = constraints.get(ln, network.grid[_t.get(ln, ln)][:2])

    fit = Fit(network, int(instance.parameters["N_chebyshev"]))
    fit.bounds_unscaled = bounds_unscaled

    spectral_resolution = int(instance.parameters["spectral_resolution"])
    fit.lsf = LSF_Fixed_R(spectral_resolution, spectrum.wavelength.value, network.wave)

    # Note the Stramut code uses inconsistent naming for "presearch", but in the operator interface we use
    # 'pre_search' in all situations. That's why there is some funny naming translation here.
    fit.N_presearch_iter = int(instance.parameters["N_pre_search_iter"])
    fit.N_pre_search = int(instance.parameters["N_pre_search"])

    fitter = UncertFit(fit, spectral_resolution)
    N, P = spectrum.flux.shape

    keys = []
    keys.extend(fitted_label_names)
    keys.extend([f"u_{ln}" for ln in fitted_label_names])
    keys.extend(["v_rad", "u_v_rad", "chi2", "theta"])

    result = {key: [] for key in keys}
    result["snr"] = spectrum.meta["snr"]

    model_fluxes = []
    log.info(f"Running ThePayne-Che on {N} spectra for {instance}")

    for i in range(N):

        flux = spectrum.flux.value[i]
        error = spectrum.uncertainty.array[0] ** -0.5

        # TODO: No NaNs/infs are allowed, but it doesn't seem like that was an issue for Stramut's code.
        #       Possibly due to different versions of scipy. In any case, raise this as a potential bug,
        #       since the errors do not always seem to be believed by ThePayne-Che.
        bad = (~np.isfinite(flux)) | (error <= 0)
        flux[bad] = 0
        error[bad] = 1e10

        fit_result = fitter.run(
            spectrum.wavelength.value,
            flux,
            error,
        )

        # The `popt` attribute is length: len(label_names) + 1 (for radial velocity) + N_chebyshev

        # Relevent attributes are:
        # - fit_result.popt
        # - fit_result.uncert
        # - fit_result.RV_uncert
        # - fit_result.model

        for j, label_name in enumerate(fitted_label_names):
            result[label_name].append(fit_result.popt[j])
            result[f"u_{label_name}"].append(fit_result.uncert[j])

        result["theta"].append(fit_result.popt[L + 1 :].tolist())
        result["chi2"].append(fit_result.chi2_func(fit_result.popt))
        result["v_rad"].append(fit_result.popt[L])
        result["u_v_rad"].append(fit_result.RV_uncert)

        model_fluxes.append(fit_result.model)

    # Write database result.
    create_task_output(instance, astradb.ThePayneChe, **result)

    # TODO: Write AstraSource object here.
    return None


def estimate_stellar_labels(pks, processes=32):

    pks = deserialize_pks(pks, flatten=True)

    in_parallel = processes != 1
    if in_parallel:
        with mp.Pool(processes=processes) as pool:
            pool.map(_estimate_stellar_labels, pks)

    else:
        for pk in pks:
            _estimate_stellar_labels(pk)
