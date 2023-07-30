from astra.models.apogee import ApogeeVisitSpectrum
from astra.models.apogeenet import ApogeeNet
from astra.models.aspcap import FerreCoarse
from astra.pipelines.aspcap.utils import approximate_log10_microturbulence 


def get_initial_guesses(spectra):
    """
    Get an initial guess of the stellar parameters for ASPCAP, given some spectra.

    :param spectra:
        An iterable of spectra.
    """

    for spectrum in spectra:

        # APOGEENet
        try:
            r = ApogeeNet.get(
                spectrum_id=spectrum.spectrum_id,
                result_flags=0
            )
        except ApogeeNet.DoesNotExist:
            # Revert to Doppler
            try:
                r = ApogeeVisitSpectrum.get(spectrum_id=spectrum.drp_spectrum_id)
            except ApogeeVisitSpectrum.DoesNotExist:
                raise
            else:
                params = {
                    "teff": r.doppler_teff,
                    "logg": r.doppler_logg,
                    "m_h": r.doppler_fe_h,
                    "initial_flags": FerreCoarse(flag_initial_guess_from_doppler=True).initial_flags
                }
        else:
            params = {
                "teff": r.teff,
                "logg": r.logg,
                "m_h": r.fe_h,
                "initial_flags": FerreCoarse(flag_initial_guess_from_apogeenet=True).initial_flags
            }
        
        initial_guess = {
            "telescope": spectrum.telescope,
            "mean_fiber": spectrum.fiber,
            "alpha_m": 0.0,
            "log10_v_sini": 1.0,
            "c_m": 0.0,
            "n_m": 0.0, 
        }
        initial_guess.update(params)
        initial_guess["log10_v_micro"] = approximate_log10_microturbulence(initial_guess["logg"])

        yield (spectrum, initial_guess)