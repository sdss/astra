from astropy.io import fits
from astra.utils import log
from astra.models.apogee import ApogeeVisitSpectrum
from astra.models.apogeenet import ApogeeNet
from astra.models.aspcap import FerreCoarse
from astra.pipelines.aspcap.utils import approximate_log10_microturbulence 

from peewee import ModelSelect


def get_initial_guesses(spectra):
    """
    Get an initial guess of the stellar parameters for ASPCAP, given some spectra.

    :param spectra:
        An iterable of spectra.
    """

    if False and isinstance(spectra, ModelSelect):
        q = (
            spectra
            .select_from(
                ApogeeNet.teff.alias("initial_teff"),
                ApogeeNet.logg.alias("initial_logg"),
                ApogeeNet.fe_h.alias("initial_fe_h"),
            )
            .join(ApogeeNet, on=(ApogeeNet.spectrum_id == spectra.model.spectrum_id))
            .where(ApogeeNet.result_flags == 0)
        )

        for spectrum in q:
            initial_guess = {
                "teff": spectrum.initial_teff,
                "logg": spectrum.initial_logg,
                "m_h": spectrum.initial_fe_h,
                "initial_flags": FerreCoarse(flag_initial_guess_from_apogeenet=True).initial_flags,
                "mean_fiber": fits.getval(spectrum.absolute_path, "MEANFIB"),
                "telescope": spectrum.telescope,
                "alpha_m": 0.0,
                "log10_v_sini": 1.0,
                "c_m": 0.0,
                "n_m": 0.0, 
            }
            initial_guess["log10_v_micro"] = approximate_log10_microturbulence(initial_guess["logg"])
            yield (spectrum, initial_guess)

    else:         
        for spectrum in spectra:

            # APOGEENet
            try:
                r = ApogeeNet.get(result_flags=0, spectrum_id=spectrum.spectrum_id)
            except ApogeeNet.DoesNotExist:
                # Revert to Doppler by matching to DRP spectrum identifier.
                try:
                    r = (
                        ApogeeVisitSpectrum
                        .select()
                        .where(
                            (ApogeeVisitSpectrum.spectrum_id == spectrum.drp_spectrum_id)
                        &   (ApogeeVisitSpectrum.doppler_teff >= 2000)
                        &   (ApogeeVisitSpectrum.doppler_teff <= 100_000)
                        )
                        .first()
                    )
                except ApogeeVisitSpectrum.DoesNotExist:
                    raise
                except:  
                    # Get Doppler result for anything matching by source.              
                    r = (
                        ApogeeVisitSpectrum
                        .select()
                        .where(
                            (ApogeeVisitSpectrum.source_id == spectrum.source_id)
                        &   (ApogeeVisitSpectrum.doppler_teff >= 2000)
                        &   (ApogeeVisitSpectrum.doppler_teff <= 100_000)
                        )
                        .first()
                    )
                finally:
                    if r is None:
                        # Try load it from the headers maybe
                        continue
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
            
            try:
                mean_fiber = spectrum.mean_fiber
            except AttributeError:
                # Probably a coadded spectrum. We might need to read it from the headers.
                try:
                    mean_fiber = fits.getval(spectrum.absolute_path, "MEANFIB")
                except:
                    continue

            initial_guess = {
                "telescope": spectrum.telescope,
                "mean_fiber": mean_fiber,
                "alpha_m": 0.0,
                "log10_v_sini": 1.0,
                "c_m": 0.0,
                "n_m": 0.0, 
            }
            initial_guess.update(params)
            initial_guess["log10_v_micro"] = approximate_log10_microturbulence(initial_guess["logg"])

            yield (spectrum, initial_guess)