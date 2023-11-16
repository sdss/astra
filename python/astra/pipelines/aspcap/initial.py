from astropy.io import fits
from astra.utils import log
from astra.models.apogee import ApogeeVisitSpectrum
from astra.models import ApogeeNetV2 as ApogeeNet
from astra.models.aspcap import FerreCoarse
from astra.pipelines.aspcap.utils import approximate_log10_microturbulence 

from peewee import ModelSelect

def get_effective_fiber(spectrum):
    for field_name in ("fiber", "mean_fiber"):
        fiber = getattr(spectrum, field_name, None)
        if fiber is not None:
            return fiber

    # Try to read it from headers.
    log.warning(f"Getting effective fiber from headers for {spectrum}; these should be ingested to the database")
    return fits.getval(spectrum.absolute_path, "MEANFIB")    

def get_initial_defaults(spectrum, logg=None):
    kwds = {
        "telescope": spectrum.telescope,
        "mean_fiber": get_effective_fiber(spectrum),
        "alpha_m": 0.0,
        "log10_v_sini": 1.0,
        "c_m": 0.0,
        "n_m": 0.0, 
    }
    if logg is not None:
        kwds["log10_v_micro"] = approximate_log10_microturbulence(logg)
    return kwds

def get_initial_guess_from_gaia_xp_zhang_2023(spectrum):
    if spectrum.source.zgr_teff is not None and spectrum.source.zgr_quality_flags == 0:
        initial_guess = {
            "teff": spectrum.source.zgr_teff,
            "logg": spectrum.source.zgr_logg,
            "m_h": spectrum.source.zgr_fe_h,
            "initial_flags": FerreCoarse(flag_initial_guess_from_gaia_xp_zhang_2023=True).initial_flags
        }
        initial_guess.update(get_initial_defaults(spectrum, spectrum.source.zgr_logg))
        return initial_guess
    else:
        return None


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
            .join(ApogeeNet, on=(ApogeeNet.spectrum_pk == spectra.model.spectrum_pk))
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

            # Zhang et al. (2023)
            initial_guess = get_initial_guess_from_gaia_xp_zhang_2023(spectrum)
            if initial_guess is not None:
                yield (spectrum, initial_guess)

            # APOGEENet
            try:
                r = ApogeeNet.get(result_flags=0, spectrum_pk=spectrum.spectrum_pk)
            except ApogeeNet.DoesNotExist:
                # Revert to Doppler by matching to DRP spectrum identifier.
                try:
                    r = (
                        ApogeeVisitSpectrum
                        .select()
                        .where(
                            (ApogeeVisitSpectrum.spectrum_pk == spectrum.drp_spectrum_pk)
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
                            (ApogeeVisitSpectrum.source_pk == spectrum.source_pk)
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
            
            mean_fiber = get_effective_fiber(spectrum)

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