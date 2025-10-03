from astra import __version__
from astra.utils import version_string_to_integer
from astra.models.apogeenet import ApogeeNet
from astra.models.aspcap import FerreCoarse
from astra.pipelines.aspcap.utils import approximate_log10_microturbulence 
from astropy.io import fits

def get_initial_guesses(spectra):
    """
    Get an initial guess of the stellar parameters for ASPCAP, given some spectra.

    :param spectra:
        An iterable of spectra.
    """
    
    spectra_dict = {s.spectrum_pk: s for s in spectra}

    functions = (
        get_initial_guesses_from_apogeenet, 
        get_initial_guess_from_gaia_xp_zhang_2023
    )
    for fun in functions:
        yield from fun(spectra_dict)


def get_effective_fiber(spectrum):
    for field_name in ("fiber", "mean_fiber"):
        fiber = getattr(spectrum, field_name, None)
        if fiber is not None:
            return fiber

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

def get_initial_guess_from_gaia_xp_zhang_2023(spectra_pk_dict):
    for pk in list(spectra_pk_dict.keys()):
        spectrum = spectra_pk_dict[pk]
        if spectrum.source.zgr_teff is not None and spectrum.source.zgr_quality_flags == 0:
            try:
                initial_guess = {
                    "teff": spectrum.source.zgr_teff,
                    "logg": spectrum.source.zgr_logg,
                    "m_h": spectrum.source.zgr_fe_h,
                    "initial_flags": FerreCoarse(flag_initial_guess_from_gaia_xp_zhang_2023=True).initial_flags
                }
                initial_guess.update(get_initial_defaults(spectrum, spectrum.source.zgr_logg))
                spectra_pk_dict.pop(pk)
                yield (spectrum, initial_guess)
            except:
                continue


def get_initial_guesses_from_apogeenet(spectra_pk_dict):

    current_version = version_string_to_integer(__version__) // 1000

    q = (
        ApogeeNet
        .select(
            ApogeeNet.spectrum_pk,
            ApogeeNet.teff.alias("initial_teff"),
            ApogeeNet.logg.alias("initial_logg"),
            ApogeeNet.fe_h.alias("initial_fe_h"),
        )
        .where(
            (ApogeeNet.spectrum_pk.in_(list(spectra_pk_dict.keys())))
        &   (ApogeeNet.v_astra_major_minor == current_version)
        &   (ApogeeNet.teff.is_null(False))
        )
        .tuples()
    )

    for spectrum_pk, initial_teff, initial_logg, initial_fe_h in q:
        spectrum = spectra_pk_dict.pop(spectrum_pk)
        try:
            mean_fiber = get_effective_fiber(spectrum)
        except:
            None
        else:
            initial_guess = {
                "teff": initial_teff,
                "logg": initial_logg,
                "m_h": initial_fe_h,
                "initial_flags": FerreCoarse(flag_initial_guess_from_apogeenet=True).initial_flags,
                "mean_fiber": mean_fiber,
                "telescope": spectrum.telescope,
                "alpha_m": 0.0,
                "log10_v_sini": 1.0,
                "c_m": 0.0,
                "n_m": 0.0, 
            }
            initial_guess["log10_v_micro"] = approximate_log10_microturbulence(initial_guess["logg"])
            yield (spectrum, initial_guess)
