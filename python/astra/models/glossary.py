
from typing import NamedTuple

SPECIAL_CONTEXTS = {
    "e_": "Error on",
    "flag_": "Flag for",
    "initial_": "Initial",
}


class GlossaryType(type):
    
    def __getattribute__(self, name):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
            
        try:
            value = object.__getattribute__(self, name)
        except AttributeError:
            name_lower = f"{name}".lower()
            for prefix, context in SPECIAL_CONTEXTS.items():
                if name_lower.startswith(prefix):
                    value = object.__getattribute__(self, name_lower[len(prefix):])
                    return " ".join([context.rstrip(), f"{value[0].lower()}{value[1:]}"])
            raise AttributeError(f"Glossary has no attribte '{name}'")        
        else:
            if name == "context":
                return value
            else:
                print(f"Here {name} {value}")
                if hasattr(self, "context"):
                    value = f"{self.context} {value[0].lower()}{value[1:]}"
                return value

class BaseGlossary(NamedTuple):

    context: str = ""




class Glossary(BaseGlossary):

    #: Identifiers
    source_id = "Unique identifier for a source."
    healpix = "Healpix location (128 sides)"
    gaia_dr3_source_id = "Gaia (DR3) source identifier"
    tic_v8_id = "TESS Input Catalog (v8) identifier"
    sdss4_dr17_apogee_id = "SDSS4 DR17 APOGEE identifier (not unique)"
    sdss4_dr17_field = "SDSS4 DR17 APOGEE field (not unique)"
    
    #: Astrometry
    ra = "SDSS-V catalog right ascension (J2000) [deg]"
    dec = "SDSS-V catalog declination (J2000) [deg]"
    plx = "Parallax [mas] (Gaia DR3)"
    e_plx = "Error on parallax [mas] (Gaia DR3)"
    pmra = "Proper motion in RA [mas/yr] (Gaia DR3)"
    e_pmra = "Error on proper motion in RA [mas/yr] (Gaia DR3)"
    pmde = "Proper motion in DEC [mas/yr] (Gaia DR3)"
    e_pmde = "Error on proper motion in DEC [mas/yr] (Gaia DR3)"
    gaia_v_rad = "Radial velocity [km/s] (Gaia DR3)"
    gaia_e_v_rad = "Error on radial velocity [km/s] (Gaia DR3)"

    #: Photometry
    g_mag = "Gaia (DR3) mean apparent G [mag]"
    bp_mag = "Gaia (DR3) mean apparent BP [mag]"
    rp_mag = "Gaia (DR3) mean apparent RP [mag]"
    j_mag = "2MASS mean apparent J magnitude [mag]"
    e_j_mag = "Error on 2MASS mean apparent J magnitude [mag]"
    h_mag = "2MASS mean apparent H magnitude [mag]"
    e_h_mag = "Error on 2MASS mean apparent H magnitude [mag]"
    k_mag = "2MASS mean apparent K magnitude [mag]"
    e_k_mag = "Error on 2MASS mean apparent K magnitude [mag]"

    #: Targeting
    carton_0 = "First carton for source (see documentation)"
    carton_flags = "Carton bit field."

    #: Spectrum information
    spectrum_id = "Unique identifier for a spectrum."
    snr = "Signal-to-noise ratio"

    #: General data product keyword arguments.
    release = "The SDSS release name."

    # BOSS specFull keyword arguments
    run2d = "BOSS data reduction pipeline version."
    mjd = "Modified Julian Date of observation."
    fieldid = "Field identifier."
    catalogid = "Catalog identifier used to target the source."




    # BOSS data reduction pipeline keywords
    alt = "Telescope altitude [deg]"
    az = "Telescope azimuth [deg]"
    exptime = "Total exposure time [s]"
    nexp = "Number of exposures taken"
    airmass = "Mean airmass"
    airtemp = "Air temperature [C]"
    dewpoint = "Dew point temperature [C]"
    humidity = "Humidity [%]"
    pressure = "Air pressure [inch Hg?]"
    moon_phase_mean = "Mean phase of the moon"
    moon_dist_mean = "Mean sky distance to the moon [deg]"
    seeing = "Median seeing conditions [arcsecond]"
    gustd = "Wind gust direction [deg]"
    gusts = "Wind gust speed [km/s]"
    windd = "Wind direction [deg]"
    winds = "Wind speed [km/s]"
    tai_beg = "MJD (TAI) at start of integrations [s]"
    tai_end = "MJD (TAI) at end of integrations [s]"
    nguide = "Number of guider frames during integration"
    v_helio = "Heliocentric velocity correction [km/s]"
    v_shift = "Relative velocity shift used in stack [km/s]"
    in_stack = "Was this spectrum used in the stack?"
    nres = "Sinc bandlimit [pixel/resolution element]"
    filtsize = "Median filter size for pseudo-continuum [pixel]"
    normsize = "Gaussian width for pseudo-continuum [pixel]"
    conscale = "Scale by pseudo-continuuwhen stacking"
    v_boss = "Version of the BOSS ICC"
    v_jaeger = "Version of Jaeger"
    v_kaiju = "Version of Kaiju"
    v_coord = "Version of coordIO"
    v_calibs = "Version of FPS calibrations"
    v_read = "Version of idlspec2d for processing raw data"
    v_idl = "Version of IDL"
    v_util = "Version of idlutils"
    v_2d = "Version of idlspec2d for 2D reduction"
    v_comb = "Version of idlspec2d for combining exposures"
    v_log = "Version of SPECLOG product"
    v_flat = "Version of SPECFLAT product"
    didflush = "Was CCD flushed before integration"
    cartid = "Cartridge identifier"
    psfsky = "Order of PSF sky subtraction"
    preject = "Profile area rejection threshold"
    lowrej = "Extraction: low rejection"
    highrej = "Extraction: high rejection"
    scatpoly = "Extraction: Order of scattered light polynomial"
    proftype = "Extraction profile: 1=Gaussian"
    nfitpoly = "Extraction: Number of profile parameters"
    rdnoise0 = "CCD read noise amp 0 [electrons]"
    skychi2 = "Mean \chi^2 of sky subtraction"
    schi2min = "Minimum \chi^2 of sky subtraction"
    schi2max = "Maximum \chi^2 of sky subtraction"
    zwarning = "See sdss.org/dr17/algorithms/bitmasks/#ZWARNING"
    fiber_offset = "Position offset applied during observations"
    delta_ra = "Offset in right ascension [arcsecond]"
    delta_dec = "Offset in declination [arcsecond]"
    date_obs = "Observation date (UTC)"

    # apVisit keywords
    apred = "APOGEE data reduction pipeline version."
    plate = "Plate number of observation."
    telescope = "Telescope used to observe the source."
    field = "Field name."
    fiber = "Fiber number."
    prefix = "Short prefix used for DR17 apVisit files."    

    # APOGEE data reduction pipeline keywords    
    v_apred = "APOGEE Data Reduction Pipeline version"
    jd = "Julian date at mid-point of visit"
    ut_mid = "Date at mid-point of visit"

    fluxflam = "ADU to flux conversion factor [ergs/s/cm^2/A]"
    n_pairs = "Number of dither pairs combined"
    dithered = "Fraction of visits that were dithered"
    nvisits = "Number of visits included in the stack"
    nvisits_apstar = "Number of visits included in the apStar file"

    on_target = "FPS fiber on target"
    assigned = "FPS target assigned"
    valid = "Valid FPS target"
    n_frames = "Number of frames combined"
    exptime = "Exposure time [s]"

    apvisit_pk = "Primary key of `apogee_drp.visit` database table"


    initial_teff = "Initial stellar effective temperature [K]"
    initial_logg = "Initial stellar surface gravity [dex]"
    initial_fe_h = "Initial stellar metallicity [dex]"
    teff = "Stellar effective temperature [K]"
    logg = "Surface gravity [log10(cm/s^2)]"
    fe_h = "Metallicity [dex]"
    metals = "Metallicity [dex]"
    e_metals = "Error in metallicity [dex]"
    o_mg_si_s_ca_ti = "[alpha/Fe] abundance ratio [dex]"
    e_o_mg_si_s_ca_ti = "Error in [alpha/Fe] abundance ratio [dex]"
    log10vdop = "Log10 of the doppler broadening [km/s]"
    e_log10vdop = "Error in the log10 doppler broadening [km/s]"
    lgvsini = "Log of the projected rotational velocity [km/s]"
    e_lgvsini = "Error in the log projected rotational velocity [km/s]"
    c_h_photosphere = "Photosphere carbon abundance [dex]"
    e_c_h_photosphere = "Error on photosphere carbon abundance [dex]"
    n_h_photosphere = "Photosphere nitrogen abundance [dex]"
    e_n_photosphere = "Error on photosphere nitrogen abundance [dex]"

    v_astra = "Version of Astra"
    component = "Spectrum component"

    #: ASPCAP-specific keywords
    coarse_id = "Database id of the coarse execution used for initialisation"


if __name__ == "__main__":

    Glossary.e_plx
