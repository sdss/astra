SPECIAL_CONTEXTS = (
    ("e_", True, "Error on"),
    ("_flags", False, "Flags for"),
    ("initial_", True, "Initial"),
)


class GlossaryType(type):
    
    def __getattribute__(self, name):
        if __name__.startswith("__") or name == "context":
            return object.__getattribute__(self, name)
        try:
            value = object.__getattribute__(self, name)
        except AttributeError:
            value = resolve_special_contexts(self, name)
        return warn_on_long_description(value)


class BaseGlossary(object, metaclass=GlossaryType):

    def __init__(self, context=None, *args):
        self.context = context or ""
        return None
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None
    
    def __getattribute__(self, name: str):
        if name.startswith("__") or name == "context":
            return object.__getattribute__(self, name)    
        value = object.__getattribute__(self, name)
        description = f"{self.context} {lower_first_letter(value)}"
        return warn_on_long_description(description)
        
    def __getattr__(self, name):
        value = resolve_special_contexts(self, name)
        description = f"{self.context} {lower_first_letter(value)}"
        return warn_on_long_description(description)
    

class Glossary(BaseGlossary):

    # Identifiers
    source_id = "Unique identifier for a source."
    healpix = "Healpix location (128 sides)"
    gaia_dr3_source_id = "Gaia (DR3) source identifier"
    tic_v8_id = "TESS Input Catalog (v8) identifier"
    sdss4_dr17_apogee_id = "SDSS4 DR17 APOGEE identifier (not unique)"
    sdss4_dr17_field = "SDSS4 DR17 APOGEE field (not unique)"
    
    # Astrometry
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

    # Photometry
    g_mag = "Gaia (DR3) mean apparent G [mag]"
    bp_mag = "Gaia (DR3) mean apparent BP [mag]"
    rp_mag = "Gaia (DR3) mean apparent RP [mag]"
    j_mag = "2MASS mean apparent J magnitude [mag]"
    e_j_mag = "Error on 2MASS mean apparent J magnitude [mag]"
    h_mag = "2MASS mean apparent H magnitude [mag]"
    e_h_mag = "Error on 2MASS mean apparent H magnitude [mag]"
    k_mag = "2MASS mean apparent K magnitude [mag]"
    e_k_mag = "Error on 2MASS mean apparent K magnitude [mag]"

    # Targeting
    carton = "Carton name"
    carton_id = "Simplified carton identifier, NOT the same as `targetdb.carton.pk`"
    carton_0 = "First carton for source (see documentation)"
    carton_flags = "Carton bit field."

    sdss4_apogee_target1_flags = "SDSS4 APOGEE1 targeting bitfield (1 of 2)"
    sdss4_apogee_target2_flags = "SDSS4 APOGEE1 targeting bitfield (2 of 2)"
    sdss4_apogee2_target1_flags = "SDSS4 APOGEE2 targeting bitfield (1 of 3)"
    sdss4_apogee2_target2_flags = "SDSS4 APOGEE2 targeting bitfield (2 of 3)"
    sdss4_apogee2_target3_flags = "SDSS4 APOGEE2 targeting bitfield (3 of 3)"

    sdss4_apogee_member_flags = "SDSS4 flags to identify likely members of clusters/dwarf galaxies"
    sdss4_extra_target_flags = "SDSS4 basic targeting information (formerly EXTRATARG)"


    # Spectrum information
    spectrum_id = "Unique identifier for a spectrum."
    snr = "Signal-to-noise ratio"

    #: General data product keyword arguments.
    release = "The SDSS release name."

    # BOSS specFull keyword arguments
    run2d = "BOSS data reduction pipeline version."
    mjd = "Modified Julian Date of observation."
    fieldid = "Field identifier."
    catalogid = "Catalog identifier used to target the source."

    # Pixel arrays
    wavelength = "Wavelength in a vacuum [Angstrom]"
    flux = "Flux [10^-17 erg/s/cm^2/Angstrom]"
    ivar = "Inverse variance of flux [1/(10^-17 erg/s/cm^2/Angstrom)^2]"
    pixel_flags = "Pixel-level bitfield flags (see documentation)."


    spectrum_flags = "Data reduction pipeline flags for this spectrum."



    # BOSS data reduction pipeline keywords
    alt = "Telescope altitude [deg]"
    az = "Telescope azimuth [deg]"
    exptime = "Total exposure time [s]"
    n_exp = "Number of exposures taken"
    airmass = "Mean airmass"
    airtemp = "Air temperature [C]"
    dewpoint = "Dew point temperature [C]"
    humidity = "Humidity [%]"
    pressure = "Air pressure [inch Hg?]"
    moon_phase_mean = "Mean phase of the moon"
    moon_dist_mean = "Mean sky distance to the moon [deg]"
    seeing = "Median seeing conditions [arcsecond]"
    gust_direction = "Wind gust direction [deg]"
    gust_speed = "Wind gust speed [km/s]"
    wind_direction = "Wind direction [deg]"
    wind_speed = "Wind speed [km/s]"
    tai_beg = "MJD (TAI) at start of integrations [s]"
    tai_end = "MJD (TAI) at end of integrations [s]"
    n_guide = "Number of guider frames during integration"
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


    rxc = "Cross-correlation R-value (1979AJ.....84.1511T)"


    teff = "Stellar effective temperature [K]"
    logg = "Surface gravity [log10(cm/s^2)]"
    fe_h = "Metallicity [dex]"
    metals = "Metallicity [dex]"
    o_mg_si_s_ca_ti = "[alpha/Fe] abundance ratio [dex]"
    log10vdop = "Log10 of the doppler broadening [km/s]"
    lgvsini = "Log of the projected rotational velocity [km/s]"
    c_h_photosphere = "Photosphere carbon abundance [dex]"
    n_h_photosphere = "Photosphere nitrogen abundance [dex]"

    v_astra = "Version of Astra"
    component = "Spectrum component"

    #: ASPCAP-specific keywords
    coarse_id = "Database id of the coarse execution used for initialisation"





def lower_first_letter(s):
    return f"{s[0].lower()}{s[1:]}"

def resolve_special_contexts(obj, name):
    name_lower = f"{name}".lower()
    for identifier, is_prefix, sub_context in SPECIAL_CONTEXTS:
        if is_prefix and name_lower.startswith(identifier):
            value = object.__getattribute__(obj, name_lower[len(identifier):])
            return f"{sub_context} {lower_first_letter(value)}"
        if not is_prefix and name_lower.endswith(identifier):
            value = object.__getattribute__(obj, name_lower[:-len(identifier)])
            return f"{sub_context} {lower_first_letter(value)}"        
    raise AttributeError(f"Glossary has no attribute '{name}'")


def warn_on_long_description(text, max_length=80):
    return text
