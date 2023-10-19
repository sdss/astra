
import warnings
from itertools import combinations

def _get_glossary_parts(value, glossary, delimiter="_"):
    # Just get the glossary terms once so that we don't cause many "No glossary definition for XXX"
    terms = list(filter(lambda x: not x.startswith(delimiter), Glossary.__dict__.keys()))
    parts = value.split(delimiter)
    matches = []
    for i, j in combinations(range(len(parts)), 2):
        if i > 0: break
        a, b = (delimiter.join(parts[i:j]), delimiter.join(parts[j:]))
        if a in terms and b in terms:
            matches.append((a, b))    
    if len(matches) == 0:
        raise ValueError(f"No matches found for '{value}'")
    elif len(matches) > 1:
        raise ValueError(f"Multiple matches found for '{value}': {matches}")
    else:
        return matches[0]
        

def _rho_context(value, glossary):
    try:
        a, b = _get_glossary_parts(value[4:], glossary)
    except ValueError:
        return ""
    else:
        return f"Correlation coefficient between {a.upper()} and {b.upper()}"


SPECIAL_CONTEXTS = (
    ("e_", True, "Error on"),
    ("_flags", False, "Flags for"),
    ("initial_", True, "Initial"),
    ("_rchi2", False, "Reduced chi-square value for"),
    ("raw_", True, "Raw"),
    ("rho_", True, _rho_context)
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

    #> Identifiers
    task_pk = "Task model primary key"
    sdss_id = "SDSS unique source identifier"
    healpix = "Healpix location (128 sides)"
    gaia_dr3_source_id = "Gaia (DR3) source identifier"
    tic_v8_id = "TESS Input Catalog (v8) identifier"
    sdss4_dr17_apogee_id = "SDSS4 DR17 APOGEE identifier (not unique)"
    sdss4_dr17_field = "SDSS4 DR17 APOGEE field (not unique)"
    
    #> Astrometry
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
    ph = "Gaia (DR3) G-band magnitude [mag]"
    bp_mag = "Gaia (DR3) BP-band magnitude [mag]"
    rp_mag = "Gaia (DR3) RP-band magnitude [mag]"

    # 2MASS
    j_mag = "2MASS J-band magnitude magnitude [mag]"
    e_j_mag = "Error on 2MASS J-band magnitude (j_msigcom) [mag]"
    h_mag = "2MASS H-band magnitude magnitude [mag]"
    e_h_mag = "Error on 2MASS H-band magnitude (h_msigcom) [mag]"
    k_mag = "2MASS K-band magnitude magnitude [mag]"
    e_k_mag = "Error on 2MASS K-band magnitude (k_msigcom) [mag]"
    ph_qual = "2MASS Photometric quality flag (see documentation)"
    bl_flg = "2MASS Blending flag (see documentation)"
    cc_flg = "2MASS Contamination and confusion flag (see documentation)"

    # unWISE
    w1_flux = "unWISE W1-band flux [Vega nMgy]"
    w2_flux = "unWISE W2-band flux [Vega nMgy]"
    w1_dflux = "Statistical uncertainty in unWISE W1-band flux"
    w2_dflux = "Statistical uncertainty in unWISE W2-band flux"
    w1_frac = "unWISE W1-band flux fraction from this source (fracflux_w2)"
    w2_frac = "unWISE W2-band flux fraction from this source (fracflux_w2)"
    w1_uflags = "unWISE W1-band coadd flags (flags_unwise)"
    w1_aflags = "Additional W1-band flags (flags_info)"
    w2_uflags = "unWISE W2-band coadd flags (flags_unwise)"
    w2_aflags = "Additional W2-band flags (flags_info)"

    # GLIMPSE
    mag4_5 = "GLIMPSE 4.5um IRAC (Band 2) magnitude [mJy]"
    d4_5m = "GLIMPSE 4.5um IRAC (Band 2) 1 sigma error [mJy]"
    rms_f4_5 = "RMS of detectionss for 4.5um IRAC (Band 2) [mJy]"
    sqf_4_5 = "Source quality flg for 4.5um IRAC (Band 2)"
    mf_4_5 = "Flux calculationi method flag 4.5um IRAC (Band 2)"
    #csf # TODO

    # Wavelength information
    crval = "Reference vacuum wavelength [Angstrom]"
    cdelt = "Vacuum wavelength step [Angstrom]"
    crpix = "Reference pixel (1-indexed)"
    npixels = "Number of pixels in the spectrum"
    ctype = "Wavelength axis type"
    cunit = "Wavelength axis unit"
    dc_flag = "Linear wavelength axis (0) or logarithmic"    

    # Targeting
    carton = "Carton name"
    carton_id = "Simplified carton identifier, NOT the same as `targetdb.carton.pk`"
    carton_0 = "First carton for source (see documentation)"
    carton_flags = "Carton bit field."

    created = "Datetime when task record was created"
    t_overhead = "Estimated core-time spent in overhads [s]"
    t_elapsed = "Core-time elapsed on this analysis [s]"
    tag = "Experiment tag for this result"

    # Spectrum information
    source_pk = "Unique source primary key"
    spectrum_pk = "Unique spectrum primary key"
    snr = "Signal-to-noise ratio"

    #: General data product keyword arguments.
    release = "The SDSS release name."

    # BOSS specFull keyword arguments
    run2d = "BOSS data reduction pipeline version"
    mjd = "Modified Julian Date of observation"
    fieldid = "Field identifier"
    catalogid = "Catalog identifier used to target the source"
    catalogid21 = "Catalog identifier (v21; v0.0)"
    catalogid25 = "Catalog identifier (v25; v0.5)"
    catalogid31 = "Catalog identifier (v31; v1.0)"
    sdss5_target_flags = "SDSS-5 targeting flags"
    n_associated = "SDSS_IDs associated with this CATALOGID"

    f_night_time = "Mid-observation time as a fraction during the night"
    f_night_time = "Mid obs time as fraction from sunset to sunrise"
    
    plateid = "Plate identifier"
    tileid = "C"
    cartid = "Cartridge used for plugging"
    mapid = "Mapping version of the loaded plate"
    slitid = "Slit identifier"
    n_std = "Number of (good) standard stars"
    n_gal = "Number of (good) galaxies in field"

    dust_a = "0.3mu-sized dust count [particles m^-3 s^-1]"
    dust_b = "1.0mu-sized dust count [particles m^-3 s^-1]"
    
    # Pixel arrays
    wavelength = "Wavelength (vacuum) [Angstrom]"
    flux = "Flux [10^-17 erg/s/cm^2/Angstrom]"
    ivar = "Inverse variance of flux values"
    wresl = "Spectral resolution [Angstrom]"
    pixel_flags = "Pixel-level quality flags (see documentation)"
    model_flux = "Best-fit model flux"
    continuum = "Best-fit continuum flux"


    spectrum_flags = "Data reduction pipeline flags for this spectrum"
    result_flags = "Flags describing the results"
    
    plug_ra = "Right ascension of plug position [deg]"
    plug_dec = "Declination of plug position [deg]"

    input_ra = "Input right ascension [deg]"
    input_dec = "Input declination [deg]"
    

    # BOSS data reduction pipeline keywords
    alt = "Telescope altitude [deg]"
    az = "Telescope azimuth [deg]"
    exptime = "Total exposure time [s]"
    n_exp = "Number of co-added exposures"
    airmass = "Mean airmass"
    airtemp = "Air temperature [C]"
    dewpoint = "Dew point temperature [C]"
    humidity = "Humidity [%]"
    pressure = "Air pressure [millibar]"
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
    conscale = "Scale by pseudo-continuum when stacking"
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

    pk = "Database primary key"
    fps = "Fibre positioner used to acquire this data?"

    v_rel = "Relative velocity [km/s]"
    v_rad = "Barycentric rest frame radial velocity [km/s]"
    bc = "Barycentric velocity correction applied [km/s]"
    median_e_v_rad = "Median error in radial velocity [km/s]"

    ccfwhm = "Cross-correlation function FWHM"
    autofwhm = "Auto-correlation function FWHM"
    n_components = "Number of components in CCF"
    
    e_v_rad = "Error on radial velocity [km/s]"

    filetype = "SDSS file type that stores this spectrum"

    # apVisit keywords
    apred = "APOGEE data reduction pipeline version."
    plate = "Plate number of observation."
    telescope = "Telescope used to observe the source."
    field = "Field identifier"
    fiber = "Fiber number."
    prefix = "Short prefix used for DR17 apVisit files"    
    reduction = "An `obj`-like keyword used for apo1m spectra"

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



    teff = "Stellar effective temperature [K]"
    logg = "Surface gravity [log10(cm/s^2)]"
    m_h = "Metallicity [dex]"
    m_h_atm = "Metallicity [dex]"
    alpha_m_atm = "[alpha/M] abundance ratio [dex]"
    v_sini = "Projected rotational velocity [km/s]"
    v_micro = "Microturbulence [km/s]"
    c_m_atm = "Atmospheric carbon abundance [dex]"
    n_m_atm = "Atmospheric nitrogen abundance [dex]"

    v_astra = "Astra version"
    component = "Spectrum component"

    #: ASPCAP-specific keywords
    coarse_id = "Database id of the coarse execution used for initialisation"


    # Elemental abundances
    al_h = "[Al/H] [dex]"
    c_12_13 = "C12/C13 ratio"
    ca_h = "[Ca/H] [dex]"
    ce_h = "[Ce/H] [dex]"
    c_1_h = "[C/H] from neutral C lines [dex]"
    c_h = "[C/H] [dex]"
    co_h = "[Co/H] [dex]"
    cr_h = "[Cr/H] [dex]"
    cu_h = "[Cu/H] [dex]"
    fe_h = "[Fe/H] [dex]"
    k_h = "[K/H] [dex]"
    mg_h = "[Mg/H] [dex]"
    mn_h = "[Mn/H] [dex]"
    na_h = "[Na/H] [dex]"
    nd_h = "[Nd/H] [dex]"
    ni_h = "[Ni/H] [dex]"
    n_h = "[N/H] [dex]"
    o_h = "[O/H] [dex]"
    p_h = "[P/H] [dex]"
    si_h = "[Si/H] [dex]"
    s_h = "[S/H] [dex]"
    ti_h = "[Ti/H] [dex]"
    ti_2_h = "[Ti/H] from singly ionized Ti lines [dex]"
    v_h = "[V/H] [dex]"

    chi2 = "Chi-square value"
    rchi2 = "Reduced chi-square value"
    initial_flags = "Flags indicating source of initial guess"

    # MDwarfType
    spectral_type = "Spectral type"
    sub_type = "Spectral sub-type"

    calibrated = "Any calibration applied to raw measurements?"

    drp_spectrum_pk = "Data Reduction Pipeline spectrum primary key"

    release = "SDSS release"
    apred = "APOGEE reduction pipeline"
    apstar = "Unused DR17 apStar keyword (default: stars)"
    obj = "Object name"
    telescope = "Short telescope name"
    healpix = "HEALPix (128 side)"
    prefix = "Prefix used to separate SDSS 4 north/south"
    plate = "Plate identifier"
    mjd = "Modified Julian date of observation"
    fiber = "Fiber number"

def lower_first_letter(s):
    return f"{s[0].lower()}{s[1:]}"

def resolve_special_contexts(obj, name):
    name_lower = f"{name}".lower()
    for identifier, is_prefix, sub_context in SPECIAL_CONTEXTS:
        if (
            (is_prefix and name_lower.startswith(identifier))
        or  (not is_prefix and name_lower.endswith(identifier))
        ):
            if callable(sub_context):
                return sub_context(name_lower, obj)
            else:
                if is_prefix:
                    # allow for recursive identifiers
                    #value = object.__getattribute__(obj, name_lower[len(identifier):])
                    value = getattr(obj, name_lower[len(identifier):])
                else:
                    #value = object.__getattribute__(obj, name_lower[:-len(identifier)])
                    value = getattr(obj, name_lower[:-len(identifier)])
                
                if value:
                    return f"{sub_context} {lower_first_letter(value)}"
                else:
                    return value
                    
    warnings.warn(f"No glossary definition for '{name}'")
    return ""


def warn_on_long_description(text, max_length=80):
    return text
