
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


    sdss4_apogee_target1_flags = "SDSS4 APOGEE1 targeting flags (1/2)"
    flag_sdss4_apogee_faint = "Star selected in faint bin of its cohort"


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
    v_macro = "Macroscopic broadening [km/s]"
    c_m_atm = "Atmospheric carbon abundance [dex]"
    n_m_atm = "Atmospheric nitrogen abundance [dex]"
    
    covar = "Covariance matrix (flattened)"

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
    he_h = "[He/H] [dex]"
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

    alpha_fe = "[alpha/fe] [dex]"
    fe_h_niu = "[Fe/H] on Niu et al. (2023) scale [dex]"

    chi2 = "Chi-square value"
    rchi2 = "Reduced chi-square value"
    initial_flags = "Flags indicating source of initial guess"

    nmf_rchi2 = "Reduced chi-square value of NMF continuum fit"
    nmf_rectified_model_flux = "Rectified NMF model flux"

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

    # AutoGlossary terms below
    # Source table:
    pk = "Database primary key"
    sdss_id = "SDSS-5 unique identifier"
    sdss4_apogee_id = "SDSS-4 DR17 APOGEE identifier"
    gaia_dr2_source_id = "Gaia DR2 source identifier"
    gaia_dr3_source_id = "Gaia DR3 source identifier"
    tic_v8_id = "TESS Input Catalog (v8) identifier"
    healpix = "HEALPix (128 side)"
    lead = "Lead catalog used for cross-match"
    version_id = "SDSS catalog version for targeting"
    catalogid = "Catalog identifier used to target the source"
    catalogid21 = "Catalog identifier (v21; v0.0)"
    catalogid25 = "Catalog identifier (v25; v0.5)"
    catalogid31 = "Catalog identifier (v31; v1.0)"
    n_associated = "SDSS_IDs associated with this CATALOGID"
    n_neighborhood = "Sources within 3\" and G_MAG < G_MAG_source + 5"
    sdss5_target_flags = "SDSS-5 targeting flags"
    sdss4_apogee_target1_flags = "SDSS4 APOGEE1 targeting flags (1/2)"
    sdss4_apogee_target2_flags = "SDSS4 APOGEE1 targeting flags (2/2)"
    sdss4_apogee2_target1_flags = "SDSS4 APOGEE2 targeting flags (1/3)"
    sdss4_apogee2_target2_flags = "SDSS4 APOGEE2 targeting flags (2/3)"
    sdss4_apogee2_target3_flags = "SDSS4 APOGEE2 targeting flags (3/3)"
    sdss4_apogee_member_flags = "SDSS4 likely cluster/galaxy member flags"
    sdss4_apogee_extra_target_flags = "SDSS4 target info (aka EXTRATARG)"
    flag_sdss4_apogee_faint = "Star selected in faint bin of its cohort"
    flag_sdss4_apogee_medium = "Star selected in medium bin of its cohort"
    flag_sdss4_apogee_bright = "Star selected in bright bin of its cohort"
    flag_sdss4_apogee_irac_dered = "Selected w/ RJCE-IRAC dereddening"
    flag_sdss4_apogee_wise_dered = "Selected w/ RJCE-WISE dereddening"
    flag_sdss4_apogee_sfd_dered = "Selected w/ SFD dereddening"
    flag_sdss4_apogee_no_dered = "Selected w/ no dereddening"
    flag_sdss4_apogee_wash_giant = "Selected as giant using Washington photometry"
    flag_sdss4_apogee_wash_dwarf = "Selected as dwarf using Washington photometry"
    flag_sdss4_apogee_sci_cluster = "Selected as probable cluster member"
    flag_sdss4_apogee_extended = "Extended object"
    flag_sdss4_apogee_short = "Selected as 'short' (~3 visit) cohort target (includes 1-visit samples)"
    flag_sdss4_apogee_intermediate = "Selected as 'intermediate' cohort (~6-visit) target"
    flag_sdss4_apogee_long = "Selected as 'long' cohort (~12- or 24-visit) target"
    flag_sdss4_apogee_do_not_observe = "Do not observe (again) -- undesired dwarf, galaxy, etc"
    flag_sdss4_apogee_serendipitous = "Serendipitous interesting target to be re-observed"
    flag_sdss4_apogee_first_light = "First Light plate target"
    flag_sdss4_apogee_ancillary = "Ancillary target"
    flag_sdss4_apogee_m31_cluster = "M31 Clusters (Allende Prieto, Schiavon, Bizyaev, OConnell, Shetrone)"
    flag_sdss4_apogee_mdwarf = "RVs of M Dwarfs (Blake, Mahadevan, Hearty, Deshpande, Nidever, Bender, Crepp, Carlberg, Terrien, Schneider) -- both original list and second-round extension"
    flag_sdss4_apogee_hires = "Stars with Optical Hi-Res Spectra (Fabbian, Allende Prieto, Smith, Cunha)"
    flag_sdss4_apogee_old_star = "Oldest Stars in Galaxy (Harding, Johnson)"
    flag_sdss4_apogee_disk_red_giant = "Ages/Compositions? of Disk Red Giants (Johnson, Epstein, Pinsonneault, Lai, Bird, Schonrich, Chiappini)"
    flag_sdss4_apogee_kepler_eb = "Kepler EBs (Mahadevan, Fleming, Bender, Deshpande, Hearty, Nidever, Terrien)"
    flag_sdss4_apogee_gc_pal1 = "Globular Cluster Pops in the MW (Simmerer, Ivans, Shetrone)"
    flag_sdss4_apogee_massive_star = "Massive Stars in the MW (Herrero, Garcia-Garcia, Ramirez-Alegria)"
    flag_sdss4_apogee_sgr_dsph = "Sgr (dSph) member"
    flag_sdss4_apogee_kepler_seismo = "Kepler asteroseismology program target (Epstein)"
    flag_sdss4_apogee_kepler_host = "Planet-host program target (Epstein)"
    flag_sdss4_apogee_faint_extra = "'Faint' target in low-target-density fields"
    flag_sdss4_apogee_segue_overlap = "SEGUE overlap"
    flag_sdss4_apogee_light_trap = "Light trap"
    flag_sdss4_apogee_flux_standard = "Flux standard"
    flag_sdss4_apogee_standard_star = "Stellar abundance/parameters standard"
    flag_sdss4_apogee_rv_standard = "RV standard"
    flag_sdss4_apogee_sky = "Sky"
    flag_sdss4_apogee_sky_bad = "Selected as sky but IDed as bad (via visual examination or observation)"
    flag_sdss4_apogee_guide_star = "Guide star"
    flag_sdss4_apogee_bundle_hole = "Bundle hole"
    flag_sdss4_apogee_telluric_bad = "Selected as telluric std but IDed as too red (via SIMBAD or observation)"
    flag_sdss4_apogee_telluric = "Targeted as telluric"
    flag_sdss4_apogee_calib_cluster = "Known calibration cluster member"
    flag_sdss4_apogee_bulge_giant = "Selected as probable giant in bulge"
    flag_sdss4_apogee_bulge_super_giant = "Selected as probable supergiant in bulge"
    flag_sdss4_apogee_embedded_cluster_star = "Young nebulous clusters (Covey, Tan)"
    flag_sdss4_apogee_long_bar = "Milky Way Long Bar (Zasowski)"
    flag_sdss4_apogee_emission_star = "Be Emission Line Stars (Chojnowski, Whelan)"
    flag_sdss4_apogee_kepler_cool_dwarf = "Kepler Cool Dwarfs (van Saders)"
    flag_sdss4_apogee_mir_cluster_star = "Outer Disk MIR Clusters (Beaton)"
    flag_sdss4_apogee_rv_monitor_ic348 = "RV Variability in IC348 (Nidever, Covey)"
    flag_sdss4_apogee_rv_monitor_kepler = "RV Variability for Kepler Planet Hosts and Binaries (Deshpande, Fleming, Mahadevan)"
    flag_sdss4_apogee_ges_calibrate = "Gaia-ESO calibration targets"
    flag_sdss4_apogee_bulge_rv_verify = "RV Verification (Nidever)"
    flag_sdss4_apogee_1m_target = "Selected as a 1-m target (Holtzman)"
    flag_sdss4_apogee2_onebit_gt_0_5 = "Selected in single (J-Ks)o > 0.5 color bin"
    flag_sdss4_apogee2_twobit_0_5_to_0_8 = "Selected in 'blue' 0.5 < (J-Ks)o < 0.8 color bin"
    flag_sdss4_apogee2_twobit_gt_0_8 = "Selected in 'red' (J-Ks)o > 0.8 color bin"
    flag_sdss4_apogee2_irac_dered = "Selected with RJCE-IRAC dereddening"
    flag_sdss4_apogee2_wise_dered = "Selected with RJCE-WISE dereddening"
    flag_sdss4_apogee2_sfd_dered = "Selected with SFD_EBV dereddening"
    flag_sdss4_apogee2_no_dered = "Selected with no dereddening"
    flag_sdss4_apogee2_wash_giant = "Selected as Wash+DDO51 photometric giant"
    flag_sdss4_apogee2_wash_dwarf = "Selected as Wash+DDO51 photometric dwarf"
    flag_sdss4_apogee2_sci_cluster = "Science cluster candidate member"
    flag_sdss4_apogee2_cluster_candidate = "Selected as globular cluster candidate"
    flag_sdss4_apogee2_short = "Selected as part of a short cohort"
    flag_sdss4_apogee2_medium = "Selected as part of a medium cohort"
    flag_sdss4_apogee2_long = "Selected as part of a long cohort"
    flag_sdss4_apogee2_normal_sample = "Selected as part of the random sample"
    flag_sdss4_apogee2_manga_led = "Star on a shared MaNGA-led design"
    flag_sdss4_apogee2_onebin_gt_0_3 = "Selected in single (J-Ks)o > 0.3 color bin"
    flag_sdss4_apogee2_wash_noclass = "Selected because it has no W+D classification"
    flag_sdss4_apogee2_stream_member = "Selected as confirmed halo tidal stream member"
    flag_sdss4_apogee2_stream_candidate = "Selected as potential halo tidal stream member (based on photometry)"
    flag_sdss4_apogee2_dsph_member = "Selected as confirmed dSph member (non Sgr)"
    flag_sdss4_apogee2_dsph_candidate = "Selected as potential dSph member (non Sgr) (based on photometry)"
    flag_sdss4_apogee2_magcloud_member = "Selected as confirmed Mag Cloud member"
    flag_sdss4_apogee2_magcloud_candidate = "Selected as potential Mag Cloud member (based on photometry)"
    flag_sdss4_apogee2_rrlyr = "Selected as an RR Lyrae star"
    flag_sdss4_apogee2_bulge_rc = "Selected as a bulge candidate RC star"
    flag_sdss4_apogee2_sgr_dsph = "Selected as confirmed Sgr core/stream member"
    flag_sdss4_apogee2_apokasc_giant = "Selected as part of APOKASC 'giant' sample"
    flag_sdss4_apogee2_apokasc_dwarf = "Selected as part of APOKASC 'dwarf' sample"
    flag_sdss4_apogee2_faint_extra = "'Faint' star (fainter than cohort limit; not required to reach survey S/N requirement)"
    flag_sdss4_apogee2_apokasc = "Selected as part of the APOKASC program (incl. seismic/gyro targets and others, both the Cygnus field and K2)"
    flag_sdss4_apogee2_k2_gap = "K2 Galactic Archeology Program Star"
    flag_sdss4_apogee2_ccloud_as4 = "California Cloud target"
    flag_sdss4_apogee2_standard_star = "Stellar parameters/abundance standard"
    flag_sdss4_apogee2_rv_standard = "Stellar RV standard"
    flag_sdss4_apogee2_sky = "Sky fiber"
    flag_sdss4_apogee2_external_calib = "External survey calibration target (generic flag; others below dedicated to specific surveys)"
    flag_sdss4_apogee2_internal_calib = "Internal survey calibration target (observed in at least 2 of: APOGEE-1, -2N, -2S)"
    flag_sdss4_apogee2_disk_substructure_member = "Bright time extension: outer disk substructure (Triand, GASS, and A13) members"
    flag_sdss4_apogee2_disk_substructure_candidate = "Bright time extension: outer disk substructure (Triand, GASS, and A13) candidates"
    flag_sdss4_apogee2_telluric = "Telluric standard"
    flag_sdss4_apogee2_calib_cluster = "Selected as calibration cluster member"
    flag_sdss4_apogee2_k2_planet_host = "Planet host in the K2 field"
    flag_sdss4_apogee2_tidal_binary = "Ancillary KOI Program (Simonian)"
    flag_sdss4_apogee2_literature_calib = "Overlap with high-resolution literature studies"
    flag_sdss4_apogee2_ges_overlap = "Overlap with Gaia-ESO"
    flag_sdss4_apogee2_argos_overlap = "Overlap with ARGOS"
    flag_sdss4_apogee2_gaia_overlap = "Overlap with Gaia"
    flag_sdss4_apogee2_galah_overlap = "Overlap with GALAH"
    flag_sdss4_apogee2_rave_overlap = "Overlap with RAVE"
    flag_sdss4_apogee2_commis_south_spec = "Commissioning special targets for APOGEE2S"
    flag_sdss4_apogee2_halo_member = "Halo Member"
    flag_sdss4_apogee2_halo_candidate = "Halo Candidate"
    flag_sdss4_apogee2_1m_target = "Selected as a 1-m target"
    flag_sdss4_apogee2_mod_bright_limit = "Selected in a cohort with H>10 rather than H>7"
    flag_sdss4_apogee2_cis = "Carnegie program target"
    flag_sdss4_apogee2_cntac = "Chilean community target"
    flag_sdss4_apogee2_external = "Proprietary external target"
    flag_sdss4_apogee2_cvz_as4_obaf = "OBAF stars selected for multi-epoc observations Andrew T."
    flag_sdss4_apogee2_cvz_as4_gi = "Submitted program to be on CVZ plate (Known Planets, ATL, Tayar-Subgiant, Canas-Cool-dwarf)"
    flag_sdss4_apogee2_cvz_as4_ctl = "Filler CTL star selected from the TESS Input Catalog"
    flag_sdss4_apogee2_cvz_as4_giant = "Filler Giant selected with RPMJ"
    flag_sdss4_apogee2_koi = "Selected as part of the long cadence KOI study"
    flag_sdss4_apogee2_eb = "Selected as part of the EB program"
    flag_sdss4_apogee2_koi_control = "Selected as part of the long cadence KOI 'control sample'"
    flag_sdss4_apogee2_mdwarf = "Selected as part of the M dwarf study"
    flag_sdss4_apogee2_substellar_companions = "Selected as part of the substellar companion search"
    flag_sdss4_apogee2_young_cluster = "Selected as part of the young cluster study (IN-SYNC)"
    flag_sdss4_apogee2_k2 = "Selected as part of the K2 program (BTX and Main Survey)"
    flag_sdss4_apogee2_object = "This object is an APOGEE-2 target"
    flag_sdss4_apogee2_ancillary = "Selected as an ancillary target"
    flag_sdss4_apogee2_massive_star = "Selected as part of the Massive Star program"
    flag_sdss4_apogee2_qso = "Ancillary QSO pilot program (Albareti)"
    flag_sdss4_apogee2_cepheid = "Ancillary Cepheid sparse targets (Beaton)"
    flag_sdss4_apogee2_low_av_windows = "Ancillary Deep Disk sample (Bovy)"
    flag_sdss4_apogee2_be_star = "Ancillary ASHELS sample (Chojnowski)"
    flag_sdss4_apogee2_young_moving_group = "Ancillary young moving group members (Downes)"
    flag_sdss4_apogee2_ngc6791 = "Ancillary NGC 6791 star (Geisler)"
    flag_sdss4_apogee2_label_star = "Ancillary Cannon calibrator Sample (Ness)"
    flag_sdss4_apogee2_faint_kepler_giants = "Ancillary APOKASC faint giants (Pinsonneault)"
    flag_sdss4_apogee2_w345 = "Ancillary W3/4/5 star forming complex (Roman-Lopes)"
    flag_sdss4_apogee2_massive_evolved = "Ancillary massive/evolved star targets (Stringfellow)"
    flag_sdss4_apogee2_extinction = "Ancillary extinction targets (Schlafly)"
    flag_sdss4_apogee2_kepler_mdwarf_koi = "Ancillary M dwarf targets (Smith)"
    flag_sdss4_apogee2_agb = "Ancillary AGB sample (Zamora)"
    flag_sdss4_apogee2_m33 = "Ancillary M33 Program (Anguiano)"
    flag_sdss4_apogee2_ultracool = "Ancillary Ultracool Dwarfs Program (Burgasser)"
    flag_sdss4_apogee2_distant_segue_giants = "Ancillary Distant SEGUE Giants program (Harding)"
    flag_sdss4_apogee2_cepheid_mapping = "Ancillary Cepheid Mapping Program (Inno)"
    flag_sdss4_apogee2_sa57 = "Ancillary SA57 Kapteyn Field Program (Majewski)"
    flag_sdss4_apogee2_k2_mdwarf = "Ancillary K2 M dwarf Program (Smith)"
    flag_sdss4_apogee2_rvvar = "Ancillary RV Variables Program (Troup)"
    flag_sdss4_apogee2_m31 = "Ancillary M31 Program (Zasowski)"
    flag_sdss4_apogee_not_main = "Not a main sample target"
    flag_sdss4_apogee_commissioning = "Commissioning observation"
    flag_sdss4_apogee_apo1m = "APO/NMSU 1M observation"
    flag_sdss4_apogee_duplicate = "Non-primary (not highest S/N) duplicate, excluding SDSS-5"
    flag_sdss4_apogee_member_m92 = "Likely member of M92"
    flag_sdss4_apogee_member_m15 = "Likely member of M15"
    flag_sdss4_apogee_member_m53 = "Likely member of M53"
    flag_sdss4_apogee_member_ngc_5466 = "Likely member of NGC 5466"
    flag_sdss4_apogee_member_ngc_4147 = "Likely member of NGC 4147"
    flag_sdss4_apogee_member_m2 = "Likely member of M2"
    flag_sdss4_apogee_member_m13 = "Likely member of M13"
    flag_sdss4_apogee_member_m3 = "Likely member of M3"
    flag_sdss4_apogee_member_m5 = "Likely member of M5"
    flag_sdss4_apogee_member_m12 = "Likely member of M12"
    flag_sdss4_apogee_member_m107 = "Likely member of M107"
    flag_sdss4_apogee_member_m71 = "Likely member of M71"
    flag_sdss4_apogee_member_ngc_2243 = "Likely member of NGC 2243"
    flag_sdss4_apogee_member_be29 = "Likely member of Be29"
    flag_sdss4_apogee_member_ngc_2158 = "Likely member of NGC 2158"
    flag_sdss4_apogee_member_m35 = "Likely member of M35"
    flag_sdss4_apogee_member_ngc_2420 = "Likely member of NGC 2420"
    flag_sdss4_apogee_member_ngc_188 = "Likely member of NGC 188"
    flag_sdss4_apogee_member_m67 = "Likely member of M67"
    flag_sdss4_apogee_member_ngc_7789 = "Likely member of NGC 7789"
    flag_sdss4_apogee_member_pleiades = "Likely member of Pleiades"
    flag_sdss4_apogee_member_ngc_6819 = "Likely member of NGC 6819"
    flag_sdss4_apogee_member_coma_berenices = "Likely member of Coma Berenices"
    flag_sdss4_apogee_member_ngc_6791 = "Likely member of NGC 6791"
    flag_sdss4_apogee_member_ngc_5053 = "Likely member of NGC 5053"
    flag_sdss4_apogee_member_m68 = "Likely member of M68"
    flag_sdss4_apogee_member_ngc_6397 = "Likely member of NGC 6397"
    flag_sdss4_apogee_member_m55 = "Likely member of M55"
    flag_sdss4_apogee_member_ngc_5634 = "Likely member of NGC 5634"
    flag_sdss4_apogee_member_m22 = "Likely member of M22"
    flag_sdss4_apogee_member_m79 = "Likely member of M79"
    flag_sdss4_apogee_member_ngc_3201 = "Likely member of NGC 3201"
    flag_sdss4_apogee_member_m10 = "Likely member of M10"
    flag_sdss4_apogee_member_ngc_6752 = "Likely member of NGC 6752"
    flag_sdss4_apogee_member_omega_centauri = "Likely member of Omega Centauri"
    flag_sdss4_apogee_member_m54 = "Likely member of M54"
    flag_sdss4_apogee_member_ngc_6229 = "Likely member of NGC 6229"
    flag_sdss4_apogee_member_pal5 = "Likely member of Pal5"
    flag_sdss4_apogee_member_ngc_6544 = "Likely member of NGC 6544"
    flag_sdss4_apogee_member_ngc_6522 = "Likely member of NGC 6522"
    flag_sdss4_apogee_member_ngc_288 = "Likely member of NGC 288"
    flag_sdss4_apogee_member_ngc_362 = "Likely member of NGC 362"
    flag_sdss4_apogee_member_ngc_1851 = "Likely member of NGC 1851"
    flag_sdss4_apogee_member_m4 = "Likely member of M4"
    flag_sdss4_apogee_member_ngc_2808 = "Likely member of NGC 2808"
    flag_sdss4_apogee_member_pal6 = "Likely member of Pal6"
    flag_sdss4_apogee_member_47tuc = "Likely member of 47 Tucane"
    flag_sdss4_apogee_member_pal1 = "Likely member of Pal1"
    flag_sdss4_apogee_member_ngc_6539 = "Likely member of NGC 6539"
    flag_sdss4_apogee_member_ngc_6388 = "Likely member of NGC 6388"
    flag_sdss4_apogee_member_ngc_6441 = "Likely member of NGC 6441"
    flag_sdss4_apogee_member_ngc_6316 = "Likely member of NGC 6316"
    flag_sdss4_apogee_member_ngc_6760 = "Likely member of NGC 6760"
    flag_sdss4_apogee_member_ngc_6553 = "Likely member of NGC 6553"
    flag_sdss4_apogee_member_ngc_6528 = "Likely member of NGC 6528"
    flag_sdss4_apogee_member_draco = "Likely member of Draco"
    flag_sdss4_apogee_member_urminor = "Likely member of Ursa Minor"
    flag_sdss4_apogee_member_bootes1 = "Likely member of Bootes 1"
    flag_sdss4_apogee_member_sexans = "Likely member of Sextans"
    flag_sdss4_apogee_member_fornax = "Likely member of Fornax"
    flag_sdss4_apogee_member_sculptor = "Likely member of Sculptor"
    flag_sdss4_apogee_member_carina = "Likely member of Carina"
    ra = "Right ascension [deg]"
    dec = "Declination [deg]"
    l = "Galactic longitude [deg]"
    b = "Galactic latitude [deg]"
    plx = "Parallax [mas]"
    e_plx = "Error on parallax [mas]"
    pmra = "Proper motion in RA [mas/yr]"
    e_pmra = "Error on proper motion in RA [mas/yr]"
    pmde = "Proper motion in DEC [mas/yr]"
    e_pmde = "Error on proper motion in DEC [mas/yr]"
    gaia_v_rad = "Gaia radial velocity [km/s]"
    gaia_e_v_rad = "Error on Gaia radial velocity [km/s]"
    g_mag = "Gaia DR3 mean G band magnitude [mag]"
    bp_mag = "Gaia DR3 mean BP band magnitude [mag]"
    rp_mag = "Gaia DR3 mean RP band magnitude [mag]"
    j_mag = "2MASS J band magnitude [mag]"
    e_j_mag = "Error on 2MASS J band magnitude [mag]"
    h_mag = "2MASS H band magnitude [mag]"
    e_h_mag = "Error on 2MASS H band magnitude [mag]"
    k_mag = "2MASS K band magnitude [mag]"
    e_k_mag = "Error on 2MASS K band magnitude [mag]"
    ph_qual = "2MASS photometric quality flag"
    bl_flg = "Number of components fit per band (JHK)"
    cc_flg = "Contamination and confusion flag"
    w1_mag = "W1 magnitude"
    e_w1_mag = "Error on W1 magnitude"
    w1_flux = "W1 flux [Vega nMgy]"
    w1_dflux = "Error on W1 flux [Vega nMgy]"
    w1_frac = "Fraction of W1 flux from this object"
    w2_mag = "W2 magnitude [Vega]"
    e_w2_mag = "Error on W2 magnitude"
    w2_flux = "W2 flux [Vega nMgy]"
    w2_dflux = "Error on W2 flux [Vega nMgy]"
    w2_frac = "Fraction of W2 flux from this object"
    w1uflags = "unWISE flags for W1"
    w2uflags = "unWISE flags for W2"
    w1aflags = "Additional flags for W1"
    w2aflags = "Additional flags for W2"
    flag_unwise_w1_in_core_or_wings = "In core or wings"
    flag_unwise_w1_in_diffraction_spike = "In diffraction spike"
    flag_unwise_w1_in_ghost = "In ghost"
    flag_unwise_w1_in_first_latent = "In first latent"
    flag_unwise_w1_in_second_latent = "In second latent"
    flag_unwise_w1_in_circular_halo = "In circular halo"
    flag_unwise_w1_saturated = "Saturated"
    flag_unwise_w1_in_geometric_diffraction_spike = "In geometric diffraction spike"
    flag_unwise_w2_in_core_or_wings = "In core or wings"
    flag_unwise_w2_in_diffraction_spike = "In diffraction spike"
    flag_unwise_w2_in_ghost = "In ghost"
    flag_unwise_w2_in_first_latent = "In first latent"
    flag_unwise_w2_in_second_latent = "In second latent"
    flag_unwise_w2_in_circular_halo = "In circular halo"
    flag_unwise_w2_saturated = "Saturated"
    flag_unwise_w2_in_geometric_diffraction_spike = "In geometric diffraction spike"
    flag_unwise_w1_in_bright_star_psf = "In PSF of bright star falling off coadd"
    flag_unwise_w1_in_hyperleda_galaxy = "In HyperLeda large galaxy"
    flag_unwise_w1_in_big_object = "In \"big object\" (e.g., a Magellanic cloud)"
    flag_unwise_w1_pixel_in_very_bright_star_centroid = "Pixel may contain the centroid of a very bright star"
    flag_unwise_w1_crowdsource_saturation = "crowdsource considers this pixel potentially affected by saturation"
    flag_unwise_w1_possible_nebulosity = "Pixel may contain nebulosity"
    flag_unwise_w1_no_aggressive_deblend = "Sources in this pixel will not be aggressively deblended"
    flag_unwise_w1_candidate_sources_must_be_sharp = "Candidate sources in this pixel must be \"sharp\" to be optimized"
    flag_unwise_w2_in_bright_star_psf = "In PSF of bright star falling off coadd"
    flag_unwise_w2_in_hyperleda_galaxy = "In HyperLeda large galaxy"
    flag_unwise_w2_in_big_object = "In \"big object\" (e.g., a Magellanic cloud)"
    flag_unwise_w2_pixel_in_very_bright_star_centroid = "Pixel may contain the centroid of a very bright star"
    flag_unwise_w2_crowdsource_saturation = "crowdsource considers this pixel potentially affected by saturation"
    flag_unwise_w2_possible_nebulosity = "Pixel may contain nebulosity"
    flag_unwise_w2_no_aggressive_deblend = "Sources in this pixel will not be aggressively deblended"
    flag_unwise_w2_candidate_sources_must_be_sharp = "Candidate sources in this pixel must be \"sharp\" to be optimized"
    mag4_5 = "IRAC band 4.5 micron magnitude [mag]"
    d4_5m = "Error on IRAC band 4.5 micron magnitude [mag]"
    rms_f4_5 = "RMS deviations from final flux [mJy]"
    sqf_4_5 = "Source quality flag for IRAC band 4.5 micron"
    mf4_5 = "Flux calculation method flag"
    csf = "Close source flag"
    flag_glimpse_poor_dark_pixel_current = "Poor pixels in dark current"
    flag_glimpse_flat_field_questionable = "Flat field applied using questionable value"
    flag_glimpse_latent_image = "Latent image"
    flag_glimpse_saturated_star_correction = "Sat star correction"
    flag_glimpse_muxbleed_correction_applied = "Muxbleed correction applied"
    flag_glimpse_hot_or_dead_pixels = "Hot, dead or otherwise unacceptable pixel"
    flag_glimpse_muxbleed_significant = "Muxbleed > 3-sigma above the background"
    flag_glimpse_allstar_tweak_positive = "Allstar tweak positive"
    flag_glimpse_allstar_tweak_negative = "Allstar tweak negative"
    flag_glimpse_confusion_in_band_merge = "Confusion in in-band merge"
    flag_glimpse_confusion_in_cross_band_merge = "Confusion in cross-band merge"
    flag_glimpse_column_pulldown_correction = "Column pulldown correction"
    flag_glimpse_banding_correction = "Banding correction"
    flag_glimpse_stray_light = "Stray light"
    flag_glimpse_no_nonlinear_correction = "Nonlinear correction not applied"
    flag_glimpse_saturated_star_wing_region = "Saturated star wing region"
    flag_glimpse_pre_lumping_in_band_merge = "Pre-lumping in in-band merge"
    flag_glimpse_post_lumping_in_cross_band_merge = "Post-lumping in cross-band merge"
    flag_glimpse_edge_of_frame = "Edge of frame (within 3 pixels of edge)"
    flag_glimpse_truth_list = "Truth list (for simulated data)"
    flag_glimpse_no_source_within_3_arcsecond = "No GLIMPSE sources within 3\" of this source"
    flag_glimpse_1_source_within_2p5_and_3_arcsecond = "1 source in GLIMPSE between 2.5\" and 3\" of this source"
    flag_glimpse_2_sources_within_2_and_2p5_arcsecond = "2 sources in GLIMPSE within 2\" and 2.5\" of this source"
    flag_glimpse_3_sources_within_1p5_and_2_arcsecond = "3 sources in GLIMPSE within 1.5\" and 2\" of this source"
    flag_glimpse_4_sources_within_1_and_1p5_arcsecond = "4 sources in GLIMPSE within 1\" and 1.5\" of this source"
    flag_glimpse_5_sources_within_0p5_and_1_arcsecond = "5 sources in GLIMPSE within 0.5\" and 1.0\" of this source"
    flag_glimpse_6_sources_within_0p5_arcsecond = "6 sources in GLIMPSE within 0.5\" of this source"
    zgr_teff = "Stellar effective temperature [K]"
    zgr_e_teff = "Error on stellar effective temperature [K]"
    zgr_logg = "Surface gravity [log10(cm/s^2)]"
    zgr_e_logg = "Error on surface gravity [log10(cm/s^2)]"
    zgr_fe_h = "[Fe/H] [dex]"
    zgr_e_fe_h = "Error on [Fe/H] [dex]"
    zgr_e = "Extinction [mag]"
    zgr_e_e = "Error on extinction [mag]"
    zgr_plx = "Parallax [mas] (Gaia DR3)"
    zgr_e_plx = "Error on parallax [mas] (Gaia DR3)"
    zgr_teff_confidence = "Confidence estimate in TEFF"
    zgr_logg_confidence = "Confidence estimate in LOGG"
    zgr_fe_h_confidence = "Confidence estimate in FE_H"
    zgr_ln_prior = "Log prior probability"
    zgr_chi2 = "Chi-square value"
    zgr_quality_flags = "Quality flags"
    r_med_geo = "Median geometric distance [pc]"
    r_lo_geo = "16th percentile of geometric distance [pc]"
    r_hi_geo = "84th percentile of geometric distance [pc]"
    r_med_photogeo = "50th percentile of photogeometric distance [pc]"
    r_lo_photogeo = "16th percentile of photogeometric distance [pc]"
    r_hi_photogeo = "84th percentile of photogeometric distance [pc]"
    bailer_jones_flags = "Bailer-Jones quality flags"
    ebv = "E(B-V) [mag]"
    e_ebv = "Error on E(B-V) [mag]"
    ebv_flags = "Flags indicating the source of E(B-V)"
    flag_ebv_upper_limit = "E(B-V) is an upper limit"
    flag_ebv_from_zhang_2023 = "E(B-V) from Zhang et al. (2023)"
    flag_ebv_from_edenhofer_2023 = "E(B-V) from Edenhofer et al. (2023)"
    flag_ebv_from_sfd = "E(B-V) from SFD"
    flag_ebv_from_rjce_glimpse = "E(B-V) from RJCE GLIMPSE"
    flag_ebv_from_rjce_allwise = "E(B-V) from RJCE AllWISE"
    flag_ebv_from_bayestar_2019 = "E(B-V) from Bayestar 2019"
    ebv_zhang_2023 = "E(B-V) from Zhang et al. (2023) [mag]"
    e_ebv_zhang_2023 = "Error on E(B-V) from Zhang et al. (2023) [mag]"
    ebv_sfd = "E(B-V) from SFD [mag]"
    e_ebv_sfd = "Error on E(B-V) from SFD [mag]"
    ebv_rjce_glimpse = "E(B-V) from RJCE GLIMPSE [mag]"
    e_ebv_rjce_glimpse = "Error on RJCE GLIMPSE E(B-V) [mag]"
    ebv_rjce_allwise = "E(B-V) from RJCE AllWISE [mag]"
    e_ebv_rjce_allwise = "Error on RJCE AllWISE E(B-V)[mag]"
    ebv_bayestar_2019 = "E(B-V) from Bayestar 2019 [mag]"
    e_ebv_bayestar_2019 = "Error on Bayestar 2019 E(B-V) [mag]"
    ebv_edenhofer_2023 = "E(B-V) from Edenhofer et al. (2023) [mag]"
    e_ebv_edenhofer_2023 = "Error on Edenhofer et al. (2023) E(B-V) [mag]"
    c_star = "Quality parameter (see Riello et al. 2021)"
    u_jkc_mag = "Gaia XP synthetic U-band (JKC) [mag]"
    u_jkc_mag_flag = "U-band (JKC) is within valid range"
    b_jkc_mag = "Gaia XP synthetic B-band (JKC) [mag]"
    b_jkc_mag_flag = "B-band (JKC) is within valid range"
    v_jkc_mag = "Gaia XP synthetic V-band (JKC) [mag]"
    v_jkc_mag_flag = "V-band (JKC) is within valid range"
    r_jkc_mag = "Gaia XP synthetic R-band (JKC) [mag]"
    r_jkc_mag_flag = "R-band (JKC) is within valid range"
    i_jkc_mag = "Gaia XP synthetic I-band (JKC) [mag]"
    i_jkc_mag_flag = "I-band (JKC) is within valid range"
    u_sdss_mag = "Gaia XP synthetic u-band (SDSS) [mag]"
    u_sdss_mag_flag = "u-band (SDSS) is within valid range"
    g_sdss_mag = "Gaia XP synthetic g-band (SDSS) [mag]"
    g_sdss_mag_flag = "g-band (SDSS) is within valid range"
    r_sdss_mag = "Gaia XP synthetic r-band (SDSS) [mag]"
    r_sdss_mag_flag = "r-band (SDSS) is within valid range"
    i_sdss_mag = "Gaia XP synthetic i-band (SDSS) [mag]"
    i_sdss_mag_flag = "i-band (SDSS) is within valid range"
    z_sdss_mag = "Gaia XP synthetic z-band (SDSS) [mag]"
    z_sdss_mag_flag = "z-band (SDSS) is within valid range"
    y_ps1_mag = "Gaia XP synthetic Y-band (PS1) [mag]"
    y_ps1_mag_flag = "Y-band (PS1) is within valid range"
    n_boss_visits = "Number of BOSS visits"
    boss_min_mjd = "Minimum MJD of BOSS visits"
    boss_max_mjd = "Maximum MJD of BOSS visits"
    n_apogee_visits = "Number of APOGEE visits"
    apogee_min_mjd = "Minimum MJD of APOGEE visits"
    apogee_max_mjd = "Maximum MJD of APOGEE visits"

    # APOGEE:
    pk = "Database primary key"
    #source = "Unique source primary key"
    spectrum_pk = "Unique spectrum primary key"

    wavelength = "Wavelength (vacuum) [Angstrom]"
    flux = "Flux [10^-17 ergs/s/cm^2/A]"
    ivar = "Inverse variance on flux [10^34 ergs^-2/s^-2/cm^-4/A^-2]"
    pixel_flags = "Pixel-level quality flags (see documentation)"

    catalogid = "SDSS input catalog identifier"
    star_pk = "APOGEE DRP `star` primary key"
    visit_pk = "APOGEE DRP `visit` primary key"
    rv_visit_pk = "APOGEE DRP `rv_visit` primary key"

    release = "SDSS release"
    filetype = "SDSS file type that stores this spectrum"
    apstar = "Unused DR17 apStar keyword (default: stars)"
    apred = "APOGEE reduction pipeline"
    plate = "Plate identifier"
    telescope = "Short telescope name"
    fiber = "Fiber number"
    mjd = "Modified Julian date of observation"
    field = "Field identifier"
    prefix = "Prefix used to separate SDSS 4 north/south"

    reduction = "An `obj`-like keyword used for apo1m spectra"

    obj = "Object name"

    date_obs = "Observation date (UTC)"
    jd = "Julian date at mid-point of visit"
    exptime = "Exposure time [s]"
    dithered = "Fraction of visits that were dithered"
    f_night_time = "Mid obs time as fraction from sunset to sunrise"
    input_ra = "Input right ascension [deg]"
    input_dec = "Input declination [deg]"
    n_frames = "Number of frames combined"
    assigned = "FPS target assigned"
    on_target = "FPS fiber on target"
    valid = "Valid FPS target"
    fps = "Fibre positioner used to acquire this data?"
    snr = "Signal-to-noise ratio"

    spectrum_flags = "Data reduction pipeline flags for this spectrum"
    flag_bad_pixels = "Spectrum has many bad pixels (>20%)."
    flag_commissioning = "Commissioning data (MJD <55761); non-standard configuration; poor LSF."
    flag_bright_neighbor = "Star has neighbor more than 10 times brighter."
    flag_very_bright_neighbor = "Star has neighbor more than 100 times brighter."
    flag_low_snr = "Spectrum has low S/N (<5)."
    flag_persist_high = "Spectrum has at least 20% of pixels in high persistence region."
    flag_persist_med = "Spectrum has at least 20% of pixels in medium persistence region."
    flag_persist_low = "Spectrum has at least 20% of pixels in low persistence region."
    flag_persist_jump_pos = "Spectrum has obvious positive jump in blue chip."
    flag_persist_jump_neg = "Spectrum has obvious negative jump in blue chip."
    flag_suspect_rv_combination = "RVs from synthetic template differ significantly (~2 km/s) from those from combined template."
    flag_suspect_broad_lines = "Cross-correlation peak with template significantly broader than autocorrelation of template."
    flag_bad_rv_combination = "RVs from synthetic template differ very significantly (~10 km/s) from those from combined template."
    flag_rv_reject = "Rejected visit because cross-correlation RV differs significantly from least squares RV."
    flag_rv_suspect = "Suspect visit (but used!) because cross-correlation RV differs slightly from least squares RV."
    flag_multiple_suspect = "Suspect multiple components from Gaussian decomposition of cross-correlation."
    flag_rv_failure = "RV failure."
    flag_suspect_rotation = "Suspect rotation: cross-correlation peak with template significantly broader than autocorretion of template"
    flag_mtpflux_lt_75 = "Spectrum falls on fiber in MTP block with relative flux < 0.75"
    flag_mtpflux_lt_50 = "Spectrum falls on fiber in MTP block with relative flux < 0.5"

    v_rad = "Barycentric rest frame radial velocity [km/s]"
    v_rel = "Relative velocity [km/s]"
    e_v_rel = "Error on relative velocity [km/s]"
    bc = "Barycentric velocity correction applied [km/s]"
    doppler_teff = "Stellar effective temperature [K]"
    doppler_e_teff = "Error on stellar effective temperature [K]"
    doppler_logg = "Surface gravity [log10(cm/s^2)]"
    doppler_e_logg = "Error on surface gravity [log10(cm/s^2)]"
    doppler_fe_h = "[Fe/H] [dex]"
    doppler_e_fe_h = "Error on [Fe/H] [dex]"
    doppler_rchi2 = "Reduced chi-square value of DOPPLER fit"
    doppler_flags = "DOPPLER flags"
    xcorr_v_rad = "Barycentric rest frame radial velocity [km/s]"
    xcorr_v_rel = "Relative velocity [km/s]"
    xcorr_e_v_rel = "Error on relative velocity [km/s]"
    ccfwhm = "Cross-correlation function FWHM"
    autofwhm = "Auto-correlation function FWHM"
    n_components = "Number of components in CCF"

    field = "Field identifier"
    prefix = "Prefix used to separate SDSS 4 north/south"
    min_mjd = "Minimum MJD of visits"
    max_mjd = "Maximum MJD of visits"

    n_entries = "apStar entries for this SDSS4_APOGEE_ID"
    n_visits = "Number of APOGEE visits"
    n_good_visits = "Number of 'good' APOGEE visits"
    n_good_rvs = "Number of 'good' APOGEE radial velocities"
    snr = "Signal-to-noise ratio"
    mean_fiber = "S/N-weighted mean visit fiber number"
    std_fiber = "Standard deviation of visit fiber numbers"
    spectrum_flags = "Data reduction pipeline flags for this spectrum"
    v_rad = "Barycentric rest frame radial velocity [km/s]"
    e_v_rad = "Error on radial velocity [km/s]"
    std_v_rad = "Standard deviation of visit V_RAD [km/s]"
    median_e_v_rad = "Median error in radial velocity [km/s]"

    tag = "Experiment tag for this result"
    t_elapsed = "Core-time elapsed on this analysis [s]"
    t_overhead = "Estimated core-time spent in overhads [s]"

    # ApogeeNet
    created = "Datetime when task record was created"
    teff = "Stellar effective temperature [K]"
    e_teff = "Error on stellar effective temperature [K]"
    logg = "Surface gravity [log10(cm/s^2)]"
    e_logg = "Error on surface gravity [log10(cm/s^2)]"
    fe_h = "[Fe/H] [dex]"
    e_fe_h = "Error on [Fe/H] [dex]"
    result_flags = "Flags describing the results"
    raw_e_teff = "Raw error on stellar effective temperature [K]"
    raw_e_logg = "Raw error on surface gravity [log10(cm/s^2)]"
    raw_e_fe_h = "Raw error on [Fe/H] [dex]"


    # AstroNNdist:
    A_k_mag = "Ks-band extinction"
    L_fakemag = "Predicted (fake) Ks-band absolute luminosity, L_fakemag = 10^(1/5*M_Ks+2)"    
    e_L_fakemag = "Prediected (fake) Ks-band absolute luminosity error"
    dist = "Heliocentric distance [pc]"
    e_dist = "Heliocentric distance error [pc]"

def lower_first_letter(s):
    return f"{s[0].lower()}{s[1:]}"

MISSING_GLOSSARY_TERMS = set()

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
                    
    warnings.warn(f"There are some missing glossary definitions. See `astra.glossary.MISSING_GLOSSARY_TERMS`.")
    MISSING_GLOSSARY_TERMS.add(name)
    return None


def warn_on_long_description(text, max_length=80):
    return text
