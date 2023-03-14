import numpy as np
from astropy.io import fits
from astra.utils import list_to_dict
from astra.sdss.meta import StarMeta, VisitMeta
from astropy.table import Table
from astra.sdss.datamodels.base import add_check_sums, fits_column_kwargs, add_glossary_comments, add_table_category_headers

from peewee import fn, Case, JOIN, Alias, ForeignKeyField, DateTimeField, BigIntegerField, FloatField, IntegerField, TextField, BooleanField

COMMON_CATEGORY_HEADERS = [
    ("cat_id", "Identifiers"),
    ("ra", "Astrometry"),
    ("g_mag", "Photometry"),
    ("carton_0", "Targeting"),
    ("doppler_teff", "Doppler"),
    ("xcsao_teff", "XCSAO"),
    ("astra_version_major", "Astra"),
]

COMMON_STAR_FIELDS = [
    StarMeta.cat_id,
    StarMeta.cat_id05,
    StarMeta.cat_id10,
    StarMeta.tic_id,
    StarMeta.gaia_source_id,
    StarMeta.gaia_data_release,
    StarMeta.healpix,

    StarMeta.ra,
    StarMeta.dec,
    StarMeta.gaia_ra,
    StarMeta.gaia_dec,
    StarMeta.plx,
    StarMeta.pmra,
    StarMeta.pmde,
    StarMeta.e_pmra,
    StarMeta.e_pmde,
    StarMeta.gaia_v_rad,
    StarMeta.gaia_e_v_rad,

    StarMeta.g_mag,
    StarMeta.bp_mag,
    StarMeta.rp_mag,
    StarMeta.j_mag,
    StarMeta.h_mag,
    StarMeta.k_mag,
    StarMeta.e_j_mag,
    StarMeta.e_h_mag,
    StarMeta.e_k_mag,

    StarMeta.carton_0,
    StarMeta.v_xmatch,
    
    StarMeta.doppler_teff,
    StarMeta.doppler_e_teff,
    StarMeta.doppler_logg,
    StarMeta.doppler_e_logg,
    StarMeta.doppler_fe_h,
    StarMeta.doppler_e_fe_h,
    StarMeta.doppler_starflag.alias("doppler_flag"),
    StarMeta.doppler_version,
    StarMeta.doppler_v_rad,

    StarMeta.xcsao_teff,
    StarMeta.xcsao_e_teff,
    StarMeta.xcsao_logg,
    StarMeta.xcsao_e_logg,
    StarMeta.xcsao_fe_h,
    StarMeta.xcsao_e_fe_h,
    StarMeta.xcsao_rxc,
    StarMeta.xcsao_v_rad,
    StarMeta.xcsao_e_v_rad,

    StarMeta.astra_version_major,
    StarMeta.astra_version_minor,
    StarMeta.astra_version_patch,
    StarMeta.created,
]

COMMON_VISIT_FIELDS = [
    VisitMeta.cat_id,
    VisitMeta.cat_id05,
    VisitMeta.cat_id10,
    VisitMeta.tic_id,
    VisitMeta.gaia_source_id,
    VisitMeta.gaia_data_release,
    VisitMeta.healpix,

    VisitMeta.ra,
    VisitMeta.dec,
    VisitMeta.gaia_ra,
    VisitMeta.gaia_dec,
    VisitMeta.plx,
    VisitMeta.pmra,
    VisitMeta.pmde,
    VisitMeta.e_pmra,
    VisitMeta.e_pmde,
    VisitMeta.gaia_v_rad,
    VisitMeta.gaia_e_v_rad,

    VisitMeta.g_mag,
    VisitMeta.bp_mag,
    VisitMeta.rp_mag,
    VisitMeta.j_mag,
    VisitMeta.h_mag,
    VisitMeta.k_mag,
    VisitMeta.e_j_mag,
    VisitMeta.e_h_mag,
    VisitMeta.e_k_mag,

    VisitMeta.carton_0,
    VisitMeta.v_xmatch,
    
    VisitMeta.doppler_teff,
    VisitMeta.doppler_e_teff,
    VisitMeta.doppler_logg,
    VisitMeta.doppler_e_logg,
    VisitMeta.doppler_fe_h,
    VisitMeta.doppler_e_fe_h,
    VisitMeta.doppler_starflag.alias("doppler_flag"),
    VisitMeta.doppler_version,

    VisitMeta.xcsao_teff,
    VisitMeta.xcsao_e_teff,
    VisitMeta.xcsao_logg,
    VisitMeta.xcsao_e_logg,
    VisitMeta.xcsao_fe_h,
    VisitMeta.xcsao_e_fe_h,
    VisitMeta.xcsao_rxc,

    VisitMeta.release,
    VisitMeta.filetype,
    VisitMeta.plate,
    VisitMeta.fiber,
    VisitMeta.field,
    VisitMeta.apred,
    VisitMeta.prefix,
    VisitMeta.mjd,
    VisitMeta.run2d,
    VisitMeta.fieldid,
    VisitMeta.isplate,
    VisitMeta.catalogid,
    
    # Common stuff.
    VisitMeta.observatory,
    VisitMeta.instrument,
    VisitMeta.hdu_data_index,
    VisitMeta.snr,
    VisitMeta.fps,
    VisitMeta.in_stack,
    VisitMeta.v_shift,
    
    VisitMeta.continuum_theta,
    
    # APOGEE-level stuff.
    VisitMeta.v_apred,
    VisitMeta.nres,
    VisitMeta.filtsize,
    VisitMeta.normsize,
    VisitMeta.conscale,
    
    # Doppler results.
    VisitMeta.doppler_teff,
    VisitMeta.doppler_e_teff,
    VisitMeta.doppler_logg,
    VisitMeta.doppler_e_logg,
    VisitMeta.doppler_fe_h,
    VisitMeta.doppler_e_fe_h,
    VisitMeta.doppler_starflag,
    VisitMeta.doppler_version,
    
    VisitMeta.date_obs,
    VisitMeta.exptime,
    VisitMeta.fluxflam,
    VisitMeta.npairs,
    VisitMeta.dithered,
    
    VisitMeta.jd,
    VisitMeta.v_rad,
    VisitMeta.e_v_rad,
    VisitMeta.v_rel,
    VisitMeta.v_bc,
    VisitMeta.rchisq,
    VisitMeta.n_rv_components,
    
    VisitMeta.visit_pk,
    VisitMeta.rv_visit_pk,
    
    VisitMeta.v_boss,
    VisitMeta.vjaeger,
    VisitMeta.vkaiju,
    VisitMeta.vcoordio,
    VisitMeta.vcalibs,
    VisitMeta.versidl,
    VisitMeta.versutil,
    VisitMeta.versread,
    VisitMeta.vers2d,
    VisitMeta.verscomb,
    VisitMeta.verslog,
    VisitMeta.versflat,
    VisitMeta.didflush,
    VisitMeta.cartid,
    VisitMeta.psfsky,
    VisitMeta.preject,
    VisitMeta.lowrej,
    VisitMeta.highrej,
    VisitMeta.scatpoly,
    VisitMeta.proftype,
    VisitMeta.nfitpoly,
    VisitMeta.skychi2,
    VisitMeta.schi2min,
    VisitMeta.schi2max,
    VisitMeta.rdnoise0,
    
    VisitMeta.alt,
    VisitMeta.az,
    VisitMeta.seeing,
    VisitMeta.airmass,
    VisitMeta.airtemp,
    VisitMeta.dewpoint,
    VisitMeta.humidity,
    VisitMeta.pressure,
    VisitMeta.gustd,
    VisitMeta.gusts,
    VisitMeta.windd,
    VisitMeta.winds,
    VisitMeta.moon_dist_mean,
    VisitMeta.moon_phase_mean,
    VisitMeta.nexp,
    VisitMeta.nguide,
    VisitMeta.tai_beg,
    VisitMeta.tai_end,
    VisitMeta.fiber_offset,
    VisitMeta.delta_ra,
    VisitMeta.delta_dec,
    VisitMeta.zwarning,
    
    VisitMeta.astra_version_major,
    VisitMeta.astra_version_minor,
    VisitMeta.astra_version_patch,
    VisitMeta.created,    

]

def fits_column_kwargs(field, values):

    field_type = type(field)
    if field_type == Alias:
        field_type = type(field.node)

    if field_type == TextField:
        max_len = max(map(len, map(str, values)))
        format = f"{max_len}A"
        fill_value = ""
    elif field_type in (BigIntegerField, ForeignKeyField):
        format = "K"
        fill_value = -1
    elif field_type == BooleanField:
        format = "L"
        fill_value = None
    elif field_type == FloatField:
        format = "E"
        fill_value = np.nan
    elif field_type == IntegerField:
        non_none_values = [v for v in values if v is not None]
        if non_none_values:
            max_v = max(non_none_values)
        else:
            max_v = 0
        is_double_precision = int(max_v >> 32) > 0
        if is_double_precision:
            format = "K"
            fill_value = -1
        else:
            format = "J"
            fill_value = -1
    else:
        # work it out from first value
        #raise ValueError(f"Unknown field type: {field_type}")
        print(f"hack for {field_type}")
        format = "E"
        fill_value = np.nan
    
    kwds = {}
    if isinstance(values[0], (np.ndarray, list)):
        # S = values.size
        
        V, P = np.atleast_2d(values).shape
        if np.array(values).ndim == 2:
            kwds["format"] = f"{P:.0f}{format}"
            kwds["dim"] = f"({P})"
        else:
            kwds["format"] = f"{format}"
    else:
        kwds["format"] = format        
    
    return (kwds, fill_value)





def create_summary_table(
    data,
    fields, 
    category_headers,
    upper=True
):

    # TODO: When the same column name is given twice, use a temporary name, warn, and keep going (eg SNR_1, SNR_2)

    columns = []
    for field in fields:
        name = field.name
        values = data[name]

        if type(field) == DateTimeField:
            array = [(v.isoformat() if v is not None else "") for v in values]
            kwds = dict(format="19A")
        else:
            kwds, fill_value = fits_column_kwargs(field, values)
            array = [(v if v is not None else fill_value) for v in values]

        column = fits.Column(
            name=name.upper() if upper else name,
            array=array,
            unit=None,
            **kwds
        )
        columns.append(column)

    hdu = fits.BinTableHDU.from_columns(columns)

    add_table_category_headers(hdu, category_headers, upper=upper)
    add_glossary_comments(hdu)

    try:
        index = 1 + hdu.header.dtype.names.index("CREATED")
        hdu.header.comments[f"TTYPE{index}"] = "Task created"
    except:
        None

    primary = fits.PrimaryHDU()
    hdu_list = fits.HDUList([primary, hdu])
    add_check_sums(hdu_list)

    return hdu_list



def create_allStar_apogeenet_table():
    
    from astra.contrib.apogeenet.base import ApogeeNetOutput

    fields = COMMON_STAR_FIELDS + [
        ApogeeNetOutput.completed,
        ApogeeNetOutput.time_elapsed,

        ApogeeNetOutput.task,
        ApogeeNetOutput.data_product,
        ApogeeNetOutput.source,

        ApogeeNetOutput.model_path,
        ApogeeNetOutput.large_error,
        ApogeeNetOutput.num_uncertainty_draws,

        ApogeeNetOutput.telescope,
        ApogeeNetOutput.instrument,
        ApogeeNetOutput.snr,

        ApogeeNetOutput.teff,
        ApogeeNetOutput.e_teff,
        ApogeeNetOutput.logg,
        ApogeeNetOutput.e_logg,
        ApogeeNetOutput.fe_h,
        ApogeeNetOutput.e_fe_h,
        ApogeeNetOutput.teff_sample_median,
        ApogeeNetOutput.logg_sample_median,
        ApogeeNetOutput.fe_h_sample_median,
        ApogeeNetOutput.bitmask_flag.alias("flag"),
    ]
    category_headers = COMMON_CATEGORY_HEADERS + [
        ("task", "Metadata"),
        ("model_path", "Task Parameters"),
        ("telescope", "Observations"),
        ("teff", "Stellar Parameters"),
    ]

    q = (
        StarMeta
        .select(*fields)
        .join(
            ApogeeNetOutput,
            on=(
                (ApogeeNetOutput.source_id == StarMeta.cat_id)
            &   (ApogeeNetOutput.fiber.is_null())
            )
        )
        .dicts()
    )

    data = list_to_dict(q)
    return create_summary_table(data, fields, category_headers)


def create_allVisit_apogeenet_table():
    
    from astra.contrib.apogeenet.base import ApogeeNetOutput

    fields = [
        VisitMeta.cat_id,
        VisitMeta.cat_id05,
        VisitMeta.cat_id10,
        VisitMeta.tic_id,
        VisitMeta.gaia_source_id,
        VisitMeta.gaia_data_release,
        VisitMeta.healpix,

        VisitMeta.ra,
        VisitMeta.dec,
        VisitMeta.gaia_ra,
        VisitMeta.gaia_dec,
        VisitMeta.plx,
        VisitMeta.pmra,
        VisitMeta.pmde,
        VisitMeta.e_pmra,
        VisitMeta.e_pmde,
        VisitMeta.gaia_v_rad,
        VisitMeta.gaia_e_v_rad,

        VisitMeta.g_mag,
        VisitMeta.bp_mag,
        VisitMeta.rp_mag,
        VisitMeta.j_mag,
        VisitMeta.h_mag,
        VisitMeta.k_mag,
        VisitMeta.e_j_mag,
        VisitMeta.e_h_mag,
        VisitMeta.e_k_mag,

        VisitMeta.carton_0,
        VisitMeta.v_xmatch,
        
        VisitMeta.doppler_teff,
        VisitMeta.doppler_e_teff,
        VisitMeta.doppler_logg,
        VisitMeta.doppler_e_logg,
        VisitMeta.doppler_fe_h,
        VisitMeta.doppler_e_fe_h,
        VisitMeta.doppler_starflag.alias("doppler_flag"),
        VisitMeta.doppler_version,

        VisitMeta.release,
        VisitMeta.filetype,
        VisitMeta.plate,
        VisitMeta.fiber,
        VisitMeta.field,
        VisitMeta.apred,
        VisitMeta.prefix,
        VisitMeta.mjd,
        
        VisitMeta.observatory,
        ApogeeNetOutput.telescope,
        ApogeeNetOutput.instrument,
        VisitMeta.hdu_data_index,
        VisitMeta.fps,
        VisitMeta.in_stack,
        VisitMeta.v_shift,
        
        # APOGEE-level stuff.
        VisitMeta.v_apred,
        VisitMeta.nres,
        VisitMeta.filtsize,
        VisitMeta.normsize,
        VisitMeta.conscale,
        
        VisitMeta.continuum_theta,

        VisitMeta.date_obs,
        VisitMeta.exptime,
        VisitMeta.fluxflam,
        VisitMeta.npairs,
        VisitMeta.dithered,
        
        VisitMeta.jd,
        VisitMeta.v_rad,
        VisitMeta.e_v_rad,
        VisitMeta.v_rel,
        VisitMeta.v_bc,
        VisitMeta.rchisq,
        VisitMeta.n_rv_components,
        
        VisitMeta.visit_pk,
        VisitMeta.rv_visit_pk,
                    
        ApogeeNetOutput.completed,
        ApogeeNetOutput.time_elapsed,

        ApogeeNetOutput.task,
        ApogeeNetOutput.data_product,
        ApogeeNetOutput.source,

        ApogeeNetOutput.model_path,
        ApogeeNetOutput.large_error,
        ApogeeNetOutput.num_uncertainty_draws,

        ApogeeNetOutput.snr,

        ApogeeNetOutput.teff,
        ApogeeNetOutput.e_teff,
        ApogeeNetOutput.logg,
        ApogeeNetOutput.e_logg,
        ApogeeNetOutput.fe_h,
        ApogeeNetOutput.e_fe_h,
        ApogeeNetOutput.teff_sample_median,
        ApogeeNetOutput.logg_sample_median,
        ApogeeNetOutput.fe_h_sample_median,
        ApogeeNetOutput.bitmask_flag.alias("flag"),
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("doppler_teff", "Doppler"),
        ("release", "Data Product Keywords"),
        ("observatory", "Observations"),

        ("v_apred", "APOGEE Data Reduction Pipeline"),     
        ("continuum_theta", "Continuum"),
        ("date_obs", "Metadata"),
        ("model_path", "Task Parameters"),
        ("teff", "Stellar Parameters")
    ]

    q = (
        VisitMeta
        .select(*fields)
        .join(
            ApogeeNetOutput,
            on=(
                (ApogeeNetOutput.source_id == VisitMeta.cat_id)
            &   (ApogeeNetOutput.fiber == VisitMeta.fiber)
            &   (ApogeeNetOutput.field == VisitMeta.field)
            &   (ApogeeNetOutput.mjd == VisitMeta.mjd)
            &   (ApogeeNetOutput.plate.cast("TEXT") == VisitMeta.plate.cast("TEXT"))
            &   (ApogeeNetOutput.fiber.is_null(False))
            )
        )
        .dicts()
    )

    data = list_to_dict(q)
    return create_summary_table(data, fields, category_headers)


def create_allStar_thepayne_table():
    
    from astra.contrib.thepayne.base import ThePayne

    fields = COMMON_STAR_FIELDS + [
        ThePayne.completed,
        ThePayne.time_elapsed,

        ThePayne.task,
        ThePayne.data_product,
        ThePayne.source,
        ThePayne.output_data_product,

        #ThePayne.model_path,
        #ThePayne.mask_path,
        #ThePayne.opt_tolerance,
        #ThePayne.v_rad_tolerance,

        ThePayne.telescope,
        ThePayne.instrument,
        ThePayne.snr,

        ThePayne.teff,
        ThePayne.e_teff,
        ThePayne.logg,
        ThePayne.e_logg,
        ThePayne.v_turb,
        ThePayne.e_v_turb,
        ThePayne.c_h,
        ThePayne.e_c_h,
        ThePayne.n_h,
        ThePayne.e_n_h,
        ThePayne.o_h,
        ThePayne.e_o_h,
        ThePayne.na_h,
        ThePayne.e_na_h,
        ThePayne.mg_h,
        ThePayne.e_mg_h,
        ThePayne.al_h,
        ThePayne.e_al_h,
        ThePayne.si_h,
        ThePayne.e_si_h,
        ThePayne.p_h,
        ThePayne.e_p_h,
        ThePayne.s_h,
        ThePayne.e_s_h,
        ThePayne.k_h,
        ThePayne.e_k_h,
        ThePayne.ca_h,
        ThePayne.e_ca_h,
        ThePayne.ti_h,
        ThePayne.e_ti_h,
        ThePayne.v_h,
        ThePayne.e_v_h,
        ThePayne.cr_h,
        ThePayne.e_cr_h,
        ThePayne.mn_h,
        ThePayne.e_mn_h,
        ThePayne.fe_h,
        ThePayne.e_fe_h,
        ThePayne.co_h,
        ThePayne.e_co_h,
        ThePayne.ni_h,
        ThePayne.e_ni_h,
        ThePayne.cu_h,
        ThePayne.e_cu_h,
        ThePayne.ge_h,
        ThePayne.e_ge_h,
        ThePayne.c12_c13,
        ThePayne.v_macro,

        ThePayne.bitmask_flag.alias("flag"),
        ThePayne.bitmask_teff.alias("flag_teff"),
        ThePayne.bitmask_logg.alias("flag_logg"),
        ThePayne.bitmask_v_turb.alias("flag_v_turb"),
        ThePayne.bitmask_c_h.alias("flag_c_h"),
        ThePayne.bitmask_n_h.alias("flag_n_h"),
        ThePayne.bitmask_o_h.alias("flag_o_h"),
        ThePayne.bitmask_na_h.alias("flag_na_h"),
        ThePayne.bitmask_mg_h.alias("flag_mg_h"),
        ThePayne.bitmask_al_h.alias("flag_al_h"),
        ThePayne.bitmask_si_h.alias("flag_si_h"),
        ThePayne.bitmask_p_h.alias("flag_p_h"),
        ThePayne.bitmask_s_h.alias("flag_s_h"),
        ThePayne.bitmask_k_h.alias("flag_k_h"),
        ThePayne.bitmask_ca_h.alias("flag_ca_h"),
        ThePayne.bitmask_ti_h.alias("flag_ti_h"),
        ThePayne.bitmask_v_h.alias("flag_v_h"),
        ThePayne.bitmask_cr_h.alias("flag_cr_h"),
        ThePayne.bitmask_mn_h.alias("flag_mn_h"),
        ThePayne.bitmask_fe_h.alias("flag_fe_h"),
        ThePayne.bitmask_co_h.alias("flag_co_h"),
        ThePayne.bitmask_ni_h.alias("flag_ni_h"),
        ThePayne.bitmask_cu_h.alias("flag_cu_h"),
        ThePayne.bitmask_ge_h.alias("flag_ge_h"),
        ThePayne.bitmask_c12_c13.alias("flag_c12_c13"),
        ThePayne.bitmask_v_macro.alias("flag_v_macro"),
        
        ThePayne.reduced_chi_sq,
        ThePayne.chi_sq,
    ]

    category_headers = COMMON_CATEGORY_HEADERS + [
        ("task", "Metadata"),
        #("model_path", "Task Parameters"),
        ("telescope", "Observations"),
        ("teff", "Stellar Labels"),
        ("flag", "Flags"),
        ("reduced_chi_sq", "Summary Statistics"),
    ]   

    q = (
        StarMeta
        .select(*fields)
        .join(
            ThePayne,
            on=(
                (ThePayne.source_id == StarMeta.cat_id)
            &   (ThePayne.fiber.is_null())
            )
        )
        .dicts()
    )
    data = list_to_dict(q)
    return create_summary_table(data, fields, category_headers)






def create_allStar_mdwarftype_table():
    
    from astra.contrib.mdwarftype.base import MDwarfTypeOutput

    fields = COMMON_STAR_FIELDS + [
        MDwarfTypeOutput.completed,
        MDwarfTypeOutput.time_elapsed,

        MDwarfTypeOutput.task,
        MDwarfTypeOutput.data_product,
        MDwarfTypeOutput.source,

        MDwarfTypeOutput.telescope,
        MDwarfTypeOutput.instrument,
        MDwarfTypeOutput.snr,

        MDwarfTypeOutput.spectral_type,
        MDwarfTypeOutput.sub_type,
        MDwarfTypeOutput.chi_sq,
        MDwarfTypeOutput.bitmask_flag.alias("flag"),
    ]

    category_headers = COMMON_CATEGORY_HEADERS + [
        ("task", "Metadata"),
        ("telescope", "Observations"),
        ("spectral_type", "Results"),
    ]   

    q = (
        StarMeta
        .select(*fields)
        .join(
            MDwarfTypeOutput,
            on=(
                (MDwarfTypeOutput.source_id == StarMeta.cat_id)
            &   (MDwarfTypeOutput.fiber.is_null())
            )
        )
        .dicts()
    )
    data = list_to_dict(q)
    return create_summary_table(data, fields, category_headers)



def create_allStar_snowwhite_table():
    
    from astra.contrib.snow_white.base import SnowWhite, SnowWhiteClassification

    fields = COMMON_STAR_FIELDS + [
        SnowWhite.completed,
        SnowWhite.time_elapsed,

        SnowWhite.task,
        SnowWhite.data_product,
        SnowWhite.source,
        SnowWhite.output_data_product,

        SnowWhite.telescope,
        SnowWhite.instrument,
        SnowWhite.snr,

        SnowWhiteClassification.wd_type,

        SnowWhite.teff,
        SnowWhite.logg,
        SnowWhite.e_logg,
        SnowWhite.v_rel,

        SnowWhite.conditioned_on_parallax,
        SnowWhite.conditioned_on_phot_g_mean_mag,

        SnowWhite.chi_sq,
        SnowWhite.reduced_chi_sq

    ]

    category_headers = COMMON_CATEGORY_HEADERS + [
        ("task", "Metadata"),
        ("telescope", "Observations"),
        ("wd_type", "Results"),
    ]   

    q = (
        StarMeta
        .select(*fields)
        .join(
            SnowWhiteClassification,
            on=(
                (SnowWhiteClassification.source_id == StarMeta.cat_id)
            &   (SnowWhiteClassification.fiber.is_null())
            )
        )
        .join(
            SnowWhite,
            JOIN.LEFT_OUTER,
            on=(SnowWhiteClassification.data_product_id == SnowWhite.data_product_id)
        )
        .dicts()
    )
    data = list_to_dict(q)    
    return create_summary_table(data, fields, category_headers)


def create_allStar_lineforest_table():

    from astra.contrib.lineforest.base import LineForestOutput

    q = (
        LineForestOutput
        .select(
            LineForestOutput.name,
            LineForestOutput.wavelength_vac
        )
        .distinct(LineForestOutput.name)
        .tuples()
    )

    lines = sorted(q, key=lambda x: x[1])

    fields = COMMON_STAR_FIELDS + [
        LineForestOutput.data_product,
        LineForestOutput.source,

        LineForestOutput.telescope,
        LineForestOutput.instrument,
        LineForestOutput.snr,        

    ]
    fill_value = -9999

    line_fields = []
    for name, wavelength in lines:
        line_fields.extend([
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw), ),
                fill_value
            )).alias(f"{name}_eqw"),
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs), ),
                fill_value
            )).alias(f"{name}_abs"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.detection_lower), ),
                fill_value
            )).alias(f"{name}_detection_lower"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.detection_upper), ),
                fill_value
            )).alias(f"{name}_detection_upper"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw_percentile_16), ),
                fill_value
            )).alias(f"{name}_eqw_percentile_16"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw_percentile_50), ),
                fill_value
            )).alias(f"{name}_eqw_percentile_50"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw_percentile_84), ),
                fill_value
            )).alias(f"{name}_eqw_percentile_84"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs_percentile_16), ),
                fill_value
            )).alias(f"{name}_abs_percentile_16"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs_percentile_50), ),
                fill_value
            )).alias(f"{name}_abs_percentile_50"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs_percentile_84), ),
                fill_value
            )).alias(f"{name}_abs_percentile_84"),   
        ])

    category_headers = COMMON_CATEGORY_HEADERS
    all_fields = fields + line_fields
    q = (
        StarMeta
        .select(*all_fields)
        .join(
            LineForestOutput,
            on=(
                (LineForestOutput.source_id == StarMeta.cat_id)
            &   (LineForestOutput.fiber.is_null())
            )
        )
        .group_by(*fields)
        .dicts()
    )
    data = list_to_dict(q)
    for field in line_fields:
        name = field.name
        values = np.array(data[name], dtype=np.float32)
        values[values == fill_value] = np.nan
        data[name] = list(values)

    return create_summary_table(data, all_fields, category_headers)


'''def create_allVisit_lineforest_table():

    from astra.contrib.lineforest.base import LineForestOutput

    q = (
        LineForestOutput
        .select(
            LineForestOutput.name,
            LineForestOutput.wavelength_vac
        )
        .distinct(LineForestOutput.name)
        .tuples()
    )

    lines = sorted(q, key=lambda x: x[1])

    fields = COMMON_STAR_FIELDS + [
        LineForestOutput.data_product,
        LineForestOutput.source,

        LineForestOutput.telescope,
        LineForestOutput.instrument,
        LineForestOutput.snr,        

    ]
    fill_value = -9999

    line_fields = []
    for name, wavelength in lines:
        line_fields.extend([
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw), ),
                fill_value
            )).alias(f"{name}_eqw"),
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs), ),
                fill_value
            )).alias(f"{name}_abs"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.detection_lower), ),
                fill_value
            )).alias(f"{name}_detection_lower"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.detection_upper), ),
                fill_value
            )).alias(f"{name}_detection_upper"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw_percentile_16), ),
                fill_value
            )).alias(f"{name}_eqw_percentile_16"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw_percentile_50), ),
                fill_value
            )).alias(f"{name}_eqw_percentile_50"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.eqw_percentile_84), ),
                fill_value
            )).alias(f"{name}_eqw_percentile_84"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs_percentile_16), ),
                fill_value
            )).alias(f"{name}_abs_percentile_16"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs_percentile_50), ),
                fill_value
            )).alias(f"{name}_abs_percentile_50"),   
            fn.MAX(Case(
                LineForestOutput.name,
                ((name, LineForestOutput.abs_percentile_84), ),
                fill_value
            )).alias(f"{name}_abs_percentile_84"),   
        ])

    category_headers = COMMON_CATEGORY_HEADERS
    all_fields = fields + line_fields
    q = (
        StarMeta
        .select(*all_fields)
        .join(
            LineForestOutput,
            on=(
                (LineForestOutput.source_id == StarMeta.cat_id)
            &   (LineForestOutput.fiber.is_null())
            )
        )
        .group_by(*fields)
        .dicts()
    )
    data = list_to_dict(q)
    for field in line_fields:
        name = field.name
        values = np.array(data[name], dtype=np.float32)
        values[values == fill_value] = np.nan
        data[name] = list(values)

    return create_summary_table(data, all_fields, category_headers)
'''





if __name__ == "__main__":
    from astra.utils import expand_path
    v_astra = "0.3.0"
    run2d = "v6_0_9"
    apred = "1.0"

    t = create_allStar_apogeenet_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-APOGEENet-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-APOGEENet")

    t = create_allStar_thepayne_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-ThePayne-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-ThePayne")


    t = create_allStar_mdwarftype_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-MDwarfType-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-MDwarfType")

    t = create_allStar_snowwhite_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-SnowWhite-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-SnowWhite")

    t = create_allStar_lineforest_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-LineForest-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-LineForest")

    t = create_allVisit_apogeenet_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allVisit-APOGEENet-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allVisit-APOGEENet")

