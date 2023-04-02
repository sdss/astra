import numpy as np
from astropy.io import fits
from astra.utils import list_to_dict
from astra.sdss.meta import StarMeta, VisitMeta
from astropy.table import Table
from astra.sdss.datamodels.base import add_check_sums, fits_column_kwargs, add_glossary_comments, add_table_category_headers, create_empty_hdu, metadata_cards
from tqdm import tqdm

from peewee import fn, Case, JOIN, Alias, ForeignKeyField, DateTimeField, BigIntegerField, FloatField, IntegerField, TextField, BooleanField

COMMON_CATEGORY_HEADERS = [
    ("cat_id", "Identifiers"),
    ("ra", "Astrometry"),
    ("g_mag", "Photometry"),
    ("carton_0", "Targeting"),
    ("doppler_teff", "Doppler"),
    ("xcsao_teff", "XCSAO"),
    ("astra_version", "Astra"),
]

COMMON_STAR_FIELDS = [
    StarMeta.cat_id,
    StarMeta.cat_id05,
    StarMeta.cat_id10,
    StarMeta.tic_id.alias("tic_v8_id"),
    StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
    StarMeta.sdss4_dr17_apogee_id,
    StarMeta.sdss4_dr17_field,
    StarMeta.sdss4_apogee_target1,
    StarMeta.sdss4_apogee_target2,
    StarMeta.sdss4_apogee2_target1,
    StarMeta.sdss4_apogee2_target2,
    StarMeta.sdss4_apogee2_target3,    
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
    
    # APOGEE-HDU-SPECIFIC
    
    StarMeta.doppler_teff,
    StarMeta.doppler_e_teff,
    StarMeta.doppler_logg,
    StarMeta.doppler_e_logg,
    StarMeta.doppler_fe_h,
    StarMeta.doppler_e_fe_h,
    StarMeta.doppler_starflag.alias("doppler_flag"),
    StarMeta.doppler_version,
    StarMeta.doppler_v_rad,
    StarMeta.doppler_v_scatter,
    StarMeta.doppler_v_err,
    StarMeta.doppler_n_good_visits,
    StarMeta.doppler_n_good_rvs,
    StarMeta.doppler_chi_sq,
    StarMeta.doppler_ccpfwhm,
    StarMeta.doppler_autofwhm,
    StarMeta.doppler_n_components,        

    # BOSS-HDU-SPECIFIC

    StarMeta.xcsao_teff,
    StarMeta.xcsao_e_teff,
    StarMeta.xcsao_logg,
    StarMeta.xcsao_e_logg,
    StarMeta.xcsao_fe_h,
    StarMeta.xcsao_e_fe_h,
    StarMeta.xcsao_rxc,
    StarMeta.xcsao_v_rad,
    StarMeta.xcsao_e_v_rad,


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
    #VisitMeta.nres,
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
    

]

def add_star_level_fps_and_dithered():
    
    from astra.database.astradb import database
    from tqdm import tqdm

    sq = (
        VisitMeta
        .select(
            VisitMeta.cat_id,
            VisitMeta.observatory,
            VisitMeta.instrument,
            fn.avg(VisitMeta.fps),
            fn.avg(VisitMeta.dithered),
        )
        .where(
            VisitMeta.in_stack
        )
        .group_by(VisitMeta.cat_id, VisitMeta.observatory, VisitMeta.instrument)
        .tuples()
    )

    fps_dithered = {}
    for cat_id, observatory, instrument, fps, dithered in tqdm(sq):
        fps_dithered.setdefault(cat_id, {})
        fps_dithered[cat_id][f"{observatory.lower()}25m_{instrument.lower()}"] = (fps, dithered or -1)
    

    q = (
        StarMeta
        .select(
            StarMeta.pk, 
            StarMeta.cat_id,
            StarMeta.fps_apo25m_apogee,
            StarMeta.dithered_apo25m_apogee,
            StarMeta.fps_lco25m_apogee,
            StarMeta.dithered_lco25m_apogee,
            StarMeta.fps_apo25m_boss,
            StarMeta.dithered_apo25m_boss,            
        )
    )
    
    items = []
    for item in tqdm(q):
        if item.cat_id not in fps_dithered:
            continue

        for key, (fps, dithered) in fps_dithered[item.cat_id].items():
            setattr(item, f"fps_{key.lower()}", fps)
            setattr(item, f"dithered_{key.lower()}", dithered)        
        items.append(item)
    
    with database.atomic():
        StarMeta.bulk_update(
            items, 
            fields=[
                StarMeta.fps_apo25m_apogee,
                StarMeta.dithered_apo25m_apogee,
                StarMeta.fps_lco25m_apogee,
                StarMeta.dithered_lco25m_apogee,
                StarMeta.fps_apo25m_boss,
                StarMeta.dithered_apo25m_boss,
            
            ],
            batch_size=1000
        )    



def add_obj_and_field_from_dr17():
    from astra.database.astradb import DataProduct

    sq = (
        DataProduct
        .select(
            DataProduct.kwargs["obj"],
        )
        .distinct(DataProduct.source_id)
        .where(
            (DataProduct.filetype == "apStar")
        &   (DataProduct.source_id == StarMeta.cat_id)
        )        
    )

    N = (
        StarMeta.update(sdss4_dr17_apogee_id=sq)
        .execute()
    )
    print(N)


    sq = (
        DataProduct
        .select(
            DataProduct.kwargs["field"],
        )
        .distinct(DataProduct.source_id)
        .where(
            (DataProduct.filetype == "apStar")
        &   (DataProduct.source_id == StarMeta.cat_id)
        )        
    )

    N = (
        StarMeta.update(sdss4_dr17_field=sq)
        .execute()
    )
    print(N)

    # add additional shit from doppler
    from astra.database.astradb import database
    from astra.database.apogee_drpdb import Star

    fields = (
        Star.apogee_target1.alias("_sdss4_apogee_target1"),
        Star.apogee_target2.alias("_sdss4_apogee_target2"),
        Star.apogee2_target1.alias("_sdss4_apogee2_target1"),
        Star.apogee2_target2.alias("_sdss4_apogee2_target2"),
        Star.apogee2_target3.alias("_sdss4_apogee2_target3"),
        Star.apogee2_target4.alias("_sdss4_apogee2_target4"),
        Star.ngoodvisits.alias("_doppler_n_good_visits"),
        Star.ngoodrvs.alias("_doppler_n_good_rvs"),
        Star.starflag.alias("_doppler_starflag"),
        Star.vscatter.alias("_doppler_v_scatter"),
        Star.vrad.alias("_doppler_v_rad"),
        Star.verr.alias("_doppler_v_err"),
        Star.chisq.alias("_doppler_chisq"),
        Star.rv_ccpfwhm.alias("_doppler_ccpfwhm"),
        Star.rv_autofwhm.alias("_doppler_autofwhm"),        
        Star.n_components.alias("_doppler_n_components"),
    )

    q = (
        StarMeta
        .select(StarMeta, *fields)
        .join(Star, on=(StarMeta.cat_id == Star.catalogid))
        .where(
            (Star.apred_vers == "1.0")
        )
    )

    items = []
    for item in q:
        for field in fields:
            setattr(item, field.name[1:], getattr(item, field.name))
        items.append(item)
    
    with database.atomic():
        StarMeta.bulk_update(
            items, 
            fields=[
                StarMeta.sdss4_apogee_target1,
                StarMeta.sdss4_apogee_target2,
                StarMeta.sdss4_apogee2_target1,
                StarMeta.sdss4_apogee2_target2,
                StarMeta.sdss4_apogee2_target3,
                StarMeta.doppler_n_good_visits,
                StarMeta.doppler_n_good_rvs,
                StarMeta.doppler_starflag,
                StarMeta.doppler_v_scatter,
                StarMeta.doppler_v_rad,
                StarMeta.doppler_v_err,
                StarMeta.doppler_chi_sq,
                StarMeta.doppler_ccpfwhm,
                StarMeta.doppler_autofwhm,
                StarMeta.doppler_n_components,
            ],
            batch_size=1000
        )

def add_sdss4_dr17_targeting():
    from astra.database.astradb import database
    from astra.database.catalogdb import CatalogdbModel
    
    class Visit(CatalogdbModel):

        class Meta:
            table_name = 'sdss_dr17_apogee_allvisits'
        

    q = (
        StarMeta
        .select(
            StarMeta.pk,
            StarMeta.sdss4_dr17_apogee_id,
            Visit.apogee_target1,
            Visit.apogee_target2,
            Visit.apogee2_target1,
            Visit.apogee2_target2,
            Visit.apogee2_target3,
            Visit.apogee2_target4
        )
        .join(Visit, on=(Visit.apogee_id == StarMeta.sdss4_dr17_apogee_id))
        .objects()
    )

    items = []
    for item in tqdm(q):
        item.sdss4_apogee_target1 = item.apogee_target1
        item.sdss4_apogee_target2 = item.apogee_target2
        item.sdss4_apogee2_target1 = item.apogee2_target1
        item.sdss4_apogee2_target2 = item.apogee2_target2
        item.sdss4_apogee2_target3 = item.apogee2_target3
        item.sdss4_apogee2_target4 = item.apogee2_target4
        items.append(item)

    with database.atomic():
        StarMeta.bulk_update(
            items, 
            fields=[
                StarMeta.sdss4_apogee_target1,
                StarMeta.sdss4_apogee_target2,
                StarMeta.sdss4_apogee2_target1,
                StarMeta.sdss4_apogee2_target2,
                StarMeta.sdss4_apogee2_target3,
                StarMeta.sdss4_apogee2_target4,
            ],
            batch_size=1000
        )
        



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
    observatory,
    instrument,
    upper=True,
    
):

    # TODO: When the same column name is given twice, use a temporary name, warn, and keep going (eg SNR_1, SNR_2)

    columns = [
        fits.Column(
            name="ASTRA_VERSION",
            array=["0.3.0"] * len(data["cat_id"]),
            unit=None,
            format="5A",
        )
    ]    
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

    header = metadata_cards(observatory, instrument)
    hdu = fits.BinTableHDU.from_columns(
        columns,
        header=fits.Header(header)
    )

    add_table_category_headers(hdu, category_headers, upper=upper)
    add_glossary_comments(hdu)

    return hdu

    primary = fits.PrimaryHDU()
    hdu_list = fits.HDUList([primary, hdu])
    add_check_sums(hdu_list)

    return hdu_list


    warnflag = (
        is_teff_error_large 
    |   is_logg_error_large 
    |   is_fe_h_error_large 
    |   is_missing_photometry
    )

    badflag = (
        is_teff_unreliable
    |   is_logg_unreliable
    |   is_fe_h_unreliable
    |   is_result_unreliable
    |   warnflag
    )
    

def create_allStar_cannon_table():
    
    from astra.contrib.thecannon.new_base import CannonOutput

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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
        
        # APOGEE-HDU-SPECIFIC
        
        StarMeta.doppler_teff,
        StarMeta.doppler_e_teff,
        StarMeta.doppler_logg,
        StarMeta.doppler_e_logg,
        StarMeta.doppler_fe_h,
        StarMeta.doppler_e_fe_h,
        StarMeta.doppler_starflag.alias("doppler_flag"),
        StarMeta.doppler_version,
        StarMeta.doppler_v_rad,
        StarMeta.doppler_v_scatter,
        StarMeta.doppler_v_err,
        StarMeta.doppler_n_good_visits,
        StarMeta.doppler_n_good_rvs,
        StarMeta.doppler_chi_sq,
        StarMeta.doppler_ccpfwhm,
        StarMeta.doppler_autofwhm,
        StarMeta.doppler_n_components,        

        # BOSS-HDU-SPECIFIC

        CannonOutput.task,
        CannonOutput.data_product,

        CannonOutput.teff,
        CannonOutput.logg,
        CannonOutput.fe_h,
        CannonOutput.vmicro.alias("v_micro"),
        CannonOutput.vbroad.alias("v_broad"),
        CannonOutput.c_fe,
        CannonOutput.n_fe,
        CannonOutput.o_fe,
        CannonOutput.na_fe,
        CannonOutput.mg_fe,
        CannonOutput.al_fe,
        CannonOutput.si_fe,
        CannonOutput.s_fe,
        CannonOutput.k_fe,
        CannonOutput.ca_fe,
        CannonOutput.ti_fe,
        CannonOutput.v_fe,
        CannonOutput.cr_fe,
        CannonOutput.mn_fe,
        CannonOutput.ni_fe,

        CannonOutput.e_teff,
        CannonOutput.e_logg,
        CannonOutput.e_fe_h,
        CannonOutput.e_vmicro.alias("e_v_micro"),
        CannonOutput.e_vbroad.alias("e_v_broad"),
        CannonOutput.e_c_fe,
        CannonOutput.e_n_fe,
        CannonOutput.e_o_fe,
        CannonOutput.e_na_fe,
        CannonOutput.e_mg_fe,
        CannonOutput.e_al_fe,
        CannonOutput.e_si_fe,
        CannonOutput.e_s_fe,
        CannonOutput.e_k_fe,
        CannonOutput.e_ca_fe,
        CannonOutput.e_ti_fe,
        CannonOutput.e_v_fe,
        CannonOutput.e_cr_fe,
        CannonOutput.e_mn_fe,
        CannonOutput.e_ni_fe,

        CannonOutput.snr,
        CannonOutput.chi_sq,
        CannonOutput.reduced_chi_sq,
        CannonOutput.ier,
        CannonOutput.nfev,
        CannonOutput.x0_index,
        CannonOutput.in_convex_hull,

        CannonOutput.bitmask,

        CannonOutput.bitmask_teff,
        CannonOutput.bitmask_logg,
        CannonOutput.bitmask_fe_h,
        CannonOutput.bitmask_vmicro.alias("bitmask_v_micro"),
        CannonOutput.bitmask_vbroad.alias("bitmask_v_broad"),
        CannonOutput.bitmask_c_fe,
        CannonOutput.bitmask_n_fe,
        CannonOutput.bitmask_o_fe,
        CannonOutput.bitmask_na_fe,
        CannonOutput.bitmask_mg_fe,
        CannonOutput.bitmask_al_fe,
        CannonOutput.bitmask_si_fe,
        CannonOutput.bitmask_s_fe,
        CannonOutput.bitmask_k_fe,
        CannonOutput.bitmask_ca_fe,
        CannonOutput.bitmask_ti_fe,
        CannonOutput.bitmask_v_fe,
        CannonOutput.bitmask_cr_fe,
        CannonOutput.bitmask_mn_fe,
        CannonOutput.bitmask_ni_fe,
    ]


    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("doppler_teff", "Doppler"),
        ("task", "Metadata"),
        ("snr", "Statistics"),
        ("bitmask", "Bitmasks"),
    ]

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "TheCannon", "Pipeline that created these results"),
        ])),
    ]
    t.extend([        
        create_empty_hdu("APO", "BOSS", is_data=False),
        create_empty_hdu("LCO", "BOSS", is_data=False),
    ])

    for telescope in ("apo25m", "lco25m"):
            
        q = (
            StarMeta
            .select(*fields)
            .join(
                CannonOutput,
                on=(
                    (CannonOutput.source_id == StarMeta.cat_id)
                &   (CannonOutput.fiber.is_null())
                )
            )
            .dicts()
        )

        data = list_to_dict(q)
        t.append(
            create_summary_table(data, fields, category_headers, observatory=telescope[:3].upper(), instrument="APOGEE")
        )

    add_check_sums(t)
    t = fits.HDUList(t)
    return t
    

def create_allVisit_classifier_table():

    from astra.contrib.classifier.base import ClassifierOutput
    

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "Classifier", "Pipeline that created these results"),
        ])),
    ]

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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
        
        
        # BOSS-HDU-SPECIFIC

        # File type stuff
        VisitMeta.release,
        VisitMeta.filetype,
        VisitMeta.run2d,
        VisitMeta.fieldid,
        VisitMeta.isplate,
        VisitMeta.catalogid,
        VisitMeta.mjd,

        # Common stuff.
        #VisitMeta.hdu_data_index,

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
        VisitMeta.fps,
        VisitMeta.in_stack,
        VisitMeta.v_shift,


        VisitMeta.xcsao_teff,
        VisitMeta.xcsao_e_teff,
        VisitMeta.xcsao_logg,
        VisitMeta.xcsao_e_logg,
        VisitMeta.xcsao_fe_h,
        VisitMeta.xcsao_e_fe_h,
        VisitMeta.xcsao_rxc    ,

        ClassifierOutput.task,
        ClassifierOutput.data_product,

        VisitMeta.snr,
        ClassifierOutput.p_cv,
        ClassifierOutput.p_fgkm,
        ClassifierOutput.p_hotstar,
        ClassifierOutput.p_wd,
        ClassifierOutput.p_sb2,
        ClassifierOutput.p_yso,
        
        ClassifierOutput.lp_cv,
        ClassifierOutput.lp_fgkm,
        ClassifierOutput.lp_hotstar,
        ClassifierOutput.lp_wd,
        ClassifierOutput.lp_sb2,
        ClassifierOutput.lp_yso,
    ]
    q = (
        VisitMeta
        .select(*fields)
        .join(
            ClassifierOutput,
            JOIN.LEFT_OUTER,
            on=(
                (ClassifierOutput.source_id == VisitMeta.cat_id)
            &   (ClassifierOutput.mjd == VisitMeta.mjd)
            )
        )
        .join(
            StarMeta,
            on=(
                (StarMeta.cat_id == VisitMeta.cat_id)
            )
        )
        .where(
            (ClassifierOutput.instrument == "BOSS")
        )
        .dicts()
    )

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("release", "Data Product Keywords"),
        ("alt", "Observing conditions"),
        ("v_boss", "Data reduction pipeline"),
        ("xcsao_teff", "XCSAO"),
        ("carton_0", "Targeting"),
        ("task", "Metadata"),
        ("p_cv", "Classification Probabilities"),
        ("lp_cv", "Classification Log Probabilities"),
    ]   

    data = list_to_dict(q)
    t.append(
        create_summary_table(data, fields, category_headers, observatory="APO", instrument="BOSS")
    )
    t.append(create_empty_hdu("LCO", "BOSS", is_data=False))

    
    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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
        
        # APOGEE-HDU-SPECIFIC


        # File type stuff
        VisitMeta.release,
        VisitMeta.filetype,
        VisitMeta.plate,
        VisitMeta.fiber,
        VisitMeta.field,
        VisitMeta.apred,
        VisitMeta.prefix,
        VisitMeta.mjd,


        # APOGEE-level stuff.
        VisitMeta.v_apred,
        VisitMeta.date_obs,
        VisitMeta.exptime,
        VisitMeta.fluxflam,
        VisitMeta.npairs,
        VisitMeta.fps,
        VisitMeta.dithered,
        VisitMeta.filtsize,
        VisitMeta.normsize,
        VisitMeta.conscale,
        VisitMeta.visit_pk,

        # Doppler results.
        StarMeta.doppler_teff,
        StarMeta.doppler_e_teff,
        StarMeta.doppler_logg,
        StarMeta.doppler_e_logg,
        StarMeta.doppler_fe_h,
        StarMeta.doppler_e_fe_h,
        StarMeta.doppler_starflag.alias("doppler_flag"),
        StarMeta.doppler_version,
        StarMeta.doppler_v_rad,
        StarMeta.doppler_v_scatter,
        StarMeta.doppler_v_err,
        StarMeta.doppler_n_good_visits,
        StarMeta.doppler_n_good_rvs,
        StarMeta.doppler_chi_sq,
        StarMeta.doppler_ccpfwhm,
        StarMeta.doppler_autofwhm,
        StarMeta.doppler_n_components,       

        VisitMeta.jd,
        VisitMeta.v_rad,
        VisitMeta.e_v_rad,
        VisitMeta.v_rel,
        VisitMeta.v_bc,
        VisitMeta.rchisq,
        VisitMeta.n_rv_components,
        VisitMeta.rv_visit_pk,
        VisitMeta.in_stack,
        VisitMeta.v_shift,

        ClassifierOutput.task,
        ClassifierOutput.data_product,
        VisitMeta.snr,

        ClassifierOutput.p_cv,
        ClassifierOutput.p_fgkm,
        ClassifierOutput.p_hotstar,
        ClassifierOutput.p_wd,
        ClassifierOutput.p_sb2,
        ClassifierOutput.p_yso,

        ClassifierOutput.lp_yso,         
        ClassifierOutput.lp_cv,
        ClassifierOutput.lp_fgkm,
        ClassifierOutput.lp_hotstar,
        ClassifierOutput.lp_wd,
        ClassifierOutput.lp_sb2,
        
    ]
    for telescope in ("apo25m", ):
        q = (
            ClassifierOutput
            .select(*fields)
            .join(
                VisitMeta,
                JOIN.LEFT_OUTER,
                on=(
                    (ClassifierOutput.source_id == VisitMeta.cat_id)
                &   (ClassifierOutput.mjd == VisitMeta.mjd)
                &   (ClassifierOutput.plate == VisitMeta.plate.cast('TEXT'))
                &   (ClassifierOutput.field == VisitMeta.field)
                &   (ClassifierOutput.fiber == VisitMeta.fiber)
                )
            )
            .join(
                StarMeta,
                on=(StarMeta.cat_id == VisitMeta.cat_id)
            )
            .where(
                (ClassifierOutput.telescope == telescope)
            &   (ClassifierOutput.instrument == "APOGEE")
            &   (ClassifierOutput.fiber.is_null(False))
            )
            .dicts()
        )        
        category_headers = [
            ("cat_id", "Identifiers"),
            ("ra", "Astrometry"),
            ("g_mag", "Photometry"),
            ("release", "Data Product Keywords"),
            ("v_apred", "APOGEE Data reduction pipeline"),
            ("doppler_teff", "Doppler"),
            ("jd", "Doppler radial velocity"),
            ("carton_0", "Targeting"),
            ("task", "Metadata"),
            ("p_cv", "Classification Probabilities"),
            ("lp_cv", "Classification Log Probabilities"),
        ]   
        data = list_to_dict(q)
        t.append(
            create_summary_table(data, fields, category_headers, observatory=telescope[:3].upper(), instrument="APOGEE")
        )

    t.append(create_empty_hdu("LCO", "APOGEE", is_data=False))
    add_check_sums(t)
    t = fits.HDUList(t)
    return t    



def create_allStar_classifier_table():

    from astra.contrib.classifier.base import StarClassification

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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

        StarClassification.most_probable_class,
        StarClassification.num_apogee_classifications,
        StarClassification.num_boss_classifications,
        StarClassification.p_cv,
        StarClassification.p_fgkm,
        StarClassification.p_hotstar,
        StarClassification.p_wd,
        StarClassification.p_sb2,
        StarClassification.p_yso,
        StarClassification.lp_cv,
        StarClassification.lp_fgkm,
        StarClassification.lp_hotstar,
        StarClassification.lp_wd,
        StarClassification.lp_sb2,
        StarClassification.lp_yso,        
    ]

    
    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("most_probable_class", "Classification"),
        ("p_cv", "Classification Probabilities"),
        ("lp_cv", "Classification Log Probabilities"),
    ]   


    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "Classifier", "Pipeline that created these results"),
        ])),
    ]

    q = (
        StarMeta
        .select(*fields)
        .join(
            StarClassification,
            on=(
                (StarClassification.source_id == StarMeta.cat_id)
            )
        )
        .dicts()
    )

    data = list_to_dict(q)
    t.append(
        create_summary_table(data, fields, category_headers, observatory="APO+LCO", instrument="BOSS+APOGEE")
    )
    add_check_sums(t)
    t = fits.HDUList(t)
    return t    


def create_allStar_slam_table():
    from astra.contrib.slam.base import SlamOutput
    from astra.database.astradb import database

    with database.atomic():
        N = (
            SlamOutput
            .update({ SlamOutput.warnflag: True })
            .where(SlamOutput.status.in_((0, 1, 3, 4)))
            .execute()
        )
        print(N)
        N = (
            SlamOutput
            .update({ SlamOutput.badflag: True })
            .where(SlamOutput.status < 0)
            .execute()
        )
        print(N)

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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

        # BOSS-HDU-SPECIFIC

        StarMeta.xcsao_teff,
        StarMeta.xcsao_e_teff,
        StarMeta.xcsao_logg,
        StarMeta.xcsao_e_logg,
        StarMeta.xcsao_fe_h,
        StarMeta.xcsao_e_fe_h,
        StarMeta.xcsao_rxc,
        StarMeta.xcsao_v_rad,
        StarMeta.xcsao_e_v_rad,
                
        SlamOutput.task,
        SlamOutput.data_product,
        SlamOutput.output_data_product,

        
        # Initial values.
        
        SlamOutput.teff,
        SlamOutput.e_teff,
        SlamOutput.logg,
        SlamOutput.e_logg,
        SlamOutput.fe_h,
        SlamOutput.e_fe_h,
        
        SlamOutput.rho_teff_logg,
        SlamOutput.rho_teff_fe_h,
        SlamOutput.rho_logg_fe_h,
        
        # Optimisation outputs.
        SlamOutput.success,
        SlamOutput.status,
        SlamOutput.optimality,
        SlamOutput.warnflag,
        SlamOutput.badflag,

        # Statistics.
        SlamOutput.snr,
        SlamOutput.chi_sq,
        SlamOutput.reduced_chi_sq,

        # Initial values
        SlamOutput.initial_teff,
        SlamOutput.initial_logg,
        SlamOutput.initial_fe_h,
    ]



    

    category_headers = [
        ("task", "Metadata"),

        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("xcsao_teff", "XCSAO"),                
        ("teff", "Stellar Parameters"),
        ("success", "Optimization outputs"),
        ("snr", "Statistics"),
        ("initial_teff", "Initial values")
    ]   


    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "SLAM", "Pipeline that created these results"),
        ])),
    ]

    q = (
        StarMeta
        .select(*fields)
        .join(
            SlamOutput,
            on=(
                (SlamOutput.source_id == StarMeta.cat_id)
            &   (SlamOutput.fiber.is_null())
            )
        )
        .dicts()
    )

    data = list_to_dict(q)
    t.append(
        create_summary_table(data, fields, category_headers, "APO", "BOSS")
    )
    t.extend([
        create_empty_hdu("LCO", "BOSS", is_data=False),
        create_empty_hdu("APO", "APOGEE", is_data=False),
        create_empty_hdu("LCO", "APOGEE", is_data=False),
    ])
    add_check_sums(t)
    t = fits.HDUList(t)
    return t    



def create_allStar_aspcap_table():
    
    from astra.contrib.aspcap.models import ASPCAPOutput

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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

        # APOGEE-HDU-SPECIFIC
        
        StarMeta.doppler_teff,
        StarMeta.doppler_e_teff,
        StarMeta.doppler_logg,
        StarMeta.doppler_e_logg,
        StarMeta.doppler_fe_h,
        StarMeta.doppler_e_fe_h,
        StarMeta.doppler_starflag,
        StarMeta.doppler_version,
        StarMeta.doppler_v_rad,
        StarMeta.doppler_v_scatter,
        StarMeta.doppler_v_err,
        StarMeta.doppler_n_good_visits,
        StarMeta.doppler_n_good_rvs,
        StarMeta.doppler_chi_sq,
        StarMeta.doppler_ccpfwhm,
        StarMeta.doppler_autofwhm,
        StarMeta.doppler_n_components,        


        ASPCAPOutput.task,
        ASPCAPOutput.data_product,

        #ASPCAPOutput.obj,
        #ASPCAPOutput.apstar_pk,

        #ASPCAPOutput.#output_data_product,
        #ASPCAPOutput.grid,


        ASPCAPOutput.teff,
        ASPCAPOutput.e_teff,
        ASPCAPOutput.bitmask_teff,

        ASPCAPOutput.logg,
        ASPCAPOutput.e_logg,
        ASPCAPOutput.bitmask_logg,

        ASPCAPOutput.m_h,
        ASPCAPOutput.e_m_h,
        ASPCAPOutput.bitmask_m_h,

        ASPCAPOutput.v_sini,
        ASPCAPOutput.e_v_sini,
        ASPCAPOutput.bitmask_v_sini,

        ASPCAPOutput.v_micro,
        ASPCAPOutput.e_v_micro,
        ASPCAPOutput.bitmask_v_micro,

        ASPCAPOutput.c_m_atm,
        ASPCAPOutput.e_c_m_atm,
        ASPCAPOutput.bitmask_c_m_atm,

        ASPCAPOutput.n_m_atm,
        ASPCAPOutput.e_n_m_atm,
        ASPCAPOutput.bitmask_n_m_atm,

        ASPCAPOutput.alpha_m_atm,
        ASPCAPOutput.e_alpha_m_atm,
        ASPCAPOutput.bitmask_alpha_m_atm,


        # elemental abundances.
        ASPCAPOutput.al_h,
        ASPCAPOutput.e_al_h,
        ASPCAPOutput.bitmask_al_h,
        ASPCAPOutput.chisq_al_h,

        ASPCAPOutput.c13_h,
        ASPCAPOutput.e_c13_h,
        ASPCAPOutput.bitmask_c13_h,
        ASPCAPOutput.chisq_c13_h,

        # TODO: Initial values used?
        # TODO: covariances?

        ASPCAPOutput.ca_h,
        ASPCAPOutput.e_ca_h,
        ASPCAPOutput.bitmask_ca_h,
        ASPCAPOutput.chisq_ca_h,

        ASPCAPOutput.ce_h,
        ASPCAPOutput.e_ce_h,
        ASPCAPOutput.bitmask_ce_h,
        ASPCAPOutput.chisq_ce_h,

        ASPCAPOutput.c1_h,
        ASPCAPOutput.e_c1_h,
        ASPCAPOutput.bitmask_c1_h,
        ASPCAPOutput.chisq_c1_h,

        ASPCAPOutput.c_h,
        ASPCAPOutput.e_c_h,
        ASPCAPOutput.bitmask_c_h,
        ASPCAPOutput.chisq_c_h,

        ASPCAPOutput.co_h,
        ASPCAPOutput.e_co_h,
        ASPCAPOutput.bitmask_co_h,
        ASPCAPOutput.chisq_co_h,

        ASPCAPOutput.cr_h,
        ASPCAPOutput.e_cr_h,
        ASPCAPOutput.bitmask_cr_h,
        ASPCAPOutput.chisq_cr_h,

        ASPCAPOutput.cu_h,
        ASPCAPOutput.e_cu_h,
        ASPCAPOutput.bitmask_cu_h,
        ASPCAPOutput.chisq_cu_h,

        ASPCAPOutput.fe_h,
        ASPCAPOutput.e_fe_h,
        ASPCAPOutput.bitmask_fe_h,
        ASPCAPOutput.chisq_fe_h,

        ASPCAPOutput.k_h,
        ASPCAPOutput.e_k_h,
        ASPCAPOutput.bitmask_k_h,
        ASPCAPOutput.chisq_k_h,

        ASPCAPOutput.mg_h,
        ASPCAPOutput.e_mg_h,
        ASPCAPOutput.bitmask_mg_h,
        ASPCAPOutput.chisq_mg_h,

        ASPCAPOutput.mn_h,
        ASPCAPOutput.e_mn_h,
        ASPCAPOutput.bitmask_mn_h,
        ASPCAPOutput.chisq_mn_h,

        ASPCAPOutput.na_h,
        ASPCAPOutput.e_na_h,
        ASPCAPOutput.bitmask_na_h,
        ASPCAPOutput.chisq_na_h,

        ASPCAPOutput.nd_h,
        ASPCAPOutput.e_nd_h,
        ASPCAPOutput.bitmask_nd_h,
        ASPCAPOutput.chisq_nd_h,

        ASPCAPOutput.ni_h,
        ASPCAPOutput.e_ni_h,
        ASPCAPOutput.bitmask_ni_h,
        ASPCAPOutput.chisq_ni_h,

        ASPCAPOutput.n_h,
        ASPCAPOutput.e_n_h,
        ASPCAPOutput.bitmask_n_h,
        ASPCAPOutput.chisq_n_h,

        ASPCAPOutput.o_h,
        ASPCAPOutput.e_o_h,
        ASPCAPOutput.bitmask_o_h,
        ASPCAPOutput.chisq_o_h,

        ASPCAPOutput.p_h,
        ASPCAPOutput.e_p_h,
        ASPCAPOutput.bitmask_p_h,
        ASPCAPOutput.chisq_p_h,

        ASPCAPOutput.si_h,
        ASPCAPOutput.e_si_h,
        ASPCAPOutput.bitmask_si_h,
        ASPCAPOutput.chisq_si_h,

        ASPCAPOutput.s_h,
        ASPCAPOutput.e_s_h,
        ASPCAPOutput.bitmask_s_h,
        ASPCAPOutput.chisq_s_h,

        ASPCAPOutput.ti_h,
        ASPCAPOutput.e_ti_h,
        ASPCAPOutput.bitmask_ti_h,
        ASPCAPOutput.chisq_ti_h,

        ASPCAPOutput.ti2_h,
        ASPCAPOutput.e_ti2_h,
        ASPCAPOutput.bitmask_ti2_h,
        ASPCAPOutput.chisq_ti2_h,

        ASPCAPOutput.v_h,
        ASPCAPOutput.e_v_h,
        ASPCAPOutput.bitmask_v_h,
        ASPCAPOutput.chisq_v_h,

        # Summary statistics for stellar parameter fits.
        ASPCAPOutput.snr,
        ASPCAPOutput.chisq,
        ASPCAPOutput.bitmask_aspcap,
        ASPCAPOutput.warnflag,
        ASPCAPOutput.badflag,
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("doppler_teff", "Doppler"),

        ("task", "Metadata"),
        #("model_path", "Task Parameters"),
        #("telescope", "Observations"),
        ("teff", "Stellar Parameters"),
        ("snr", "Statistics and flags"),
    ]

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "ASPCAP", "Pipeline that created these results"),
        ])),
        create_empty_hdu("APO", "BOSS", is_data=False),
        create_empty_hdu("LCO", "BOSS", is_data=False)
    ]
    for hdu, telescope in enumerate(("apo25m", "lco25m"), start=3):
        q = (
            StarMeta
            .select(*fields)
            .join(
                ASPCAPOutput,
                on=(
                    (ASPCAPOutput.source_id == StarMeta.cat_id)
                )
            )
            .where(ASPCAPOutput.hdu == hdu)
            .dicts()
        )

        data = list_to_dict(q)
        t.append(create_summary_table(
            data, 
            fields, 
            category_headers,
            observatory=telescope[:3].upper(),
            instrument="APOGEE"
            ))
    
    add_check_sums(t)
    t = fits.HDUList(t)
    return t



def create_allStar_apogeenet_table():
    
    from astra.contrib.apogeenet.base import ApogeeNetOutput

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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

        # APOGEE-HDU-SPECIFIC
        
        StarMeta.doppler_teff,
        StarMeta.doppler_e_teff,
        StarMeta.doppler_logg,
        StarMeta.doppler_e_logg,
        StarMeta.doppler_fe_h,
        StarMeta.doppler_e_fe_h,
        StarMeta.doppler_starflag,
        StarMeta.doppler_version,
        StarMeta.doppler_v_rad,
        StarMeta.doppler_v_scatter,
        StarMeta.doppler_v_err,
        StarMeta.doppler_n_good_visits,
        StarMeta.doppler_n_good_rvs,
        StarMeta.doppler_chi_sq,
        StarMeta.doppler_ccpfwhm,
        StarMeta.doppler_autofwhm,
        StarMeta.doppler_n_components,        

        ApogeeNetOutput.task,
        ApogeeNetOutput.data_product,

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
        ApogeeNetOutput.bitmask_flag.alias("bitmask"),
        ApogeeNetOutput.warnflag.alias("warnflag"),
        ApogeeNetOutput.badflag.alias("badflag"),
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("doppler_teff", "Doppler"),

        ("task", "Metadata"),
        #("model_path", "Task Parameters"),
        #("telescope", "Observations"),
        ("teff", "Stellar Parameters"),
    ]

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "APOGEENet", "Pipeline that created these results"),
        ])),
        create_empty_hdu("APO", "BOSS", is_data=False),
        create_empty_hdu("LCO", "BOSS", is_data=False)
    ]
    for telescope in ("apo25m", "lco25m"):
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
            .where(ApogeeNetOutput.telescope == telescope)
            .dicts()
        )

        data = list_to_dict(q)
        t.append(create_summary_table(
            data, 
            fields, 
            category_headers,
            observatory=telescope[:3].upper(),
            instrument="APOGEE"
            ))
    
    add_check_sums(t)
    t = fits.HDUList(t)
    return t





def create_allVisit_apogeenet_table():
    
    from astra.contrib.apogeenet.base import ApogeeNetOutput

    fields = [
                        
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
        StarMeta.healpix,


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
        
        VisitMeta.release,
        VisitMeta.filetype,
        VisitMeta.plate,
        VisitMeta.fiber,
        VisitMeta.field,
        VisitMeta.apred,
        VisitMeta.prefix,
        VisitMeta.mjd,    
        
        # APOGEE-level stuff.
        VisitMeta.v_apred,
        VisitMeta.nres,
        VisitMeta.filtsize,
        VisitMeta.normsize,
        VisitMeta.conscale,
        
        VisitMeta.date_obs,
        VisitMeta.exptime,
        VisitMeta.fluxflam,
        VisitMeta.npairs,
        VisitMeta.dithered,
        VisitMeta.fps,

        StarMeta.doppler_teff,
        StarMeta.doppler_e_teff,
        StarMeta.doppler_logg,
        StarMeta.doppler_e_logg,
        StarMeta.doppler_fe_h,
        StarMeta.doppler_e_fe_h,
        StarMeta.doppler_starflag,
        StarMeta.doppler_version,
        StarMeta.doppler_v_rad,
        StarMeta.doppler_v_scatter,
        StarMeta.doppler_v_err,
        StarMeta.doppler_n_good_visits,
        StarMeta.doppler_n_good_rvs,
        StarMeta.doppler_chi_sq,
        StarMeta.doppler_ccpfwhm,
        StarMeta.doppler_autofwhm,
        StarMeta.doppler_n_components,                
        VisitMeta.jd,
        VisitMeta.v_rad,
        VisitMeta.e_v_rad,
        VisitMeta.v_rel,
        VisitMeta.v_bc,
        VisitMeta.rchisq,
        VisitMeta.n_rv_components,
        VisitMeta.in_stack,
        VisitMeta.v_shift,        
        VisitMeta.visit_pk,
        VisitMeta.rv_visit_pk,
    
        ApogeeNetOutput.task,
        ApogeeNetOutput.data_product,

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
        ApogeeNetOutput.bitmask_flag.alias("bitmask"),
        ApogeeNetOutput.warnflag.alias("warnflag"),
        ApogeeNetOutput.badflag.alias("badflag"),
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("doppler_teff", "Doppler"),
        ("v_apred", "APOGEE Data Reduction Pipeline"),
        ("task", "Metadata"),
        ("date_obs", "Observing conditions"),
        #("model_path", "Task Parameters"),
        #("telescope", "Observations"),
        ("teff", "Stellar Parameters"),
    ]

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "APOGEENet", "Pipeline that created these results"),
        ])),
        create_empty_hdu("APO", "BOSS", is_data=False),
        create_empty_hdu("LCO", "BOSS", is_data=False)
    ]
    for telescope in ("apo25m", "lco25m"):
        q = (
            ApogeeNetOutput
            .select(*fields)
            .join(
                VisitMeta,
                on=(
                    (ApogeeNetOutput.source_id == VisitMeta.cat_id)
                #&   (ApogeeNetOutput.telescope == VisitMeta.telescope)
                &   (ApogeeNetOutput.plate.cast("TEXT") == VisitMeta.plate.cast("TEXT"))
                &   (ApogeeNetOutput.fiber == VisitMeta.fiber)
                &   (ApogeeNetOutput.field == VisitMeta.field)
                &   (ApogeeNetOutput.mjd == VisitMeta.mjd)
                )
            )
            .join(
                StarMeta,
                on=(
                    (VisitMeta.cat_id == StarMeta.cat_id)
                )
            )
            .where(
                (ApogeeNetOutput.telescope == telescope)
            &   (ApogeeNetOutput.instrument == "APOGEE")
            &   (VisitMeta.filetype == "apVisit")
            &   (VisitMeta.observatory == telescope[:3].upper())
            )
            .dicts()
        )

        data = list_to_dict(q)
        t.append(create_summary_table(
            data, 
            fields, 
            category_headers,
            observatory=telescope[:3].upper(),
            instrument="APOGEE"
            ))
    
    add_check_sums(t)
    t = fits.HDUList(t)
    return t



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

    fields =  [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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
        
        # APOGEE-HDU-SPECIFIC
        
        StarMeta.doppler_teff,
        StarMeta.doppler_e_teff,
        StarMeta.doppler_logg,
        StarMeta.doppler_e_logg,
        StarMeta.doppler_fe_h,
        StarMeta.doppler_e_fe_h,
        StarMeta.doppler_starflag.alias("doppler_flag"),
        StarMeta.doppler_version,
        StarMeta.doppler_v_rad,
        StarMeta.doppler_v_scatter,
        StarMeta.doppler_v_err,
        StarMeta.doppler_n_good_visits,
        StarMeta.doppler_n_good_rvs,
        StarMeta.doppler_chi_sq,
        StarMeta.doppler_ccpfwhm,
        StarMeta.doppler_autofwhm,
        StarMeta.doppler_n_components,           

        ThePayne.task,
        ThePayne.data_product,
        ThePayne.source,
        ThePayne.output_data_product,

        #ThePayne.model_path,
        #ThePayne.mask_path,
        #ThePayne.opt_tolerance,
        #ThePayne.v_rad_tolerance,

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

    category_headers =  [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("doppler_teff", "Doppler"),
        ("task", "Metadata"),
        #("model_path", "Task Parameters"),
        ("teff", "Stellar Labels"),
        ("flag", "Flags"),
        ("reduced_chi_sq", "Summary Statistics"),
    ]   

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "ThePayne", "Pipeline that created these results"),
        ])),
        create_empty_hdu("APO", "BOSS", is_data=False),
        create_empty_hdu("LCO", "BOSS", is_data=False),
    ]

    for telescope in ("apo25m", "lco25m"):
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
            .where(ThePayne.telescope == telescope)
            .dicts()
        )
        data = list_to_dict(q)
        t.append(create_summary_table(data, fields, category_headers, observatory=telescope[:3].upper(), instrument="APOGEE"))

    add_check_sums(t)
    t = fits.HDUList(t)
    return t



def create_allStar_mdwarftype_table():
    
    from astra.contrib.mdwarftype.base import MDwarfTypeOutput

    fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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

        # BOSS-HDU-SPECIFIC

        StarMeta.xcsao_teff,
        StarMeta.xcsao_e_teff,
        StarMeta.xcsao_logg,
        StarMeta.xcsao_e_logg,
        StarMeta.xcsao_fe_h,
        StarMeta.xcsao_e_fe_h,
        StarMeta.xcsao_rxc,
        StarMeta.xcsao_v_rad,
        StarMeta.xcsao_e_v_rad,
                
        MDwarfTypeOutput.task,
        MDwarfTypeOutput.data_product,

        MDwarfTypeOutput.snr,

        MDwarfTypeOutput.spectral_type,
        MDwarfTypeOutput.sub_type,
        MDwarfTypeOutput.chi_sq,
        MDwarfTypeOutput.bitmask_flag.alias("badflag"),
        #MDwarfTypeOutput.bitmask_flag.cast("BOOLEAN").alias("badflag"),
    ]

    category_headers = [
        ("task", "Metadata"),

        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("xcsao_teff", "XCSAO"),                
        ("spectral_type", "Results"),
    ]   


    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "MDwarfType", "Pipeline that created these results"),
        ])),
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
    t.append(
        create_summary_table(data, fields, category_headers, "APO", "BOSS")
    )
    t.extend([
        create_empty_hdu("LCO", "BOSS", is_data=False),
        create_empty_hdu("APO", "APOGEE", is_data=False),
        create_empty_hdu("LCO", "APOGEE", is_data=False),
    ])
    add_check_sums(t)
    t = fits.HDUList(t)
    return t    




def create_allStar_snowwhite_table():
    
    from astra.contrib.snow_white.base import SnowWhite, SnowWhiteClassification

    fields = [

        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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
                
        # BOSS-HDU-SPECIFIC

        StarMeta.xcsao_teff,
        StarMeta.xcsao_e_teff,
        StarMeta.xcsao_logg,
        StarMeta.xcsao_e_logg,
        StarMeta.xcsao_fe_h,
        StarMeta.xcsao_e_fe_h,
        StarMeta.xcsao_rxc,
        StarMeta.xcsao_v_rad,
        StarMeta.xcsao_e_v_rad,        

        SnowWhite.task,
        SnowWhite.data_product,
        SnowWhite.output_data_product,

        SnowWhite.snr,

        SnowWhiteClassification.wd_type,

        SnowWhite.teff,
        SnowWhite.e_teff,
        SnowWhite.logg,
        SnowWhite.e_logg,
        SnowWhite.v_rel,

        SnowWhite.conditioned_on_parallax,
        SnowWhite.conditioned_on_phot_g_mean_mag,

        SnowWhite.chi_sq,
        SnowWhite.reduced_chi_sq
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("xcsao_teff", "XCSAO"),
        ("task", "Metadata"),
        ("wd_type", "Results"),
    ]   


    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "SnowWhite", "Pipeline that created these results"),
        ])),
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
    t.append(create_summary_table(data, fields, category_headers, "APO", "BOSS"))

    t.extend([
        create_empty_hdu("LCO", "BOSS", is_data=False),
        create_empty_hdu("APO", "APOGEE", is_data=False),
        create_empty_hdu("LCO", "APOGEE", is_data=False),
    ])
    add_check_sums(t)
    t = fits.HDUList(t)
    return t




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

    fields = [   
        LineForestOutput.source_id,    
        LineForestOutput.data_product,
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

    LINES = [
        ['Halpha',      'hlines.model', 6562.8, 200],
        ['Hbeta',       'hlines.model', 4861.3, 200],
        ['Hgamma',      'hlines.model', 4340.5, 200],
        ['Hdelta',      'hlines.model', 4101.7, 200],
        ['Hepsilon',    'hlines.model', 3970.1, 200],
        ['H8',          'hlines.model', 3889.064, 200],
        ['H9',          'hlines.model', 3835.391, 200],
        ['H10',         'hlines.model', 3797.904, 200],
        ['H11',         'hlines.model', 3770.637, 200],
        ['H12',         'zlines.model', 3750.158, 50],
        ['H13',         'zlines.model', 3734.369, 50],
        ['H14',         'zlines.model', 3721.945, 50],
        ['H15',         'zlines.model', 3711.977, 50],
        ['H16',         'zlines.model', 3703.859, 50],
        ['H17',         'zlines.model', 3697.157, 50],
        ['Pa7',         'hlines.model', 10049.4889, 200],
        ['Pa8',         'hlines.model', 9546.0808, 200],
        ['Pa9',         'hlines.model', 9229.12, 200],
        ['Pa10',        'hlines.model', 9014.909, 200],
        ['Pa11',        'hlines.model', 8862.782, 200],
        ['Pa12',        'hlines.model', 8750.472, 200],
        ['Pa13',        'hlines.model', 8665.019, 200],
        ['Pa14',        'hlines.model', 8598.392, 200],
        ['Pa15',        'hlines.model', 8545.383, 200],
        ['Pa16',        'hlines.model', 8502.483, 200],
        ['Pa17',        'hlines.model', 8467.254, 200],
        ['CaII8662',    'zlines.model', 8662.14, 50],
        ['CaII8542',    'zlines.model', 8542.089, 50],
        ['CaII8498',    'zlines.model', 8498.018, 50],
        ['CaK3933',     'hlines.model', 3933.6614, 200],
        ['CaH3968',     'hlines.model', 3968.4673, 200],
        ['HeI6678',     'zlines.model', 6678.151, 50],
        ['HeI5875',     'zlines.model', 5875.621, 50],
        ['HeI5015',     'zlines.model', 5015.678, 50],
        ['HeI4471',     'zlines.model', 4471.479, 50],
        ['HeII4685',    'zlines.model', 4685.7, 50],
        ['NII6583',     'zlines.model', 6583.45, 50],
        ['NII6548',     'zlines.model', 6548.05, 50],
        ['SII6716',     'zlines.model', 6716.44, 50],
        ['SII6730',     'zlines.model', 6730.816, 50],
        ['FeII5018',    'zlines.model', 5018.434, 50],
        ['FeII5169',    'zlines.model', 5169.03, 50],
        ['FeII5197',    'zlines.model', 5197.577, 50],
        ['FeII6432',    'zlines.model', 6432.68, 50],
        ['OI5577',      'zlines.model', 5577.339, 50],
        ['OI6300',      'zlines.model', 6300.304, 50],
        ['OI6363',      'zlines.model', 6363.777, 50],
        ['OII3727',     'zlines.model', 3727.42, 50],
        ['OIII4959',    'zlines.model', 4958.911, 50],
        ['OIII5006',    'zlines.model', 5006.843, 50],
        ['OIII4363',    'zlines.model', 4363.85, 50],
        ['LiI',         'zlines.model', 6707.76, 50],
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("xcsao_teff", "XCSAO"),
        ("data_product", "Metadata"),
    ]
    from astra.contrib.lineforest.base import airtovac

    for line, _, wavelength_air, __ in LINES:
        wavelength_vac = np.array([airtovac(wavelength_air)]).flatten()[0]
        category_headers.append((f"{line}_eqw", f"{line} at {wavelength_vac:.2f} A [vacuum]"))
    

    all_fields = fields + line_fields
    q = (
        StarMeta
        .select(*all_fields)
        .join(
            LineForestOutput,
            on=(
                (LineForestOutput.source_id == StarMeta.cat_id)
            &   (LineForestOutput.mjd.is_null(True))
            )
        )
        .group_by(*fields)
        .dicts()
    )

    meta_fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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
        

        # BOSS-HDU-SPECIFIC

        StarMeta.xcsao_teff,
        StarMeta.xcsao_e_teff,
        StarMeta.xcsao_logg,
        StarMeta.xcsao_e_logg,
        StarMeta.xcsao_fe_h,
        StarMeta.xcsao_e_fe_h,
        StarMeta.xcsao_rxc,
        StarMeta.xcsao_v_rad,
        StarMeta.xcsao_e_v_rad,        
    ]    
    q2 = (
        StarMeta
        .select(*meta_fields)
        .join(LineForestOutput, on=(StarMeta.cat_id == LineForestOutput.source_id))
        .dicts()
    )

    moo = { item["cat_id"]: item for item in tqdm(q2) }

    data = []
    for item in tqdm(q):
        item.update(moo[item.pop("source")])
        data.append(item)
    data = list_to_dict(data)

    for field in line_fields:
        name = field.name
        values = np.array(data[name], dtype=np.float32)
        values[values == fill_value] = np.nan
        data[name] = list(values)

    fields = []
    fields.extend(meta_fields)
    fields.extend([
        LineForestOutput.data_product,
        LineForestOutput.snr,        
    ])
    fields.extend(line_fields)

    #H HERE

    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "LineForest", "Pipeline that created these results"),
        ])),
    ]

    data = list_to_dict(q)    
    t.append(create_summary_table(data, fields, category_headers, "APO", "BOSS"))
    t.extend([
        create_empty_hdu("LCO", "BOSS", is_data=False),
        create_empty_hdu("APO", "APOGEE", is_data=False),
        create_empty_hdu("LCO", "APOGEE", is_data=False),
    ])
    add_check_sums(t)
    t = fits.HDUList(t)
    return t



def create_allVisit_lineforest_table():

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

    fields = [   
        LineForestOutput.source_id,    
        LineForestOutput.data_product,
        LineForestOutput.mjd,
        LineForestOutput.field,
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

    LINES = [
        ['Halpha',      'hlines.model', 6562.8, 200],
        ['Hbeta',       'hlines.model', 4861.3, 200],
        ['Hgamma',      'hlines.model', 4340.5, 200],
        ['Hdelta',      'hlines.model', 4101.7, 200],
        ['Hepsilon',    'hlines.model', 3970.1, 200],
        ['H8',          'hlines.model', 3889.064, 200],
        ['H9',          'hlines.model', 3835.391, 200],
        ['H10',         'hlines.model', 3797.904, 200],
        ['H11',         'hlines.model', 3770.637, 200],
        ['H12',         'zlines.model', 3750.158, 50],
        ['H13',         'zlines.model', 3734.369, 50],
        ['H14',         'zlines.model', 3721.945, 50],
        ['H15',         'zlines.model', 3711.977, 50],
        ['H16',         'zlines.model', 3703.859, 50],
        ['H17',         'zlines.model', 3697.157, 50],
        ['Pa7',         'hlines.model', 10049.4889, 200],
        ['Pa8',         'hlines.model', 9546.0808, 200],
        ['Pa9',         'hlines.model', 9229.12, 200],
        ['Pa10',        'hlines.model', 9014.909, 200],
        ['Pa11',        'hlines.model', 8862.782, 200],
        ['Pa12',        'hlines.model', 8750.472, 200],
        ['Pa13',        'hlines.model', 8665.019, 200],
        ['Pa14',        'hlines.model', 8598.392, 200],
        ['Pa15',        'hlines.model', 8545.383, 200],
        ['Pa16',        'hlines.model', 8502.483, 200],
        ['Pa17',        'hlines.model', 8467.254, 200],
        ['CaII8662',    'zlines.model', 8662.14, 50],
        ['CaII8542',    'zlines.model', 8542.089, 50],
        ['CaII8498',    'zlines.model', 8498.018, 50],
        ['CaK3933',     'hlines.model', 3933.6614, 200],
        ['CaH3968',     'hlines.model', 3968.4673, 200],
        ['HeI6678',     'zlines.model', 6678.151, 50],
        ['HeI5875',     'zlines.model', 5875.621, 50],
        ['HeI5015',     'zlines.model', 5015.678, 50],
        ['HeI4471',     'zlines.model', 4471.479, 50],
        ['HeII4685',    'zlines.model', 4685.7, 50],
        ['NII6583',     'zlines.model', 6583.45, 50],
        ['NII6548',     'zlines.model', 6548.05, 50],
        ['SII6716',     'zlines.model', 6716.44, 50],
        ['SII6730',     'zlines.model', 6730.816, 50],
        ['FeII5018',    'zlines.model', 5018.434, 50],
        ['FeII5169',    'zlines.model', 5169.03, 50],
        ['FeII5197',    'zlines.model', 5197.577, 50],
        ['FeII6432',    'zlines.model', 6432.68, 50],
        ['OI5577',      'zlines.model', 5577.339, 50],
        ['OI6300',      'zlines.model', 6300.304, 50],
        ['OI6363',      'zlines.model', 6363.777, 50],
        ['OII3727',     'zlines.model', 3727.42, 50],
        ['OIII4959',    'zlines.model', 4958.911, 50],
        ['OIII5006',    'zlines.model', 5006.843, 50],
        ['OIII4363',    'zlines.model', 4363.85, 50],
        ['LiI',         'zlines.model', 6707.76, 50],
    ]

    category_headers = [
        ("cat_id", "Identifiers"),
        ("ra", "Astrometry"),
        ("g_mag", "Photometry"),
        ("carton_0", "Targeting"),
        ("xcsao_teff", "XCSAO"),
        ("data_product", "Metadata"),
    ]
    from astra.contrib.lineforest.base import airtovac

    for line, _, wavelength_air, __ in LINES:
        wavelength_vac = np.array([airtovac(wavelength_air)]).flatten()[0]
        category_headers.append((f"{line}_eqw", f"{line} at {wavelength_vac:.2f} A [vacuum]"))
    

    all_fields = fields + line_fields
    q = (
        LineForestOutput
        .select(*all_fields)
        .join(
            VisitMeta,
            on=(
                (LineForestOutput.source_id == VisitMeta.cat_id)
            &   (LineForestOutput.mjd == VisitMeta.mjd)
            &   (LineForestOutput.field == VisitMeta.field)
            &   (LineForestOutput.instrument == "BOSS")
            )
        )
        .where(
            (LineForestOutput.instrument == "BOSS")
        &   (LineForestOutput.mjd.is_null(False))
        )
        .group_by(*fields)
        .dicts()
    )

    meta_fields = [
        StarMeta.cat_id,
        StarMeta.cat_id05,
        StarMeta.cat_id10,
        StarMeta.tic_id.alias("tic_v8_id"),
        StarMeta.gaia_source_id.alias("gaia_dr3_source_id"),
        StarMeta.sdss4_dr17_apogee_id,
        StarMeta.sdss4_dr17_field,
        StarMeta.sdss4_apogee_target1,
        StarMeta.sdss4_apogee_target2,
        StarMeta.sdss4_apogee2_target1,
        StarMeta.sdss4_apogee2_target2,
        StarMeta.sdss4_apogee2_target3,    
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


        # BOSS-HDU-SPECIFIC
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
        
        StarMeta.xcsao_teff,
        StarMeta.xcsao_e_teff,
        StarMeta.xcsao_logg,
        StarMeta.xcsao_e_logg,
        StarMeta.xcsao_fe_h,
        StarMeta.xcsao_e_fe_h,
        StarMeta.xcsao_rxc,
        StarMeta.xcsao_v_rad,
        StarMeta.xcsao_e_v_rad,    

        VisitMeta.in_stack,
        VisitMeta.v_shift,
        VisitMeta.fps,
        VisitMeta.snr


    ]    
    q2 = (
        VisitMeta
        .select(*meta_fields)
        .join(
            StarMeta,
            on=(
                (StarMeta.cat_id == VisitMeta.cat_id)
            )
        )
        .switch(VisitMeta)
        .join(
            LineForestOutput, 
            on=(
                (LineForestOutput.source_id == VisitMeta.cat_id)
            &   (LineForestOutput.mjd == VisitMeta.mjd)
            &   (LineForestOutput.field == VisitMeta.field)
            &   (LineForestOutput.instrument == "BOSS")                
            )
        )
        .dicts()
    )

    moo = { item["cat_id"]: item for item in tqdm(q2) }

    data = []
    for item in tqdm(q):
        item.update(moo[item.pop("source")])
        data.append(item)
    data = list_to_dict(data)

    for field in line_fields:
        name = field.name
        values = np.array(data[name], dtype=np.float32)
        values[values == fill_value] = np.nan
        data[name] = list(values)

    fields = []
    fields.extend(meta_fields)
    fields.extend([
        LineForestOutput.data_product,
        #LineForestOutput.snr,        
    ])
    fields.extend(line_fields)

    #H HERE
    t = [
        fits.PrimaryHDU(header=fits.Header([
            ("PIPELINE", "LineForest", "Pipeline that created these results"),
        ])),
    ]

    data = list_to_dict(q)    
    t.append(create_summary_table(data, fields, category_headers, "APO", "BOSS"))
    t.extend([
        create_empty_hdu("LCO", "BOSS", is_data=False),
        create_empty_hdu("APO", "APOGEE", is_data=False),
        create_empty_hdu("LCO", "APOGEE", is_data=False),
    ])
    add_check_sums(t)
    t = fits.HDUList(t)
    return t


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


    t = create_allStar_snowwhite_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-SnowWhite-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-SnowWhite")

    raise a

    # [x] SLAM
    # [x] lineforest star
    # [x] lineforest visit
    # [x] apogeenet visits
    # [x] cannon apogee star
    # [ ] cannon boss
    # [ ] aspcap
    # [x] classifier allStar
    # [x] classifier allVisit

    t = create_allStar_aspcap_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-ASPCAP-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    raise a

    t = create_allStar_cannon_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-TheCannon-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )    

    raise a


    t = create_allVisit_classifier_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allVisit-Classifier-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )    


    t = create_allStar_classifier_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-Classifier-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )    

    #t = create_allVisit_apogeenet_table()
    #t.writeto(
    #    expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allVisit-APOGEENet-{v_astra}-{run2d}-{apred}.fits"),
    #    overwrite=True
    #)
    #print("Wrote allStar-APOGEENet")



    t = create_allStar_slam_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-SLAM-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )    

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


    t = create_allVisit_apogeenet_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allVisit-APOGEENet-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allVisit-APOGEENet")


    t = create_allStar_lineforest_table()
    t.writeto(
        expand_path(f"$MWM_ASTRA/{v_astra}/{run2d}-{apred}/summary/allStar-LineForest-{v_astra}-{run2d}-{apred}.fits"),
        overwrite=True
    )
    print("Wrote allStar-LineForest")



    #t = create_allVisit_lineforest_table()
    #raise a

    raise a

    t = create_allStar_lineforest_table()

    #t = create_allStar_snowwhite_table()

    raise a 



    t = create_allStar_thepayne_table()
    raise a

    t = create_allStar_mdwarftype_table()

    raise a

    t = create_allStar_apogeenet_table()

    raise a

    t = create_allStar_cannon_table()

    raise a
