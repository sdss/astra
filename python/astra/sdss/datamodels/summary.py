from astropy.table import Table
from astra.database.astradb import (
    Source, ApogeeNetOutput, ZetaPayneOutput, Task, DataProduct, TaskInputDataProducts, SourceDataProduct,
    WhiteDwarfLineRatiosOutput, WhiteDwarfClassifierOutput, WhiteDwarfOutput, ThePayneOutput, SlamOutput,
    AspcapOutput, ClassifySourceOutput, ClassifierOutput, MDwarfTypeOutput
)

from sdss_access import SDSSPath
from peewee import Alias
from tqdm import tqdm
from astropy.io import fits
from astropy.table.operations import join


from astra.database.catalogdb import (
    Catalog,
    CatalogToTIC_v8,
    TIC_v8 as TIC,
    TwoMassPSC,
    Gaia_DR2 as Gaia
)
from peewee import JOIN

'''
expression = (Task.version == "0.2.5")
fields = (
    SourceDataProduct.source_id.alias("cat_id"),
    Task.id.alias("task_id"),
    DataProduct.release,
    DataProduct.filetype,
    DataProduct.kwargs,
    ZetaPayneOutput.snr,
    ZetaPayneOutput.teff,
    ZetaPayneOutput.e_teff,
    ZetaPayneOutput.logg,
    ZetaPayneOutput.e_logg,
    ZetaPayneOutput.fe_h,
    ZetaPayneOutput.e_fe_h,
    ZetaPayneOutput.vsini,
    ZetaPayneOutput.e_vsini,
    ZetaPayneOutput.v_micro,
    ZetaPayneOutput.e_v_micro,
    ZetaPayneOutput.v_rel,
    ZetaPayneOutput.e_v_rel,
    #ZetaPayneOutput.theta,
    ZetaPayneOutput.chi_sq,
    ZetaPayneOutput.reduced_chi_sq,
)
q = (
    ZetaPayneOutput
    .select(*fields)
    .join(Task)
    .join(TaskInputDataProducts)
    .join(DataProduct)
    .join(SourceDataProduct)
    .where(expression)
    .dicts()
)


rows = []
for output in q:
    kwargs = output.pop("kwargs")
    row = {**output, **kwargs}
    rows.append(row)

names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
# put all other names where kwargs is
kwarg_names = list(
    set.union(*(set(ea.keys()) for ea in rows)).difference(names)
)

for kwarg_name in kwarg_names[::-1]:
    names.insert(names.index("kwargs"), kwarg_name)
names.pop(names.index("kwargs"))

t = Table(rows=rows, names=names)
'''

from astra.database.targetdb import Target, CartonToTarget, Carton

def auxiliary_table():
    """
    Return a table of photometry, astrometry, and identifiers for every source in the Astra database.
    """
    fields = [
        Catalog.catalogid.alias("cat_id"),
        TIC.id.alias("tic_id"),
        Gaia.source_id.alias("gaia_id"),
        Catalog.ra.alias("cat_ra"),
        Catalog.dec.alias("cat_dec"),
        Gaia.ra,
        Gaia.dec,
        Gaia.parallax,
        Gaia.parallax_error,
        Gaia.pmra,
        Gaia.pmra_error,
        Gaia.pmdec,
        Gaia.pmdec_error,
        Gaia.radial_velocity,
        Gaia.radial_velocity_error,
        Gaia.phot_g_mean_mag,
        Gaia.phot_bp_mean_mag,
        Gaia.phot_rp_mean_mag,
        TwoMassPSC.j_m,
        TwoMassPSC.j_cmsig,
        TwoMassPSC.h_m,
        TwoMassPSC.h_cmsig,
        TwoMassPSC.k_m,
        TwoMassPSC.k_cmsig,
    ]

    q = (
        Catalog.select(*fields)
        .distinct(Catalog.catalogid)
        .join(Source, on=(Source.catalogid == Catalog.catalogid))
        .switch(Catalog)
        .join(CatalogToTIC_v8, JOIN.LEFT_OUTER)
        .join(TIC)
        .join(Gaia, JOIN.LEFT_OUTER)
        .switch(TIC)
        .join(TwoMassPSC, JOIN.LEFT_OUTER)
        .dicts()
    )

    # Carton information?
    
    # Doppler results as well?
    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    rows = list(q)
    return Table(rows=rows, names=names)


def summary_table_v025(fields):

    model = fields[0].model
    if len(set([ea.model for ea in fields])) > 1:
        raise ValueError(f"Only one model table allowed")

    expression = (Task.version == "0.2.5")
    fields = (
        SourceDataProduct.source_id.alias("cat_id"),
        Task.id.alias("task_id"),
        model.output_id,
        DataProduct.release,
        DataProduct.filetype,
        DataProduct.kwargs,
        *fields
    )
    q = (
        model
        .select(*fields)
        .join(Task)
        .join(TaskInputDataProducts)
        .join(DataProduct)
        .join(SourceDataProduct)
        .where(expression)
        .dicts()
    )

    rows = []
    for output in tqdm(q):
        kwargs = output.pop("kwargs")
        row = {**output, **kwargs}
        rows.append(row)

    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    # put all other names where kwargs is
    kwarg_names = list(
        set.union(*(set(ea.keys()) for ea in rows)).difference(names)
    )

    for kwarg_name in kwarg_names[::-1]:
        names.insert(names.index("kwargs"), kwarg_name)
    names.pop(names.index("kwargs"))

    return Table(rows=rows, names=names)

def _check_all_one_model(fields):
    model = fields[0].model
    if len(set([ea.model for ea in fields if not isinstance(ea, Alias)])) > 1:
        raise ValueError(f"Only one model table allowed")
    return model

def all_visit_table_v026(fields, expression=None, defaults=None):
    # Restrict by filetype
    _expression = (
        (Task.version == "0.2.6")
    &   (DataProduct.filetype.in_(("apVisit", "specFull")))
    )
    if expression is not None:
        _expression = _expression & expression

    model = _check_all_one_model(fields)
    fields = (
        model.source_id.alias("cat_id"),
        model.task_id.alias("task_id"),
        model.output_id.alias("output_id"),
        DataProduct.id.alias("data_product_id"),
        DataProduct.release,
        DataProduct.filetype,
        DataProduct.kwargs,
        *fields
    )
    q = (
        model
        .select(*fields)
        .join(DataProduct, on=(DataProduct.id == model.parent_data_product_id))
        .switch(model)
        .join(Task)
        .where(_expression)
        .dicts()
    )
    defaults = defaults or {}

    # TODO: Check that we still recover rows even if parent_data_product_id is None!
    rows = []
    for output in tqdm(q):
        kwargs = output.pop("kwargs")
        row = {}
        for k, v in {**output, **kwargs}.items():
            row[k] = v or defaults.get(k, None)
        
        rows.append(row)

    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    # put all other names where kwargs is
    kwarg_names = list(
        set.union(*(set(ea.keys()) for ea in rows)).difference(names)
    )

    for kwarg_name in kwarg_names[::-1]:
        names.insert(names.index("kwargs"), kwarg_name)
    names.pop(names.index("kwargs"))

    return Table(rows=rows, names=names)




def all_star_table_v026(fields, expression=None, defaults=None):
    # Restrict by filetype
    _expression = (
        (Task.version == "0.2.6")
    &   (DataProduct.filetype == "mwmStar")
    &   (DataProduct.kwargs["v_astra"] == "0.2.6")
    )
    if expression is not None:
        _expression = _expression & expression

    model = _check_all_one_model(fields)
    fields = (
        model.source_id.alias("cat_id"),
        model.task_id.alias("task_id"),
        model.output_id.alias("output_id"),
        DataProduct.id.alias("data_product_id"),
        DataProduct.release,
        DataProduct.filetype,
        DataProduct.kwargs,
        *fields
    )
    q = (
        model
        .select(*fields)
        .join(DataProduct, on=(DataProduct.id == model.parent_data_product_id))
        .switch(model)
        .join(Task)
        .where(_expression)
        .dicts()
    )
    defaults = defaults or {}

    # TODO: Check that we still recover rows even if parent_data_product_id is None!
    rows = []
    for output in tqdm(q):
        kwargs = output.pop("kwargs")
        row = {}
        for k, v in {**output, **kwargs}.items():
            row[k] = v or defaults.get(k, None)
        
        rows.append(row)

    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    # put all other names where kwargs is
    kwarg_names = list(
        set.union(*(set(ea.keys()) for ea in rows)).difference(names)
    )

    for kwarg_name in kwarg_names[::-1]:
        names.insert(names.index("kwargs"), kwarg_name)
    names.pop(names.index("kwargs"))

    return Table(rows=rows, names=names)

def all_visit_apogeenet_v026():
    fields = (
        ApogeeNetOutput.snr,
        ApogeeNetOutput.teff,
        ApogeeNetOutput.e_teff,
        ApogeeNetOutput.logg,
        ApogeeNetOutput.e_logg,
        ApogeeNetOutput.fe_h,
        ApogeeNetOutput.e_fe_h,
        ApogeeNetOutput.bitmask_flag
    )
    t = all_visit_table_v026(fields)
    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", #"V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    other_keys = ("DITHERED", "JD", "V_RAD", "E_V_RAD", "V_REL", "V_BC", "RCHISQ", "V_SHIFT", "STARFLAG", "IN_STACK", "VISIT_PK", "RV_VISIT_PK")

    aux_info = {key.lower(): [] for key in keys}
    for key in other_keys:
        aux_info[key.lower()] = []
    aux_info["output_id"] = []
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for j, row in enumerate(tqdm(t)):
        aux_info["output_id"].append(row["output_id"])
        #path = p.full(**dict(zip(t.dtype.names, row)))
        path = SDSSPath("sdss5").full("mwmVisit", cat_id=row["cat_id"], apred="1.0", run2d="v6_0_9", v_astra="0.2.6")
        #image = fits.open(path)
        with fits.open(path) as image:
            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))
            match = np.where(image[3].data["DATA_PRODUCT_ID"] == row["data_product_id"])[0][0]
            for key in other_keys:
                aux_info[key.lower()].append(image[3].data[key][match])

    all_keys = ["output_id"] + list(map(str.lower, keys)) + list(map(str.lower, other_keys))
    t_aux = Table(data=aux_info, names=all_keys)
    t_merge = join(t_aux, t, keys="output_id")  
    # make sure things are the right type
    for k in ("teff", "logg", "fe_h", "e_teff", "e_logg", "e_fe_h", "snr"):
        t_merge[k] = t_merge[k].astype(float)
    t_merge["bitmask_flag"][t_merge["bitmask_flag"] == None] = 0
    t_merge["bitmask_flag"] = t_merge["bitmask_flag"].astype(int)

    return t_merge


    # get data from APOGEE DRP file

import datetime
from astropy.io import fits
from astropy.table import Table, unique
from astra.sdss.datamodels.base import BLANK_CARD, FILLER_CARD, fits_column_kwargs, add_check_sums, add_table_category_headers, add_glossary_comments
import numpy as np
from astropy.table.operations import join

def _construct_astraAllStar_apogeenet(temporary_apogeenet_path):

    t_apogeenet = Table.read(temporary_apogeenet_path)
    t_apogeenet = unique(t_apogeenet, keys=("output_id", ))

    t_drp = Table.read('/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/1.0/summary/allStar-1.0-apo25m.fits')

    t_temp = join(
        t_apogeenet, t_drp, join_type="left", keys_left=("cat_id", ), keys_right=("catalogid", ),
        table_names=["", "_drp"], uniq_col_name="{col_name}{table_name}")

    category_headers = [
        ("GAIA_ID", "IDENTIFIERS"),
        ("RA", "SKY POSITION"),
        ("RA_GAIA", "ASTROMETRY (GAIA DR2)"),
        ("G_MAG", "PHOTOMETRY"),
        ("CARTON_0", "TARGETING"),
        ("V_RAD", "RADIAL VELOCITIES (DOPPLER)"),
        ("TEFF_D", "STELLAR PARAMETERS (DOPPLER)"),
        ("RELEASE", "INPUT DATA MODEL KEYWORDS"),
        ("TEFF", "STELLAR PARAMETERS (APOGEENet)"),
        ("SNR", "SUMMARY STATISTICS"),
        ("TASK_ID", "DATABASE IDENTIFIERS"),
    ]
    mappings = [
        ("GAIA_ID", t_temp["gaia_id"]),
        ("TIC_ID", t_temp["tic_id"]),
        ("APOGEE_ID", t_temp["apogee_id"]),

        ("RA", t_temp["ra"]),
        ("DEC", t_temp["dec"]),
        #("GLON", t_temp["glon"]),
        #("GLAT", t_temp["glat"]),
        
        ("HEALPIX", t_temp["healpix"]),
        ("RA_GAIA", t_temp["gaia_ra"]),
        ("DEC_GAIA", t_temp["gaia_dec"]),
        ("PLX", t_temp["plx"]),
        ("E_PLX", t_temp["e_plx"]),
        ("PMRA", t_temp["pmra"]),
        ("E_PMRA", t_temp["e_pmra"]),
        ("PMDE", t_temp["pmde"]),
        ("E_PMDE", t_temp["e_pmde"]),
        ("V_RAD_GAIA", t_temp["v_rad"]),
        ("E_V_RAD_GAIA", t_temp["e_v_rad"]),
        ("G_MAG", t_temp["g_mag"]),
        ("BP_MAG", t_temp["bp_mag"]),
        ("RP_MAG", t_temp["rp_mag"]),
        ("J_MAG", t_temp["j_mag"]),
        ("H_MAG", t_temp["h_mag"]),
        ("K_MAG", t_temp["k_mag"]),
        ("CARTON_0", t_temp["carton_0"]),
        ("PROGRAMS", t_temp["programs"]),
        ("MAPPERS", t_temp["mappers"]),

        ("V_RAD", t_temp["vrad"]),
        ("E_V_RAD", t_temp["verr"]),
        ("V_SCATTER", t_temp["vscatter"]),
        ("E_V_MED", t_temp["vmederr"]),
        ("CHISQ_RV", t_temp["chisq"]),
        ("CCPFWHM_D", t_temp["rv_ccpfwhm"]),
        ("AUTOFWHM_D", t_temp["rv_autofwhm"]),
        ("N_RV_COMPONENTS", t_temp["n_components"]),
        ("NVISITS_APSTAR", t_temp["nvisits"]),
        ("NGOODVISITS", t_temp["ngoodvisits"]),
        ("NGOODRVS", t_temp["ngoodrvs"]),
        ("STARFLAG", t_temp["starflag"]),
        ("STARFLAGS", t_temp["starflags"]),
        ("MEANFIB", t_temp["meanfib"]),
        ("SIGFIB", t_temp["sigfib"]),

        ("TEFF_D", t_temp["rv_teff"]),
        ("E_TEFF_D", t_temp["rv_tefferr"]),
        ("LOGG_D", t_temp["rv_logg"]),
        ("E_LOGG_D", t_temp["rv_loggerr"]),
        ("FEH_D", t_temp["rv_feh"]),
        ("E_FEH_D", t_temp["rv_feherr"]),

        ("RELEASE", t_temp["release"]),
        ("FILETYPE", t_temp["filetype"]),
        ("V_ASTRA", t_temp["v_astra"]),
        ("RUN2D", t_temp["run2d"]),
        ("APRED", t_temp["apred"]),
        ("CAT_ID", t_temp["cat_id"]),


        ("TEFF", t_temp["teff"]),
        ("E_TEFF", t_temp["e_teff"]),
        ("LOGG", t_temp["logg"]),
        ("E_LOGG", t_temp["e_logg"]),
        ("FE_H", t_temp["fe_h"]),
        ("E_FE_H", t_temp["e_fe_h"]),
        ("BITMASK_FLAG", t_temp["bitmask_flag"]),

        ("SNR", t_temp["snr"]),

        ("TASK_ID", t_temp["task_id"]),
        ("OUTPUT_ID", t_temp["output_id"]),
        ("DATA_PRODUCT_ID", t_temp["data_product_id"]),        
        ("STAR_PK", t_temp["pk"]),
    ]
    columns = []
    
    for key, values in mappings:
        columns.append(
            fits.Column(
                name=key,
                array=values,
                unit=None,
                **fits_column_kwargs(np.array(values)),
            )
        )

    header = fits.Header(
        [
            BLANK_CARD,
            (" ", "METADATA", None),
            ("V_ASTRA", "0.2.6"),
            ("RUN2D", "v6_0_9"),
            ("APRED", "1.0"),
            ("PIPELINE", "APOGEENet"),
            ("CREATED", datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")),
            FILLER_CARD,
        ]
    )
        
    hdu = fits.BinTableHDU.from_columns(
        columns,
        #header=header,
        # name=f"{header['INSTRMNT']}/{header['OBSRVTRY']}"
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    add_glossary_comments(primary_hdu)

    add_table_category_headers(hdu, category_headers)
    add_glossary_comments(hdu)
    hdu_list = fits.HDUList([primary_hdu, hdu])
    add_check_sums(hdu_list)

    return hdu_list
    

def construct_boss_allStar(temp_results_path, pipeline, mappings, category_headers=None):
    t_results = Table.read(temp_results_path)
    if "output_id" in t_results.dtype.names:
        t_results = unique(t_results, keys=("output_id", ))

    t_drp = Table.read("/uufs/chpc.utah.edu/common/home/sdss50/ipl-1/spectro/boss/redux/v6_0_9/spAll-v6_0_9.fits")
    t_temp = join(
        t_results, t_drp, join_type="left", keys_left=("cat_id", ), keys_right=("CATALOGID", ),
        table_names=["", "_drp"], uniq_col_name="{col_name}{table_name}")

    all_category_headers = [
        ("GAIA_ID", "IDENTIFIERS"),
        ("RA", "SKY POSITION"),
        ("RA_GAIA", "ASTROMETRY (GAIA DR2)"),
        ("G_MAG", "PHOTOMETRY"),
        ("CARTON_0", "TARGETING"),
        ("CLASS_NOQSO", "BOSS DATA REDUCTION PIPELINE"),
        ("V_RAD", "RADIAL VELOCITIES (XCSAO)"),
        #("TEFF_XCSAO", "STELLAR PARAMETERS (XCSAO)"),
        ("RELEASE", "INPUT DATA MODEL KEYWORDS"),
    ]
    if "TEFF" in t_temp.dtype.names:
        all_category_headers.append(("TEFF", f"STELLAR PARAMETERS ({pipeline})"))
    if "SNR" in t_temp.dtype.names:
        all_category_headers.append(("SNR", "SUMMARY STATISTICS"))

    if category_headers is not None:
        all_category_headers.extend(category_headers)

    # Map with spAll file.
    all_mappings = [
        ("GAIA_ID", t_temp["gaia_id"]),
        ("TIC_ID", t_temp["tic_id"]),
        ("BOSS_SPECOBJ_ID", t_temp["BOSS_SPECOBJ_ID"]),
        #("APOGEE_ID", t_temp["apogee_id"]),

        ("RA", t_temp["ra"]),
        ("DEC", t_temp["dec"]),
        #("GLON", t_temp["glon"]),
        #("GLAT", t_temp["glat"]),
        
        ("HEALPIX", t_temp["healpix"]),
        ("RA_GAIA", t_temp["gaia_ra"]),
        ("DEC_GAIA", t_temp["gaia_dec"]),
        ("PLX", t_temp["plx"]),
        ("E_PLX", t_temp["e_plx"]),
        ("PMRA", t_temp["pmra"]),
        ("E_PMRA", t_temp["e_pmra"]),
        ("PMDE", t_temp["pmde"]),
        ("E_PMDE", t_temp["e_pmde"]),
        ("V_RAD_GAIA", t_temp["v_rad"]),
        ("E_V_RAD_GAIA", t_temp["e_v_rad"]),
        ("G_MAG", t_temp["g_mag"]),
        ("BP_MAG", t_temp["bp_mag"]),
        ("RP_MAG", t_temp["rp_mag"]),
        ("J_MAG", t_temp["j_mag"]),
        ("H_MAG", t_temp["h_mag"]),
        ("K_MAG", t_temp["k_mag"]),
        ("CARTON_0", t_temp["carton_0"]),
        ("PROGRAMS", t_temp["programs"]),
        ("MAPPERS", t_temp["mappers"]),

        ("CLASS_NOQSO", t_temp["CLASS_NOQSO"]),
        ("SUBCLASS_NOQSO", t_temp["SUBCLASS_NOQSO"]),
        ("ZWARNING", t_temp["ZWARNING"]),
        ("ZWARNING_NOQSO", t_temp["ZWARNING_NOQSO"]),

        ("V_RAD", t_temp["XCSAO_RV"]),
        ("E_V_RAD", t_temp["XCSAO_ERV"]),
        ("RXC_XCSAO", t_temp["XCSAO_RXC"]),
        ("TEFF_XCSAO", t_temp["XCSAO_TEFF"]),
        ("E_TEFF_XCSAO", t_temp["XCSAO_ETEFF"]),
        ("LOGG_XCSAO", t_temp["XCSAO_LOGG"]),
        ("E_LOGG_XCSAO", t_temp["XCSAO_ELOGG"]),
        ("FEH_XCSAO", t_temp["XCSAO_FEH"]),
        ("E_FEH_XCSAO", t_temp["XCSAO_EFEH"]),

        ("RELEASE", t_temp["release"]),
        ("FILETYPE", t_temp["filetype"]),
        ("V_ASTRA", t_temp["v_astra"]),
        ("RUN2D", t_temp["run2d"]),
        ("APRED", t_temp["apred"]),
        ("CAT_ID", t_temp["cat_id"]),
    ]
    all_mappings.extend([(k, t_temp[v]) for k, v in mappings])
    
    if "task_id" in t_temp.dtype.names:
        all_mappings.append(("TASK_ID", t_temp["task_id"]))
        all_category_headers.append(("TASK_ID", "DATABASE IDENTIFIER"))

    if "output_id" in t_temp.dtype.names:
        all_mappings.append(("OUTPUT_ID", t_temp["output_id"]))
    
    all_mappings.append(("DATA_PRODUCT_ID", t_temp["data_product_id"]))
    columns = []
    
    for key, values in all_mappings:
        columns.append(
            fits.Column(
                name=key,
                array=values,
                unit=None,
                **fits_column_kwargs(np.array(values)),
            )
        )

    header = fits.Header(
        [
            BLANK_CARD,
            (" ", "METADATA", None),
            ("V_ASTRA", "0.2.6"),
            ("RUN2D", "v6_0_9"),
            ("APRED", "1.0"),
            ("PIPELINE", pipeline),
            ("CREATED", datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")),
            FILLER_CARD,
        ]
    )
        
    hdu = fits.BinTableHDU.from_columns(
        columns,
        #header=header,
        # name=f"{header['INSTRMNT']}/{header['OBSRVTRY']}"
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    add_glossary_comments(primary_hdu)

    add_table_category_headers(hdu, all_category_headers)
    add_glossary_comments(hdu)
    hdu_list = fits.HDUList([primary_hdu, hdu])
    add_check_sums(hdu_list)

    return hdu_list


def construct_apogee_allStar(temp_results_path, pipeline, mappings, category_headers=None):

    t_results = Table.read(temp_results_path)
    t_results = unique(t_results, keys=("output_id", ))

    t_drp = Table.read('/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/1.0/summary/allStar-1.0-apo25m.fits')

    t_temp = join(
        t_results, t_drp, join_type="left", keys_left=("cat_id", ), keys_right=("catalogid", ),
        table_names=["", "_drp"], uniq_col_name="{col_name}{table_name}")

    all_category_headers = [
        ("GAIA_ID", "IDENTIFIERS"),
        ("RA", "SKY POSITION"),
        ("RA_GAIA", "ASTROMETRY (GAIA DR2)"),
        ("G_MAG", "PHOTOMETRY"),
        ("CARTON_0", "TARGETING"),
        ("V_RAD", "RADIAL VELOCITIES (DOPPLER)"),
        ("TEFF_D", "STELLAR PARAMETERS (DOPPLER)"),
        ("RELEASE", "INPUT DATA MODEL KEYWORDS"),
        ("TEFF", f"STELLAR PARAMETERS ({pipeline})"),
        ("SNR", "SUMMARY STATISTICS"),
        ("TASK_ID", "DATABASE IDENTIFIERS"),
    ]
    if category_headers is not None:
        all_category_headers.extend(category_headers)

    all_mappings = [
        ("GAIA_ID", t_temp["gaia_id"]),
        ("TIC_ID", t_temp["tic_id"]),
        ("APOGEE_ID", t_temp["apogee_id"]),

        ("RA", t_temp["ra"]),
        ("DEC", t_temp["dec"]),
        #("GLON", t_temp["glon"]),
        #("GLAT", t_temp["glat"]),
        
        ("HEALPIX", t_temp["healpix"]),
        ("RA_GAIA", t_temp["gaia_ra"]),
        ("DEC_GAIA", t_temp["gaia_dec"]),
        ("PLX", t_temp["plx"]),
        ("E_PLX", t_temp["e_plx"]),
        ("PMRA", t_temp["pmra"]),
        ("E_PMRA", t_temp["e_pmra"]),
        ("PMDE", t_temp["pmde"]),
        ("E_PMDE", t_temp["e_pmde"]),
        ("V_RAD_GAIA", t_temp["v_rad"]),
        ("E_V_RAD_GAIA", t_temp["e_v_rad"]),
        ("G_MAG", t_temp["g_mag"]),
        ("BP_MAG", t_temp["bp_mag"]),
        ("RP_MAG", t_temp["rp_mag"]),
        ("J_MAG", t_temp["j_mag"]),
        ("H_MAG", t_temp["h_mag"]),
        ("K_MAG", t_temp["k_mag"]),
        ("CARTON_0", t_temp["carton_0"]),
        ("PROGRAMS", t_temp["programs"]),
        ("MAPPERS", t_temp["mappers"]),

        ("V_RAD", t_temp["vrad"]),
        ("E_V_RAD", t_temp["verr"]),
        ("V_SCATTER", t_temp["vscatter"]),
        ("E_V_MED", t_temp["vmederr"]),
        ("CHISQ_RV", t_temp["chisq"]),
        ("CCPFWHM_D", t_temp["rv_ccpfwhm"]),
        ("AUTOFWHM_D", t_temp["rv_autofwhm"]),
        ("N_RV_COMPONENTS", t_temp["n_components"]),
        ("NVISITS_APSTAR", t_temp["nvisits"]),
        ("NGOODVISITS", t_temp["ngoodvisits"]),
        ("NGOODRVS", t_temp["ngoodrvs"]),
        ("STARFLAG", t_temp["starflag"]),
        ("STARFLAGS", t_temp["starflags"]),
        ("MEANFIB", t_temp["meanfib"]),
        ("SIGFIB", t_temp["sigfib"]),

        ("TEFF_D", t_temp["rv_teff"]),
        ("E_TEFF_D", t_temp["rv_tefferr"]),
        ("LOGG_D", t_temp["rv_logg"]),
        ("E_LOGG_D", t_temp["rv_loggerr"]),
        ("FEH_D", t_temp["rv_feh"]),
        ("E_FEH_D", t_temp["rv_feherr"]),

        ("RELEASE", t_temp["release"]),
        ("FILETYPE", t_temp["filetype"]),
        ("V_ASTRA", t_temp["v_astra"]),
        ("RUN2D", t_temp["run2d"]),
        ("APRED", t_temp["apred"]),
        ("CAT_ID", t_temp["cat_id"]),
    ]
    all_mappings.extend([(k, t_temp[v]) for k, v in mappings])

    all_mappings.extend([
        ("TASK_ID", t_temp["task_id"]),
        ("OUTPUT_ID", t_temp["output_id"]),
        ("DATA_PRODUCT_ID", t_temp["data_product_id"]),        
        ("STAR_PK", t_temp["pk"]),
    ])
    columns = []
    
    for key, values in all_mappings:
        columns.append(
            fits.Column(
                name=key,
                array=values,
                unit=None,
                **fits_column_kwargs(np.array(values)),
            )
        )

    header = fits.Header(
        [
            BLANK_CARD,
            (" ", "METADATA", None),
            ("V_ASTRA", "0.2.6"),
            ("RUN2D", "v6_0_9"),
            ("APRED", "1.0"),
            ("PIPELINE", pipeline),
            ("CREATED", datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")),
            FILLER_CARD,
        ]
    )
        
    hdu = fits.BinTableHDU.from_columns(
        columns,
        #header=header,
        # name=f"{header['INSTRMNT']}/{header['OBSRVTRY']}"
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    add_glossary_comments(primary_hdu)

    add_table_category_headers(hdu, all_category_headers)
    add_glossary_comments(hdu)
    hdu_list = fits.HDUList([primary_hdu, hdu])
    add_check_sums(hdu_list)

    return hdu_list


def all_star_apogeenet_v026():
    fields = (
        ApogeeNetOutput.snr,
        ApogeeNetOutput.teff,
        ApogeeNetOutput.e_teff,
        ApogeeNetOutput.logg,
        ApogeeNetOutput.e_logg,
        ApogeeNetOutput.fe_h,
        ApogeeNetOutput.e_fe_h,
        ApogeeNetOutput.bitmask_flag
    )
    t = all_star_table_v026(fields)
    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    return join(t_aux, t, keys="cat_id")  



def all_star_slam_v026():
    fields = (
        SlamOutput.snr,
        SlamOutput.teff,
        SlamOutput.e_teff,
        SlamOutput.logg,
        SlamOutput.e_logg,
        SlamOutput.fe_h,
        SlamOutput.e_fe_h,
        SlamOutput.success,
        SlamOutput.chi_sq,
        SlamOutput.initial_teff,
        SlamOutput.initial_logg,
        SlamOutput.initial_fe_h,
        SlamOutput.reduced_chi_sq,
    )
    t = all_star_table_v026(fields)
    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    return join(t_aux, t, keys="cat_id")       

def all_star_thepayne_v026():
    fields = (
        ThePayneOutput.snr,
        ThePayneOutput.teff,
        ThePayneOutput.logg,
        ThePayneOutput.v_turb,
        ThePayneOutput.c_h,
        ThePayneOutput.n_h,
        ThePayneOutput.o_h,
        ThePayneOutput.na_h,
        ThePayneOutput.mg_h,
        ThePayneOutput.al_h,
        ThePayneOutput.si_h,
        ThePayneOutput.p_h,
        ThePayneOutput.s_h,
        ThePayneOutput.k_h,
        ThePayneOutput.ca_h,
        ThePayneOutput.ti_h,
        ThePayneOutput.v_h,
        ThePayneOutput.cr_h,
        ThePayneOutput.mn_h,
        ThePayneOutput.fe_h,
        ThePayneOutput.co_h,
        ThePayneOutput.ni_h,
        ThePayneOutput.cu_h,
        ThePayneOutput.ge_h,
        ThePayneOutput.c12_c13,
        ThePayneOutput.v_macro,
        ThePayneOutput.e_teff,
        ThePayneOutput.e_logg,
        ThePayneOutput.e_v_turb,
        ThePayneOutput.e_c_h,
        ThePayneOutput.e_n_h,
        ThePayneOutput.e_o_h,
        ThePayneOutput.e_na_h,
        ThePayneOutput.e_mg_h,
        ThePayneOutput.e_al_h,
        ThePayneOutput.e_si_h,
        ThePayneOutput.e_p_h,
        ThePayneOutput.e_s_h,
        ThePayneOutput.e_k_h,
        ThePayneOutput.e_ca_h,
        ThePayneOutput.e_ti_h,
        ThePayneOutput.e_v_h,
        ThePayneOutput.e_cr_h,
        ThePayneOutput.e_mn_h,
        ThePayneOutput.e_fe_h,
        ThePayneOutput.e_co_h,
        ThePayneOutput.e_ni_h,
        ThePayneOutput.e_cu_h,
        ThePayneOutput.e_ge_h,
        ThePayneOutput.e_c12_c13,
        ThePayneOutput.e_v_macro,     
        ThePayneOutput.chi_sq,
        ThePayneOutput.reduced_chi_sq,
        ThePayneOutput.bitmask_flag,
    )
    t = all_star_table_v026(fields)
    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            if "CAT_ID" not in image[0].header:
                for key in keys:
                    aux_info[key.lower()].append(default.get(key, np.nan))
                continue

            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    t_merge = join(t_aux, t, keys="cat_id")    
    for name in t_merge.dtype.names:
        if name.startswith("log_chisq_fit"):
            t_merge[name] = t_merge[name].astype(float)
        elif name.startswith("bitmask_"):
            t_merge[name][t_merge[name] == None] = 0
            t_merge[name] = t_merge[name].astype(int)    
    return t_merge

"""
t = all_star_apogeenet_v026()
t.write("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-APOGEENet-0.26-v6_0_9-1.0.fits")
"""

import numpy as np
def all_star_classifications_v026(defaults={}):
    fields = (
        ClassifySourceOutput.p_cv,
        ClassifySourceOutput.lp_cv,
        ClassifySourceOutput.p_fgkm,
        ClassifySourceOutput.lp_fgkm,
        ClassifySourceOutput.p_hotstar,
        ClassifySourceOutput.lp_hotstar,
        ClassifySourceOutput.p_wd,
        ClassifySourceOutput.lp_wd,
        ClassifySourceOutput.p_sb2,
        ClassifySourceOutput.lp_sb2,
        ClassifySourceOutput.p_yso,
        ClassifySourceOutput.lp_yso,
    )
    # Restrict by filetype
    _expression = (
        (Task.version == "0.2.6")
    )

    model = _check_all_one_model(fields)
    fields = (
        model.source_id.alias("cat_id"),
        model.task_id.alias("task_id"),
        model.output_id.alias("output_id"),
        *fields
    )
    q = (
        model
        .select(*fields)
        .join(Task)
        .where(_expression)
        .dicts()
    )
    defaults = defaults or {}

    # TODO: Check that we still recover rows even if parent_data_product_id is None!
    rows = []
    for output in tqdm(q):
        #kwargs = output.pop("kwargs")
        ##row = {}
        #for k, v in {**output, **kwargs}.items():
        #    row[k] = v or defaults.get(k, None)
        
        rows.append(output)

    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    # put all other names where kwargs is
    #kwarg_names = list(
    #    set.union(*(set(ea.keys()) for ea in rows)).difference(names)
    #)

    #for kwarg_name in kwarg_names[::-1]:
    #    names.insert(names.index("kwargs"), kwarg_name)
    #names.pop(names.index("kwargs"))

    t = Table(rows=rows, names=names)

    dtypes = dict(field=str, isplate=str)
    for name in t.dtype.names:
        if name in dtypes:
            t[name] = t[name].astype(dtypes[name])
        elif name.startswith("p_") or name.startswith("lp_"):
            t[name] = t[name].astype(float)
    
    # Add a most probable class.
    p_class_names = [name for name in t.dtype.names if name.startswith("p_")]
    probs = np.array([t[name] for name in p_class_names])

    idx = np.argmax(probs, axis=0)
    t["class"] = [p_class_names[i][2:] for i in idx]
    # exclude a test one
    return t


def all_visit_classifications_v026(defaults=None):
    fields = (
        ClassifierOutput.p_cv,
        ClassifierOutput.lp_cv,
        ClassifierOutput.p_fgkm,
        ClassifierOutput.lp_fgkm,
        ClassifierOutput.p_hotstar,
        ClassifierOutput.lp_hotstar,
        ClassifierOutput.p_wd,
        ClassifierOutput.lp_wd,
        ClassifierOutput.p_sb2,
        ClassifierOutput.lp_sb2,
        ClassifierOutput.p_yso,
        ClassifierOutput.lp_yso,
    )
    # Restrict by filetype
    _expression = (
        (Task.version == "0.2.6")
    )

    model = _check_all_one_model(fields)
    fields = (
        model.source_id.alias("cat_id"),
        model.task_id.alias("task_id"),
        model.output_id.alias("output_id"),
        DataProduct.id.alias("data_product_id"),
        DataProduct.release,
        DataProduct.filetype,
        DataProduct.kwargs,
        *fields
    )
    q = (
        model
        .select(*fields)
        .join(DataProduct, on=(DataProduct.id == model.parent_data_product_id))
        .switch(model)
        .join(Task)
        .where(_expression)
        .dicts()
    )
    defaults = defaults or {}

    # TODO: Check that we still recover rows even if parent_data_product_id is None!
    rows = []
    for output in tqdm(q):
        kwargs = output.pop("kwargs")
        row = {}
        for k, v in {**output, **kwargs}.items():
            row[k] = v or defaults.get(k, None)
        
        rows.append(row)

    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    # put all other names where kwargs is
    kwarg_names = list(
        set.union(*(set(ea.keys()) for ea in rows)).difference(names)
    )

    for kwarg_name in kwarg_names[::-1]:
        names.insert(names.index("kwargs"), kwarg_name)
    names.pop(names.index("kwargs"))

    t = Table(rows=rows, names=names)

    dtypes = dict(field=str, isplate=str)
    for name in t.dtype.names:
        if name in dtypes:
            t[name] = t[name].astype(dtypes[name])
        elif name.startswith("p_") or name.startswith("lp_"):
            t[name] = t[name].astype(float)
    
    # Add a most probable class.
    p_class_names = [name for name in t.dtype.names if name.startswith("p_")]
    probs = np.array([t[name] for name in p_class_names])

    idx = np.argmax(probs, axis=0)
    t["class"] = [p_class_names[i][2:] for i in idx]
    # exclude a test one
    t = t[t["filetype"] != "mwmStar"]
    return t


def all_star_zeta_payne_v026():
    fields = (
        ZetaPayneOutput.snr,
        ZetaPayneOutput.teff,
        ZetaPayneOutput.e_teff,
        ZetaPayneOutput.logg,
        ZetaPayneOutput.e_logg,
        ZetaPayneOutput.fe_h,
        ZetaPayneOutput.e_fe_h,
        ZetaPayneOutput.vsini,
        ZetaPayneOutput.e_vsini,
        ZetaPayneOutput.v_micro,
        ZetaPayneOutput.e_v_micro,
        ZetaPayneOutput.v_rel,
        ZetaPayneOutput.e_v_rel,
        ZetaPayneOutput.chi_sq,
        ZetaPayneOutput.reduced_chi_sq,
    )
    t = all_star_table_v026(fields)
    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    return join(t_aux, t, keys="cat_id")



import numpy as np
def all_star_aspcap_v026():
    fields = (
        AspcapOutput.snr,
        AspcapOutput.teff,
        AspcapOutput.e_teff,
        AspcapOutput.logg,
        AspcapOutput.e_logg,
        AspcapOutput.metals,
        AspcapOutput.e_metals,
        AspcapOutput.log10vdop,
        AspcapOutput.e_log10vdop,
        AspcapOutput.o_mg_si_s_ca_ti,
        AspcapOutput.e_o_mg_si_s_ca_ti,
        AspcapOutput.lgvsini,
        (AspcapOutput.c + AspcapOutput.metals).alias("c_h_photosphere"),
        AspcapOutput.e_c.alias("e_c_h_photosphere"),
        (AspcapOutput.n + AspcapOutput.metals).alias("n_h_photosphere"),
        AspcapOutput.e_n.alias("e_n_h_photosphere"),
        AspcapOutput.cn_h,
        AspcapOutput.e_cn_h,
        AspcapOutput.al_h,
        AspcapOutput.e_al_h,
        (AspcapOutput.ci_h + AspcapOutput.metals).alias("ci_h"),
        AspcapOutput.e_ci_h,
        (AspcapOutput.ca_h + AspcapOutput.metals).alias("ca_h"),
        AspcapOutput.e_ca_h,
        AspcapOutput.ce_h,
        AspcapOutput.e_ce_h,
        AspcapOutput.c_h,
        AspcapOutput.e_c_h,
        AspcapOutput.co_h,
        AspcapOutput.e_co_h,
        AspcapOutput.cr_h,
        AspcapOutput.e_cr_h,
        AspcapOutput.cu_h,
        AspcapOutput.e_cu_h,
        AspcapOutput.fe_h,
        AspcapOutput.e_fe_h,
        AspcapOutput.ge_h,
        AspcapOutput.e_ge_h,
        AspcapOutput.k_h,
        AspcapOutput.e_k_h,
        (AspcapOutput.mg_h + AspcapOutput.metals).alias("mg_h"),
        AspcapOutput.e_mg_h,
        AspcapOutput.mn_h,
        AspcapOutput.e_mn_h,
        AspcapOutput.na_h,
        AspcapOutput.e_na_h,
        AspcapOutput.nd_h,
        AspcapOutput.e_nd_h,
        AspcapOutput.ni_h,
        AspcapOutput.e_ni_h,
        AspcapOutput.n_h,
        AspcapOutput.e_n_h,
        (AspcapOutput.o_h + AspcapOutput.metals).alias("o_h"),
        AspcapOutput.e_o_h,
        AspcapOutput.p_h,
        AspcapOutput.e_p_h,
        AspcapOutput.rb_h,
        AspcapOutput.e_rb_h,
        (AspcapOutput.si_h + AspcapOutput.metals).alias("si_h"),
        AspcapOutput.e_si_h,
        (AspcapOutput.s_h + AspcapOutput.metals).alias("s_h"),
        AspcapOutput.e_s_h,
        (AspcapOutput.ti_h + AspcapOutput.metals).alias("ti_h"),
        AspcapOutput.e_ti_h,
        #AspcapOutput.tiii_h,
        #AspcapOutput.e_tiii_h,
        AspcapOutput.v_h,
        AspcapOutput.e_v_h,
        AspcapOutput.yb_h,
        AspcapOutput.e_yb_h,        
        AspcapOutput.bitmask_teff,
        AspcapOutput.bitmask_logg,
        AspcapOutput.bitmask_metals,
        AspcapOutput.bitmask_log10vdop,
        AspcapOutput.bitmask_o_mg_si_s_ca_ti,
        AspcapOutput.bitmask_lgvsini,
        AspcapOutput.bitmask_c.alias("bitmask_c_h_photosphere"),
        AspcapOutput.bitmask_n.alias("bitmask_n_h_photosphere"),
        AspcapOutput.bitmask_cn_h,
        AspcapOutput.bitmask_al_h,
        AspcapOutput.bitmask_ci_h,
        AspcapOutput.bitmask_ca_h,
        AspcapOutput.bitmask_ce_h,
        AspcapOutput.bitmask_c_h,
        AspcapOutput.bitmask_co_h,
        AspcapOutput.bitmask_cr_h,
        AspcapOutput.bitmask_cu_h,
        AspcapOutput.bitmask_fe_h,
        AspcapOutput.bitmask_ge_h,
        AspcapOutput.bitmask_k_h,
        AspcapOutput.bitmask_mg_h,
        AspcapOutput.bitmask_mn_h,
        AspcapOutput.bitmask_na_h,
        AspcapOutput.bitmask_nd_h,
        AspcapOutput.bitmask_ni_h,
        AspcapOutput.bitmask_n_h,
        AspcapOutput.bitmask_o_h,
        AspcapOutput.bitmask_p_h,
        AspcapOutput.bitmask_rb_h,
        AspcapOutput.bitmask_si_h,
        AspcapOutput.bitmask_s_h,
        AspcapOutput.bitmask_ti_h,
        #AspcapOutput.bitmask_tiii_h,
        AspcapOutput.bitmask_v_h,
        AspcapOutput.bitmask_yb_h,
        AspcapOutput.log_chisq_fit,
        AspcapOutput.log_chisq_fit_cn_h,
        AspcapOutput.log_chisq_fit_al_h,
        AspcapOutput.log_chisq_fit_ci_h,
        AspcapOutput.log_chisq_fit_ca_h,
        AspcapOutput.log_chisq_fit_ce_h,
        AspcapOutput.log_chisq_fit_c_h,
        AspcapOutput.log_chisq_fit_co_h,
        AspcapOutput.log_chisq_fit_cr_h,
        AspcapOutput.log_chisq_fit_cu_h,
        AspcapOutput.log_chisq_fit_fe_h,
        AspcapOutput.log_chisq_fit_ge_h,
        AspcapOutput.log_chisq_fit_k_h,
        AspcapOutput.log_chisq_fit_mg_h,
        AspcapOutput.log_chisq_fit_mn_h,
        AspcapOutput.log_chisq_fit_na_h,
        AspcapOutput.log_chisq_fit_nd_h,
        AspcapOutput.log_chisq_fit_ni_h,
        AspcapOutput.log_chisq_fit_n_h,
        AspcapOutput.log_chisq_fit_o_h,
        AspcapOutput.log_chisq_fit_p_h,
        AspcapOutput.log_chisq_fit_rb_h,
        AspcapOutput.log_chisq_fit_si_h,
        AspcapOutput.log_chisq_fit_s_h,
        AspcapOutput.log_chisq_fit_ti_h,
        AspcapOutput.log_chisq_fit_tiii_h,
        AspcapOutput.log_chisq_fit_v_h,
        AspcapOutput.log_chisq_fit_yb_h,        
    )
    t = all_star_table_v026(
        fields, 
        expression=(Task.parameters["chemical_abundance_task_ids"] != "[]")
    )
    aspcap_names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]

    for name in t.dtype.names:
        if name in aspcap_names:
            if name.startswith("log_chisq_fit"):
                t[name] = t[name].astype(float)
            elif name.startswith("bitmask_"):
                t[name][t[name] == None] = 0
                t[name] = t[name].astype(int)
            else:
                t[name] = t[name].astype(float)

    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    t_merge = join(t_aux, t, keys="cat_id")
    from astropy.table import unique
    return unique(t_merge, keys=["cat_id"])



def all_star_snow_white_v026():
    fields = (
        WhiteDwarfClassifierOutput.source_id.alias("cat_id"),
        DataProduct.id.alias("data_product_id"),
        DataProduct.release,
        DataProduct.filetype,
        DataProduct.kwargs,            
        WhiteDwarfClassifierOutput.wd_type,
        WhiteDwarfOutput.snr,
        WhiteDwarfOutput.teff,
        WhiteDwarfOutput.e_teff,
        WhiteDwarfOutput.logg,
        WhiteDwarfOutput.e_logg,
        WhiteDwarfOutput.v_rel,
        #WhiteDwarfOutput.e_v_rel,
        WhiteDwarfOutput.chi_sq,
        WhiteDwarfOutput.reduced_chi_sq,
        WhiteDwarfOutput.conditioned_on_parallax,
        WhiteDwarfOutput.conditioned_on_phot_g_mean_mag
    )
    q = (
        WhiteDwarfClassifierOutput
        .select(*fields)
        .distinct(WhiteDwarfClassifierOutput.source_id)
        .join(Task)
        .switch(WhiteDwarfClassifierOutput)
        .join(
            WhiteDwarfOutput,
            on=(WhiteDwarfClassifierOutput.source_id == WhiteDwarfOutput.source_id),
            join_type=JOIN.LEFT_OUTER
        )
        .join(DataProduct, on=(DataProduct.id == WhiteDwarfClassifierOutput.parent_data_product_id))
        .where(Task.version == "0.2.6")
        .dicts()
    )

    # TODO: Check that we still recover rows even if parent_data_product_id is None!
    rows = []
    for output in tqdm(q):
        kwargs = output.pop("kwargs")
        row = {**output, **kwargs}
        rows.append(row)

    names = [ea._alias if isinstance(ea, Alias) else ea.name for ea in fields]
    # put all other names where kwargs is
    kwarg_names = list(
        set.union(*(set(ea.keys()) for ea in rows)).difference(names)
    )

    for kwarg_name in kwarg_names[::-1]:
        names.insert(names.index("kwargs"), kwarg_name)
    names.pop(names.index("kwargs"))
    
    dtype = [int, int, str, str]
    dtype += [str] * len(kwarg_names)
    dtype += [str, float, float, float, float, float, float, float, float, float, float]

    t_param = Table(rows=rows, names=names, dtype=dtype)
    fields = (
        WhiteDwarfLineRatiosOutput.source_id.alias("cat_id"),
        ((WhiteDwarfLineRatiosOutput.wavelength_start + WhiteDwarfLineRatiosOutput.wavelength_end)/2).alias("wavelength_center"),
        WhiteDwarfLineRatiosOutput.line_ratio
    )

    q = (
        WhiteDwarfLineRatiosOutput
        .select(*fields)
        .join(Task)
        .where(Task.version == "0.2.6")
        .tuples()
    )
    ratios = {}
    for cat_id, wavelength_center, line_ratio in tqdm(q):
        ratios.setdefault(cat_id, {})
        ratios[cat_id][f"line_ratio_{wavelength_center:.0f}"] = line_ratio

    rows = [dict(cat_id=cat_id, **ratios[cat_id]) for cat_id in ratios]
    names = sorted(rows[0].keys())

    t_ratios = Table(rows=rows, names=names, dtype=[int] + [float] * (len(names) - 1))


    t = join(t_param, t_ratios, keys="cat_id", join_type="outer")

    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}

    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            for key in keys:
                aux_info[key.lower()].append(image[0].header[key])

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    for k in map(str.lower, ("J_MAG", "H_MAG", "K_MAG", "V_RAD", "E_V_RAD")): 
        t_aux[k] = t_aux[k].astype(float)
    t_final = join(t_aux, t, keys="cat_id")
    return t_final



def construct_apogee_allStar_apogeenet():
    return construct_apogee_allStar(
        '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-APOGEENet-0.2.6-v6_0_9-1.0.fits',
        "APOGEENet",
        [
            ("TEFF", "teff"),
            ("E_TEFF", "e_teff"),
            ("LOGG", "logg"),
            ("E_LOGG", "e_logg"),
            ("FE_H", "fe_h"),
            ("E_FE_H", "e_fe_h"),
            ("BITMASK_FLAG", "bitmask_flag"),
            ("SNR", "snr"),
        ]
    )

def construct_apogee_allStar_thepayne():
    return construct_apogee_allStar(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-ThePayne-0.2.6-v6_0_9-1.0.fits",
        "ThePayne",
        [
            ("TEFF", "teff"),
            ("LOGG", "logg"),
            ("V_TURB", "v_turb"),
            ("C_H", "c_h"),
            ("N_H", "n_h"),
            ("O_H", "o_h"),
            ("NA_H", "na_h"),
            ("MG_H", "mg_h"),
            ("AL_H", "al_h"),
            ("SI_H", "si_h"),
            ("P_H", "p_h"),
            ("S_H", "s_h"),
            ("K_H", "k_h"),
            ("CA_H", "ca_h"),
            ("TI_H", "ti_h"),
            ("V_H", "v_h"),
            ("CR_H", "cr_h"),
            ("MN_H", "mn_h"),
            ("FE_H", "fe_h"),
            ("CO_H", "co_h"),
            ("NI_H", "ni_h"),
            ("CU_H", "cu_h"),
            ("GE_H", "ge_h"),
            ("C12_C13", "c12_c13"),
            ("V_MACRO", "v_macro"),
            ("E_TEFF", "e_teff"),
            ("E_LOGG", "e_logg"),
            ("E_V_TURB", "e_v_turb"),
            ("E_C_H", "e_c_h"),
            ("E_N_H", "e_n_h"),
            ("E_O_H", "e_o_h"),
            ("E_NA_H", "e_na_h"),
            ("E_MG_H", "e_mg_h"),
            ("E_AL_H", "e_al_h"),
            ("E_SI_H", "e_si_h"),
            ("E_P_H", "e_p_h"),
            ("E_S_H", "e_s_h"),
            ("E_K_H", "e_k_h"),
            ("E_CA_H", "e_ca_h"),
            ("E_TI_H", "e_ti_h"),
            ("E_V_H", "e_v_h"),
            ("E_CR_H", "e_cr_h"),
            ("E_MN_H", "e_mn_h"),
            ("E_FE_H", "e_fe_h"),
            ("E_CO_H", "e_co_h"),
            ("E_NI_H", "e_ni_h"),
            ("E_CU_H", "e_cu_h"),
            ("E_GE_H", "e_ge_h"),
            ("E_C12_C13", "e_c12_c13"),
            ("E_V_MACRO", "e_v_macro"),
            ("BITMASK_FLAG", "bitmask_flag"),      
            ("SNR", "snr"),
            ("CHI_SQ", "chi_sq"),
            ("REDUCED_CHI_SQ", "reduced_chi_sq"),
        ]
    )


def construct_apogee_allStar_zetapayne():
    return construct_apogee_allStar(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-ZetaPayne-0.2.6-v6_0_9-1.0.fits",
        "ZetaPayne",
        [
            ("TEFF", "teff"),
            ("LOGG", "logg"),
            ("FE_H", "fe_h"),
            ("VSINI", "vsini"),
            ("V_MICRO", "v_micro"),
            ("V_REL", "v_rel"),
            ("E_TEFF", "e_teff"),
            ("E_LOGG", "e_logg"),
            ("E_FE_H", "e_fe_h"),
            ("E_VSINI", "e_vsini"),
            ("E_V_MICRO", "e_v_micro"),
            ("E_V_REL", "e_v_rel"),
            ("SNR", "snr"),
            ("CHI_SQ", "chi_sq"),
            ("REDUCED_CHI_SQ", "reduced_chi_sq")
        ]
    )

def construct_boss_allStar_snow_white():
    l = [
        'line_ratio_3880',
        'line_ratio_3892',
        'line_ratio_3932',
        'line_ratio_3965',
        'line_ratio_3968',
        'line_ratio_3975',
        'line_ratio_4023',
        'line_ratio_4102',
        'line_ratio_4125',
        'line_ratio_4340',
        'line_ratio_4390',
        'line_ratio_4468',
        'line_ratio_4675',
        'line_ratio_4715',
        'line_ratio_4860',
        'line_ratio_4925',
        'line_ratio_5015',
        'line_ratio_5080',
        'line_ratio_5875',
        'line_ratio_6560',
        'line_ratio_6685',
        'line_ratio_7070',
        'line_ratio_7282'
    ]
    mappings =         [
        ("WD_TYPE", "wd_type"),
        ("TEFF", "teff"),
        ("LOGG", "logg"),
        ("V_REL", "v_rel"),
        ("E_TEFF", "e_teff"),
        ("E_LOGG", "e_logg"),
        #("E_V_REL", "e_v_rel"),
        ("CONDITIONED_ON_PARALLAX", "conditioned_on_parallax"),
        ("CONDITIONED_ON_PHOT_G_MEAN_MAG", "conditioned_on_phot_g_mean_mag"),
        ("SNR", "snr"),
        ("CHI_SQ", "chi_sq"),
        ("REDUCED_CHI_SQ", "reduced_chi_sq"),
    ]
    for name in l:
        mappings.append((name.upper(), name))
    return construct_boss_allStar(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-WD-0.2.6-v6_0_9-1.0.fits",
        "SnowWhite",
        mappings,
        [
            ("LINE_RATIO_3880", "LINE RATIOS"),
        ]
    )

def construct_boss_allStar_slam(): 
    mappings = [(k.upper(), k) for k in ("teff", "logg", "fe_h", "e_teff", "e_logg", "e_fe_h", "initial_teff", "initial_logg", "initial_fe_h", "snr", "success", "chi_sq", "reduced_chi_sq")]
    return construct_boss_allStar(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-SLAM-0.2.6-v6_0_9-1.0.fits",
        "SLAM",
        mappings,
    )    

def construct_apogee_allStar_aspcap():
    mappings = [
        'teff',
        'e_teff',
        'logg',
        'e_logg',
        'metals',
        'e_metals',
        'log10vdop',
        'e_log10vdop',
        'o_mg_si_s_ca_ti',
        'e_o_mg_si_s_ca_ti',
        'lgvsini',
        'c_h_photosphere',
        'e_c_h_photosphere',
        'n_h_photosphere',
        'e_n_h_photosphere',
        'cn_h',
        'e_cn_h',
        'al_h',
        'e_al_h',
        'ci_h',
        'e_ci_h',
        'ca_h',
        'e_ca_h',
        'ce_h',
        'e_ce_h',
        'c_h',
        'e_c_h',
        'co_h',
        'e_co_h',
        'cr_h',
        'e_cr_h',
        'cu_h',
        'e_cu_h',
        'fe_h',
        'e_fe_h',
        'ge_h',
        'e_ge_h',
        'k_h',
        'e_k_h',
        'mg_h',
        'e_mg_h',
        'mn_h',
        'e_mn_h',
        'na_h',
        'e_na_h',
        'nd_h',
        'e_nd_h',
        'ni_h',
        'e_ni_h',
        'n_h',
        'e_n_h',
        'o_h',
        'e_o_h',
        'p_h',
        'e_p_h',
        'rb_h',
        'e_rb_h',
        'si_h',
        'e_si_h',
        's_h',
        'e_s_h',
        'ti_h',
        'e_ti_h',
        'v_h',
        'e_v_h',
        'yb_h',
        'e_yb_h',
        'bitmask_teff',
        'bitmask_logg',
        'bitmask_metals',
        'bitmask_log10vdop',
        'bitmask_o_mg_si_s_ca_ti',
        'bitmask_lgvsini',
        'bitmask_c_h_photosphere',
        'bitmask_n_h_photosphere',
        'bitmask_cn_h',
        'bitmask_al_h',
        'bitmask_ci_h',
        'bitmask_ca_h',
        'bitmask_ce_h',
        'bitmask_c_h',
        'bitmask_co_h',
        'bitmask_cr_h',
        'bitmask_cu_h',
        'bitmask_fe_h',
        'bitmask_ge_h',
        'bitmask_k_h',
        'bitmask_mg_h',
        'bitmask_mn_h',
        'bitmask_na_h',
        'bitmask_nd_h',
        'bitmask_ni_h',
        'bitmask_n_h',
        'bitmask_o_h',
        'bitmask_p_h',
        'bitmask_rb_h',
        'bitmask_si_h',
        'bitmask_s_h',
        'bitmask_ti_h',
        'bitmask_v_h',
        'bitmask_yb_h',
        'snr',

        'log_chisq_fit',
        'log_chisq_fit_cn_h',
        'log_chisq_fit_al_h',
        'log_chisq_fit_ci_h',
        'log_chisq_fit_ca_h',
        'log_chisq_fit_ce_h',
        'log_chisq_fit_c_h',
        'log_chisq_fit_co_h',
        'log_chisq_fit_cr_h',
        'log_chisq_fit_cu_h',
        'log_chisq_fit_fe_h',
        'log_chisq_fit_ge_h',
        'log_chisq_fit_k_h',
        'log_chisq_fit_mg_h',
        'log_chisq_fit_mn_h',
        'log_chisq_fit_na_h',
        'log_chisq_fit_nd_h',
        'log_chisq_fit_ni_h',
        'log_chisq_fit_n_h',
        'log_chisq_fit_o_h',
        'log_chisq_fit_p_h',
        'log_chisq_fit_rb_h',
        'log_chisq_fit_si_h',
        'log_chisq_fit_s_h',
        'log_chisq_fit_ti_h',
        'log_chisq_fit_tiii_h',
        'log_chisq_fit_v_h',
        'log_chisq_fit_yb_h'    
    ]

    return construct_apogee_allStar(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-ASPCAP-0.2.6-v6_0_9-1.0.fits",
        "ASPCAP",
        [(k.upper(), k) for k in mappings]
    )


def construct_all_star_classifier():
    t = Table.read("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-Classifier-0.2.6-v6_0_9-1.0.fits")


    all_category_headers = [
        ("CAT_ID", "IDENTIFIERS"),
        ("CLASS", "CLASSIFICATION"),
        ("P_CV", "RELATIVE PROBABILITIES"),
        ("LP_CV", "LOG PROBABILITIES"),
    ]
    # Map with spAll file.
    all_mappings = [
        ("CAT_ID", t["cat_id"]),
        ("TASK_ID", t["task_id"]),
        ("OUTPUT_ID", t["output_id"]),
        ("CLASS", t["class"]),
        ("P_CV", t["p_cv"]),
        ("P_FGKM", t["p_fgkm"]),
        ("P_HOTSTAR", t["p_hotstar"]),
        ("P_WD", t["p_wd"]),
        ("P_SB2", t["p_sb2"]),
        ("P_YSO", t["p_yso"]),
        ("LP_CV", t["lp_cv"]),
        ("LP_FGKM", t["lp_fgkm"]),
        ("LP_HOTSTAR", t["lp_hotstar"]),
        ("LP_WD", t["lp_wd"]),
        ("LP_SB2", t["lp_sb2"]),
        ("LP_YSO", t["lp_yso"]),        
    ]
    
    columns = []
    for key, values in all_mappings:
        columns.append(
            fits.Column(
                name=key,
                array=values,
                unit=None,
                **fits_column_kwargs(np.array(values)),
            )
        )

    header = fits.Header(
        [
            BLANK_CARD,
            (" ", "METADATA", None),
            ("V_ASTRA", "0.2.6"),
            ("RUN2D", "v6_0_9"),
            ("APRED", "1.0"),
            ("PIPELINE", "Classifier"),
            ("CREATED", datetime.datetime.utcnow().strftime("%y-%m-%d %H:%M:%S")),
            FILLER_CARD,
        ]
    )
        
    hdu = fits.BinTableHDU.from_columns(
        columns,
        #header=header,
        # name=f"{header['INSTRMNT']}/{header['OBSRVTRY']}"
    )

    primary_hdu = fits.PrimaryHDU(header=header)
    add_glossary_comments(primary_hdu)

    add_table_category_headers(hdu, all_category_headers)
    add_glossary_comments(hdu)
    hdu_list = fits.HDUList([primary_hdu, hdu])
    add_check_sums(hdu_list)
    return hdu_list



def all_star_mdwarftype_v026():
    fields = (
        MDwarfTypeOutput.spectral_type,
        MDwarfTypeOutput.sub_type,
        MDwarfTypeOutput.chi_sq,
    )
    t = all_star_table_v026(fields)
    # Extract information from the headers of the mwmStar files, since the database has become unworkably slow.
    p = SDSSPath("sdss5")
    keys = (
        "CAT_ID", "TIC_ID", "GAIA_ID", "HEALPIX", "RA", "DEC", "GAIA_RA", "GAIA_DEC", "PLX", "E_PLX", "PMRA", "E_PMRA", "PMDE", "E_PMDE", "V_RAD", "E_V_RAD",
        "G_MAG", "BP_MAG", "RP_MAG", 
        "J_MAG", "H_MAG", "K_MAG", 
        "CARTON_0", "PROGRAMS", "MAPPERS"
    )
    aux_info = {key.lower(): [] for key in keys}
    default = {"CARTON_0": "", "PROGRAMS": "", "MAPPERS": "", "TIC_ID": -1, "GAIA_ID": -1, "HEALPIX": -1}
    for row in tqdm(t):
        path = p.full(**dict(zip(t.dtype.names, row)))
        with fits.open(path) as image:
            for key in keys:
                value = image[0].header[key]
                aux_info[key.lower()].append(value or default.get(key, np.nan))

    t_aux = Table(data=aux_info, names=list(map(str.lower, keys)))
    return join(t_aux, t, keys="cat_id")  


def construct_boss_allStar_mdwarftype(): 
    mappings = [(k.upper(), k) for k in ("spectral_type", "sub_type", "chi_sq")]
    return construct_boss_allStar(
        "tmp.fits",
        "MDwarfType",
        mappings,
    )    

if __name__ == "__main__":

    t = all_star_mdwarftype_v026()
    t.write("tmp.fits")

    t = all_star_thepayne_v026()
    t.write("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-ThePayne-0.2.6-v6_0_9-1.0.fits", overwrite=True)


    t = all_star_zeta_payne_v026()
    t.write("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-ZetaPayne-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = all_star_snow_white_v026()
    t.write("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-WD-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = all_star_apogeenet_v026()
    t.write("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-APOGEENet-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = all_star_slam_v026()
    t.write("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/allStar-SLAM-0.2.6-v6_0_9-1.0.fits", overwrite=True)


    t = construct_apogee_allStar_aspcap()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-ASPCAP-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = construct_boss_allStar_slam()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-SLAM-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = construct_boss_allStar_snow_white()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-SnowWhite-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = construct_apogee_allStar_zetapayne()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-ZetaPayne-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    
    t = construct_apogee_allStar_thepayne()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-ThePayne-0.2.6-v6_0_9-1.0.fits", overwrite=True)

    t = construct_apogee_allStar_apogeenet()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-APOGEENet-0.2.6-v6_0_9-1.0.fits", overwrite=True)


    t = construct_all_star_classifier()
    t.writeto("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.6/v6_0_9-1.0/results/astraAllStar-Classifier-0.2.6-v6_0_9-1.0.fits", overwrite=True)
