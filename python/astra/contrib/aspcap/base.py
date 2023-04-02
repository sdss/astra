import numpy as np
from tqdm import tqdm
from astra.base import task
from astra.utils import log
from typing import Iterable
from astra.database.astradb import DataProduct, database
from astra.contrib.aspcap.models import ASPCAPInitial, ASPCAPOutput, ASPCAPStellarParameters, ASPCAPAbundances
from astra.contrib.aspcap.abundances import get_species
from astra.contrib.aspcap.utils import get_species_label_references
from peewee import fn, Case, FloatField, Alias, chunked
from astra.contrib.ferre.bitmask import BitFlagNameMap, ParamBitMask


def translate_species(species):
    if species == "TiII":
        return "ti2"
    elif species == "CI":
        return "c1"
    else:
        return species.lower()

# TODO: this belongs in aspcap, not ferre.
class ASPCAPFlag(BitFlagNameMap):

    TEFF_WARN = (0, "WARNING on effective temperature (see PARAMFLAG[0] for details)")
    LOGG_WARN = (1, "WARNING on log g (see PARAMFLAG[1] for details)")
    V_MICRO_WARN = (2, "WARNING on vmicro (see PARAMFLAG[2] for details)")
    M_H_WARN = (3, "WARNING on metals (see PARAMFLAG[3] for details)")
    ALPHA_M_WARN = (4, "WARNING on [alpha/M] (see PARAMFLAG[4] for details)")
    C_M_WARN = (5, "WARNING on [C/M] (see PARAMFLAG[5] for details)")
    N_M_WARN = (6, "WARNING on [N/M] (see PARAMFLAG[6] for details)")
    STAR_WARN = (7, "WARNING overall for star: set if either TEFF/LOGG warn are set")

    TEFF_BAD = (8, "potentially BAD effective temperature (see PARAMFLAG[0] for details)")
    LOGG_BAD = (9, "potentially BAD log g (see PARAMFLAG[1] for details)")
    V_MICRO_BAD = (10, "potentially BAD vmicro (see PARAMFLAG[2] for details)")
    M_H_BAD = (11, "potentially BAD metals (see PARAMFLAG[3] for details)")
    ALPHA_M_BAD = (12, "potentially BAD [alpha/M] (see PARAMFLAG[4] for details)")
    C_M_BAD = (13, "potentially BAD [C/M] (see PARAMFLAG[5] for details)")
    N_M_BAD = (14, "potentially BAD [N/M] (see PARAMFLAG[6] for details)")
    STAR_BAD = (15, "BAD overall for star: set if any of TEFF, LOGG, CHI2, COLORTE, ROTATION, SN error are set, or any GRIDEDGE_BAD")
    NO_GRID = (16, "Not processed by any ASPCAP grid")
    FERRE_FAIL = (17, "FERRE failure (bad input?)")
    FERRE_TIMEOUT = (18, "FERRE timed out on or before this spectrum")
    MANY_COARSE_SOLUTIONS = (19, "Many equally good solutions in different grids in the coarse run")


def aspcap(ignore_abundances_before_task_id=48892845):
    """
    Create summary rows of ASPCAP results from previously executed FERRE runs.
    """

    # Maybe m_h, alpha_m is sufficient? Up to our Czar (Wheeler!)
    param_suffix = "_atm"

    fields = (
        ASPCAPStellarParameters.data_product_id,
        ASPCAPStellarParameters.source_id,
        ASPCAPStellarParameters.hdu,
        ASPCAPStellarParameters.snr,
        ASPCAPStellarParameters.telescope,
        ASPCAPStellarParameters.instrument,
        ASPCAPStellarParameters.obj,
        #ASPCAPStellarParameters.mjd,
        ASPCAPStellarParameters.plate,
        ASPCAPStellarParameters.field,
        #ASPCAPStellarParameters.fiber,
        #ASPCAPStellarParameters.apstar_pk,
        #ASPCAPStellarParameters.apvisit_pk,
        ASPCAPStellarParameters.teff,
        ASPCAPStellarParameters.e_teff,
        ASPCAPStellarParameters.bitmask_teff.alias("bitmask_teff"),
        ASPCAPStellarParameters.logg,
        ASPCAPStellarParameters.e_logg,
        ASPCAPStellarParameters.bitmask_logg.alias("bitmask_logg"),            
        ASPCAPStellarParameters.lgvsini.alias("v_sini"),
        ASPCAPStellarParameters.e_lgvsini.alias("e_v_sini"),
        ASPCAPStellarParameters.bitmask_lgvsini.alias("bitmask_v_sini"),            
        ASPCAPStellarParameters.log10vdop.alias("v_micro"),
        ASPCAPStellarParameters.e_log10vdop.alias("e_v_micro"),
        ASPCAPStellarParameters.bitmask_log10vdop.alias("bitmask_v_micro"),
        ASPCAPStellarParameters.metals.alias("m_h"),
        ASPCAPStellarParameters.e_metals.alias("e_m_h"),
        ASPCAPStellarParameters.bitmask_metals.alias("bitmask_m_h"),
        ASPCAPStellarParameters.o_mg_si_s_ca_ti.alias(f"alpha_m{param_suffix}"),
        ASPCAPStellarParameters.e_o_mg_si_s_ca_ti.alias(f"e_alpha_m{param_suffix}"),
        ASPCAPStellarParameters.bitmask_o_mg_si_s_ca_ti.alias(f"bitmask_alpha_m{param_suffix}"),
        ASPCAPStellarParameters.c.alias(f"c_m{param_suffix}"),
        ASPCAPStellarParameters.e_c.alias(f"e_c_m{param_suffix}"),
        ASPCAPStellarParameters.bitmask_c.alias(f"bitmask_c_m{param_suffix}"),
        ASPCAPStellarParameters.n.alias(f"n_m{param_suffix}"),
        ASPCAPStellarParameters.e_n.alias(f"e_n_m{param_suffix}"),
        ASPCAPStellarParameters.bitmask_n.alias(f"bitmask_n_m{param_suffix}"),
        ASPCAPStellarParameters.log_chisq_fit.alias("chisq"),
        ASPCAPStellarParameters.header_path,
        ASPCAPStellarParameters.ferre_timeout
    )
    float_field_names = [
        f.name for f in fields 
        if isinstance(f, FloatField) or (
            isinstance(f, Alias) and isinstance(f.node, FloatField)
        )
    ]

    q = (
        ASPCAPStellarParameters
        .select(*fields)
        .distinct(ASPCAPStellarParameters.task_id)
        .join(ASPCAPAbundances, on=(ASPCAPStellarParameters.data_product_id == ASPCAPAbundances.data_product_id))
        .where(ASPCAPAbundances.task_id > ignore_abundances_before_task_id)
        .dicts()
    )

    # Do a once-off query to find things where there were multiple equally good 
    # solutions in the coarse run.
    # TODO: Implement this as a flag that gets set after the penalized chi-sq values are calculated.
    A = ASPCAPInitial.alias()
    sq = (
        A.select(
            A.data_product_id, 
            A.hdu, 
            fn.MIN(A.penalized_log_chisq_fit).alias("min_chisq")
        )
        .group_by(A.data_product_id, A.hdu).alias("sq")
    )

    q_multiple_solutions = list(
        ASPCAPInitial
        .select(
            ASPCAPInitial.data_product_id, 
            ASPCAPInitial.hdu,
        )
        .join(sq, on=(
            (ASPCAPInitial.data_product_id == sq.c.data_product_id) 
        &   (ASPCAPInitial.hdu == sq.c.hdu) 
        &   (ASPCAPInitial.penalized_log_chisq_fit == sq.c.min_chisq)
            )
        )
        .group_by(ASPCAPInitial.data_product_id, ASPCAPInitial.hdu)
        .having(fn.COUNT(fn.DISTINCT(ASPCAPInitial.header_path)) > 1)
        .tuples()
    )
    
    flag_param = ParamBitMask()
    aspcap_flag_map = ASPCAPFlag()
    
    species_label_reference = get_species_label_references()
    
    aspcap_rows = []
    for row in tqdm(q):
        
        # Propagate uncertainties for log quantities.
        v_sini, e_v_sini = propagate_uncertainty_of_log10_quantity(
            row.get("v_sini", np.nan),
            row.get("e_v_sini", np.nan)
        )
        row.update(v_sini=v_sini, e_v_sini=e_v_sini)
        v_micro, e_v_micro = propagate_uncertainty_of_log10_quantity(
            row.get("v_micro", np.nan),
            row.get("e_v_micro", np.nan)
        )
        row.update(v_micro=v_micro, e_v_micro=e_v_micro)
        row["chisq"] = 10**row["chisq"]
        row["grid"] = parse_grid_name(row.pop("header_path"))

        # Get abundance details.
        # TODO: this will screw up when we do visits. need to match by all.
        q_abundances = (
            ASPCAPAbundances
            .select(
                ASPCAPAbundances.data_product_id,
                ASPCAPAbundances.source_id,
                ASPCAPAbundances.metals,
                ASPCAPAbundances.e_metals,
                ASPCAPAbundances.bitmask_metals.alias("bitmask_metals"),
                ASPCAPAbundances.o_mg_si_s_ca_ti,
                ASPCAPAbundances.e_o_mg_si_s_ca_ti,
                ASPCAPAbundances.bitmask_o_mg_si_s_ca_ti.alias("bitmask_o_mg_si_s_ca_ti"),
                ASPCAPAbundances.c,
                ASPCAPAbundances.e_c,
                ASPCAPAbundances.bitmask_c.alias("bitmask_c"),
                ASPCAPAbundances.n,
                ASPCAPAbundances.e_n,
                ASPCAPAbundances.bitmask_n.alias("bitmask_n"),
                ASPCAPAbundances.header_path,
                ASPCAPAbundances.weight_path,
                ASPCAPAbundances.log_chisq_fit
            )
            .where(
                (ASPCAPAbundances.data_product_id == row["data_product"])
            #&   (ASPCAPAbundances.source_id == row["source"])
            &   (ASPCAPAbundances.hdu == row["hdu"])
            &   (ASPCAPAbundances.task_id > ignore_abundances_before_task_id)
            )
            .dicts()            
        )

        for abundance_row in q_abundances:
            dirty_species = get_species(abundance_row["weight_path"])
            dirty_label, is_x_m = species_label_reference[dirty_species]

            # Cleanse.
            label = dirty_label.lower().replace(" ", "_")
            species = translate_species(dirty_species)

            value = abundance_row[f"{label}"]
            e_value = abundance_row[f"e_{label}"]
            bitmask = abundance_row[f"bitmask_{label}"]

            if is_x_m:
                value += row["m_h"]
                # Propagate uncertainties, ignoring covariance.
                e_value = np.sqrt(e_value**2 + row["e_m_h"]**2)

            # Put these values to the row.
            row[f"{species}_h"] = value
            row[f"e_{species}_h"] = e_value
            row[f"bitmask_{species}_h"] = bitmask
            row[f"chisq_{species}_h"] = 10**abundance_row["log_chisq_fit"]
        
        for k, v in row.items():
            if v is None and k in float_field_names:
                row[k] = np.nan

        # Flag shit.
        bitmask_bad, bitmask_warn = (0, 0)
        if row["bitmask_teff"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("TEFF_WARN")
            bitmask_warn |= aspcap_flag_map.get_value("STAR_WARN")
        if row["bitmask_logg"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("LOGG_WARN")
            bitmask_warn |= aspcap_flag_map.get_value("STAR_WARN")
        if row["bitmask_m_h"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("M_H_WARN")
        if row[f"bitmask_alpha_m{param_suffix}"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("ALPHA_M_WARN")
        if row["bitmask_v_micro"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("V_MICRO_WARN")
        if row[f"bitmask_c_m{param_suffix}"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("C_M_WARN")
        if row[f"bitmask_n_m{param_suffix}"] & flag_param.get_value("GRIDEDGE_WARN"):
            bitmask_warn |= aspcap_flag_map.get_value("N_M_WARN")
        
        if row["bitmask_teff"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("TEFF_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")
        if row["bitmask_logg"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("LOGG_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")
        if row["bitmask_m_h"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("M_H_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")

        if row[f"bitmask_alpha_m{param_suffix}"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("ALPHA_M_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")

        if row["bitmask_v_micro"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("V_MICRO_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")

        if row[f"bitmask_c_m{param_suffix}"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("C_M_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")

        if row[f"bitmask_n_m{param_suffix}"] & flag_param.get_value("GRIDEDGE_BAD"):
            bitmask_bad |= aspcap_flag_map.get_value("N_M_BAD")
            bitmask_bad |= aspcap_flag_map.get_value("STAR_BAD")

        if row.pop("ferre_timeout"):
            bitmask_bad |= aspcap_flag_map.get_value("FERRE_TIMEOUT")
            bitmask_bad |= aspcap_flag_map.get_value("FERRE_FAIL")
        
        fail_whale = -1000
        for key in ("teff", "logg", "m_h", f"alpha_m{param_suffix}", "v_micro", f"c_m{param_suffix}", f"n_m{param_suffix}"):
            if row[key] == fail_whale:
                bitmask_bad |= aspcap_flag_map.get_value("FERRE_FAIL")
                row[key] = np.nan
            if row[f"e_{key}"] == fail_whale:
                bitmask_bad |= aspcap_flag_map.get_value("FERRE_FAIL")
                row[f"e_{key}"] = np.nan


        # Check for equally good solutions in the initial coarse run.
        if ((row["data_product"], row["hdu"]) in q_multiple_solutions):
            bitmask_warn |= aspcap_flag_map.get_value("MANY_COARSE_SOLUTIONS")
    
        bitmask_aspcap = 0
        bitmask_aspcap |= bitmask_warn
        bitmask_aspcap |= bitmask_bad

        row.update(
            bitmask_aspcap=bitmask_aspcap,
            warnflag=bitmask_warn > 0,
            badflag=bitmask_bad > 0
        )
        
        # TODO: CREATE pipeline product.
        aspcap_rows.append(row)

    log.info(f"Bulk creating {len(aspcap_rows)} rows")
    database.create_tables([ASPCAPOutput])
    
    with database.atomic():
        for batch in chunked(aspcap_rows, 1000):
            ASPCAPOutput.insert_many(batch).execute()

    raise a




def parse_grid_name(header_path):
    return header_path.split("/")[-2].split("_")[0]



def propagate_uncertainty_of_log10_quantity(log10_y, e_log10_y, fill_value=np.nan):
    if log10_y is None or log10_y is None:
        return (fill_value, fill_value)
    y = 10**log10_y
    e_y = y * np.log(10) * e_log10_y
    return (y, e_y)




