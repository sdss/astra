
from astropy.table import Table
import os
import numpy as np
from itertools import combinations
from tqdm import tqdm
from astra import __version__
from astra.utils import expand_path
import pickle


overwrite = True


def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan
    

def select_pairwise_combinations(
    output_path, 
    query, 
    field_names, 
    group_by=("source_pk", ), 
    meta_keys=("spectrum_pk", "task_pk", "snr"),
    overwrite=False,
    exclude_edges=True,
    limit=None
):    
    path = expand_path(f"$MWM_ASTRA/{__version__}/aux/{output_path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        with open(path, "rb") as fp:
            pairwise = pickle.load(fp)
        return pairwise
    
    results = Table(rows=list(query))
    # exclude values on the edge
    percentiles = {}
    keep = np.ones(len(results), dtype=bool)
    for name in field_names:
        v = np.array(results[name]).astype(float)
        e_v = np.array(results[f"e_{name}"]).astype(float)
        lower, upper = np.nanpercentile(v, [5, 95])
        if np.isfinite([lower, upper]).all():            
            print(name, lower, upper)
            keep *= (upper >= v) & (v >= lower)
            keep *= (e_v > 0) & (1000 > e_v)
    
    results = results[keep]
    
    pairwise = { k: [] for k in group_by }
    for name in field_names:        
        pairwise.update({
            f"{name}_0": [],
            f"{name}_1": [],
            f"e_{name}_0": [],
            f"e_{name}_1": [],
        })
    for name in meta_keys:
        pairwise.update({
            f"{name}_0": [],
            f"{name}_1": [],
        })
    
    for group in tqdm(results.group_by(group_by).groups):
        G = len(group)
        if G == 1:
            continue

        group_by_values = [group[k][0] for k in group_by]
        
        for rows in combinations(group, 2):
            
            # group-level metadata keys first
            for k, v in zip(group_by, group_by_values):
                pairwise[k].append(v)

            # Then pairwise metadata keys
            for suffix, row in enumerate(rows):
                
                for name in meta_keys:                
                    pairwise[f"{name}_{suffix}"].append(safe_float(row[name]))
                
                for name in field_names:
                    pairwise[f"{name}_{suffix}"].append(safe_float(row[name]))
                    pairwise[f"e_{name}_{suffix}"].append(safe_float(row[f"e_{name}"]))
        
    for k in pairwise.keys():
        pairwise[k] = np.array(pairwise[k])

    with open(path, "wb") as fp:
        pickle.dump(pairwise, fp)
        
    print(f"Results written to {path}")
    return pairwise
        
    
from scipy import stats



def get_names(pairwise):
    names = []
    for k in pairwise.keys():
        if k.startswith("e_") and k.endswith("_0"):
            names.append(k[2:-2])
    return tuple(names)

# Compute grid of corrections.
def compute_corrections(
    output_path,
    pairwise,
    z_bins=np.linspace(-5, 5, 100),
    scales=np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2]),
    offsets=dict(
        teff=np.logspace(-1, 2, 10),
        default=np.logspace(-2, 0, 10),
    ),
    overwrite=False,
):  
    
    path = expand_path(f"$MWM_ASTRA/{__version__}/aux/{output_path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        with open(path, "rb") as fp:
            r = pickle.load(fp)
        return r
    
    names = get_names(pairwise)
    
    meta = {
        "bin_edges": None,
        "reference_pdf": None,
    }
    results = {}
    
    for name in names:    
        
        var_0, var_1 = (pairwise[f"e_{name}_0"]**2 , pairwise[f"e_{name}_1"]**2)
        delta = (pairwise[f"{name}_0"] - pairwise[f"{name}_1"])
        
        use = (
            np.isfinite(delta * var_0 * var_1) 
        &   (pairwise[f"e_{name}_0"] >= 0) 
        &   (pairwise[f"e_{name}_1"] >= 0)
        )
        delta = delta[use]
        var_0, var_1 = (var_0[use], var_1[use])
        
        x, y = (offsets.get(name, offsets["default"]), scales)
        grid = np.meshgrid(x, y)
        grid_offsets, grid_scales = (grid[0].flatten(), grid[1].flatten())
        
        costs, best_index, best_cost, best_z_pdf = ([], None, None, None)
        for i, (o, s) in tqdm(enumerate(zip(grid_offsets, grid_scales)), total=x.size * y.size):
            inv_z_e = 1/np.sqrt((s**2 * var_0) + (s**2 * var_1) + (2*o)**2)
            z_pdf, bin_edges = np.histogram(delta * inv_z_e, bins=z_bins, density=True)            
            if meta["reference_pdf"] is None:
                meta.update(
                    bin_edges=bin_edges,
                    reference_pdf=stats.norm.pdf(bin_edges[:-1] + 0.5 * np.diff(bin_edges)[0], loc=0, scale=1)
                )
            cost = np.sum((z_pdf - meta["reference_pdf"])**2)
            if best_cost is None or cost < best_cost:
                best_index, best_cost, best_z_pdf = (i, cost, z_pdf)
            costs.append(cost)

        results[name] = dict(
            N=delta.size,
            costs=costs,
            offsets=grid_offsets,
            scales=grid_scales,
            offset=grid_offsets[best_index],
            scale=grid_scales[best_index],
            best_index=best_index,
            best_cost=best_cost,
            best_z_pdf=best_z_pdf,            
        )
    
    with open(path, "wb") as fp:
        pickle.dump((results, meta), fp)
    print(f"Results written to {path}")
    return (results, meta)
    

from astra.models import ASPCAP, ApogeeCoaddedSpectrumInApStar

pipeline_model = ASPCAP

q = (
    pipeline_model
    .select(
        ApogeeCoaddedSpectrumInApStar.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.raw_teff.alias("teff"),
        pipeline_model.raw_e_teff.alias("e_teff"),
        pipeline_model.raw_logg.alias("logg"),
        pipeline_model.raw_e_logg.alias("e_logg"),
        pipeline_model.raw_v_micro.alias("v_micro"),
        (pipeline_model.raw_e_v_micro / pipeline_model.raw_v_micro).alias("e_v_micro"),
        pipeline_model.raw_v_sini.alias("v_sini"),
        (pipeline_model.raw_e_v_sini / pipeline_model.raw_v_sini).alias("e_v_sini"),
        pipeline_model.raw_m_h_atm.alias("m_h_atm"),
        pipeline_model.raw_e_m_h_atm.alias("e_m_h_atm"),
        pipeline_model.raw_alpha_m_atm.alias("alpha_m_atm"),
        pipeline_model.raw_e_alpha_m_atm.alias("e_alpha_m_atm"),
        pipeline_model.raw_c_m_atm.alias("c_m_atm"),
        pipeline_model.raw_e_c_m_atm.alias("e_c_m_atm"),
        pipeline_model.raw_n_m_atm.alias("n_m_atm"),
        pipeline_model.raw_e_n_m_atm.alias("e_n_m_atm"),
        pipeline_model.raw_al_h.alias("al_h"),
        pipeline_model.raw_e_al_h.alias("e_al_h"),
        pipeline_model.raw_c_12_13.alias("c_12_13"),
        pipeline_model.raw_e_c_12_13.alias("e_c_12_13"),
        pipeline_model.raw_ca_h.alias("ca_h"),
        pipeline_model.raw_e_ca_h.alias("e_ca_h"),
        pipeline_model.raw_ce_h.alias("ce_h"),
        pipeline_model.raw_e_ce_h.alias("e_ce_h"),
        pipeline_model.raw_c_1_h.alias("c_1_h"),
        pipeline_model.raw_e_c_1_h.alias("e_c_1_h"),
        pipeline_model.raw_c_h.alias("c_h"),
        pipeline_model.raw_e_c_h.alias("e_c_h"),
        pipeline_model.raw_co_h.alias("co_h"),
        pipeline_model.raw_e_co_h.alias("e_co_h"),
        pipeline_model.raw_cr_h.alias("cr_h"),
        pipeline_model.raw_e_cr_h.alias("e_cr_h"),
        pipeline_model.raw_cu_h.alias("cu_h"),
        pipeline_model.raw_e_cu_h.alias("e_cu_h"),
        pipeline_model.raw_fe_h.alias("fe_h"),
        pipeline_model.raw_e_fe_h.alias("e_fe_h"),
        pipeline_model.raw_k_h.alias("k_h"),
        pipeline_model.raw_e_k_h.alias("e_k_h"),
        pipeline_model.raw_mg_h.alias("mg_h"),
        pipeline_model.raw_e_mg_h.alias("e_mg_h"),
        pipeline_model.raw_mn_h.alias("mn_h"),
        pipeline_model.raw_e_mn_h.alias("e_mn_h"),
        pipeline_model.raw_na_h.alias("na_h"),
        pipeline_model.raw_e_na_h.alias("e_na_h"),
        pipeline_model.raw_nd_h.alias("nd_h"),
        pipeline_model.raw_e_nd_h.alias("e_nd_h"),
        pipeline_model.raw_ni_h.alias("ni_h"),
        pipeline_model.raw_e_ni_h.alias("e_ni_h"),
        pipeline_model.raw_n_h.alias("n_h"),
        pipeline_model.raw_e_n_h.alias("e_n_h"),
        pipeline_model.raw_o_h.alias("o_h"),
        pipeline_model.raw_e_o_h.alias("e_o_h"),
        pipeline_model.raw_p_h.alias("p_h"),
        pipeline_model.raw_e_p_h.alias("e_p_h"),
        pipeline_model.raw_si_h.alias("si_h"),
        pipeline_model.raw_e_si_h.alias("e_si_h"),
        pipeline_model.raw_s_h.alias("s_h"),
        pipeline_model.raw_e_s_h.alias("e_s_h"),
        pipeline_model.raw_ti_h.alias("ti_h"),
        pipeline_model.raw_e_ti_h.alias("e_ti_h"),
        pipeline_model.raw_ti_2_h.alias("ti_2_h"),
        pipeline_model.raw_e_ti_2_h.alias("e_ti_2_h"),
        pipeline_model.raw_v_h.alias("v_h"),
        pipeline_model.raw_e_v_h.alias("e_v_h"),
    )
    .distinct(ApogeeCoaddedSpectrumInApStar.spectrum_pk)
    .join(ApogeeCoaddedSpectrumInApStar, on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == pipeline_model.spectrum_pk))
    .where(ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3"))) # exclude daily
    .where(~pipeline_model.flag_bad)
    .dicts()
)
field_names = (    
    "teff",
    "logg",
    "v_micro",
    "v_sini",
    "m_h_atm",
    "alpha_m_atm",
    "c_m_atm",
    "n_m_atm",
    "al_h",
    "c_12_13",
    "ca_h",
    "ce_h",
    "c_1_h",
    "c_h",
    "co_h",
    "cr_h",
    "cu_h",
    "fe_h",
    "k_h",
    "mg_h",
    "mn_h",
    "na_h",
    "nd_h",
    "ni_h",
    "n_h",
    "o_h",
    "p_h",
    "si_h",
    "s_h",
    "ti_h",
    "ti_2_h",
    "v_h",
)

pairwise = select_pairwise_combinations("ASPCAP.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("ASPCAP_corrections.pkl", pairwise, overwrite=overwrite)

'''

from astra.models import ApogeeNetV2, ApogeeVisitSpectrumInApStar, ApogeeVisitSpectrum

pipeline_model = ApogeeNetV2

q = (
    pipeline_model
    .select(
        ApogeeVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff,
        pipeline_model.logg,
        pipeline_model.fe_h,
        pipeline_model.e_teff,
        pipeline_model.e_logg,
        pipeline_model.e_fe_h,
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(ApogeeVisitSpectrumInApStar, on=(ApogeeVisitSpectrumInApStar.spectrum_pk == pipeline_model.spectrum_pk))
    .join(ApogeeVisitSpectrum, on=(ApogeeVisitSpectrum.spectrum_pk == ApogeeVisitSpectrumInApStar.drp_spectrum_pk))
    .where(pipeline_model.result_flags == 0)
    .where(pipeline_model.v_astra == __version__)
    .dicts()
)

field_names = ("teff", "logg", "fe_h")
pairwise = select_pairwise_combinations("ApogeeNetV2.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("ApogeeNetV2_corrections.pkl", pairwise, overwrite=overwrite)
'''


from astra.models import ApogeeNet, ApogeeVisitSpectrumInApStar, ApogeeVisitSpectrum

pipeline_model = ApogeeNet

from astra.models.apogeenet import apply_result_flags, apply_noise_model

apply_result_flags()

q = (
    pipeline_model
    .select(
        ApogeeVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.raw_teff.alias("teff"),
        pipeline_model.raw_logg.alias("logg"),
        pipeline_model.raw_fe_h.alias("fe_h"),
        pipeline_model.raw_e_teff.alias("e_teff"),
        pipeline_model.raw_e_logg.alias("e_logg"),
        pipeline_model.raw_e_fe_h.alias("e_fe_h"),
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(ApogeeVisitSpectrumInApStar, on=(ApogeeVisitSpectrumInApStar.spectrum_pk == pipeline_model.spectrum_pk))
    .join(ApogeeVisitSpectrum, on=(ApogeeVisitSpectrum.spectrum_pk == ApogeeVisitSpectrumInApStar.drp_spectrum_pk))
    .where(pipeline_model.result_flags == 0)
    .where(pipeline_model.v_astra == __version__)
    .dicts()
)
field_names = ("teff", "logg", "fe_h")
pairwise = select_pairwise_combinations("ApogeeNet.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("ApogeeNet_corrections.pkl", pairwise, overwrite=overwrite)

# Apply corrections 
apply_noise_model()



from astra.models import AstroNN, ApogeeVisitSpectrumInApStar

pipeline_model = AstroNN

q = (
    pipeline_model
    .select(
        ApogeeVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff,
        pipeline_model.e_teff,
        pipeline_model.logg,
        pipeline_model.e_logg,
        pipeline_model.c_h,
        pipeline_model.e_c_h,
        pipeline_model.c_1_h,
        pipeline_model.e_c_1_h,
        pipeline_model.n_h,
        pipeline_model.e_n_h,
        pipeline_model.o_h,
        pipeline_model.e_o_h,
        pipeline_model.na_h,
        pipeline_model.e_na_h,
        pipeline_model.mg_h,
        pipeline_model.e_mg_h,
        pipeline_model.al_h,
        pipeline_model.e_al_h,
        pipeline_model.si_h,
        pipeline_model.e_si_h,
        pipeline_model.p_h,
        pipeline_model.e_p_h,
        pipeline_model.s_h,
        pipeline_model.e_s_h,
        pipeline_model.k_h,
        pipeline_model.e_k_h,
        pipeline_model.ca_h,
        pipeline_model.e_ca_h,
        pipeline_model.ti_h,
        pipeline_model.e_ti_h,
        pipeline_model.ti_2_h,
        pipeline_model.e_ti_2_h,
        pipeline_model.v_h,
        pipeline_model.e_v_h,
        pipeline_model.cr_h,
        pipeline_model.e_cr_h,
        pipeline_model.mn_h,
        pipeline_model.e_mn_h,
        pipeline_model.fe_h,
        pipeline_model.e_fe_h,
        pipeline_model.co_h,
        pipeline_model.e_co_h,
        pipeline_model.ni_h,
        pipeline_model.e_ni_h,
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(ApogeeVisitSpectrumInApStar, on=(ApogeeVisitSpectrumInApStar.spectrum_pk == pipeline_model.spectrum_pk))
    .join(ApogeeVisitSpectrum, on=(ApogeeVisitSpectrum.spectrum_pk == ApogeeVisitSpectrumInApStar.drp_spectrum_pk))
    .where(pipeline_model.result_flags == 0)
    .where(pipeline_model.v_astra == __version__)
    .dicts()
)
field_names = (
    "teff",
    "logg",
    "c_h",
    "c_1_h",
    "n_h",
    "o_h",
    "na_h",
    "mg_h",
    "al_h",
    "si_h",
    "p_h",
    "s_h",
    "k_h",
    "ca_h",
    "ti_h",
    "ti_2_h",
    "v_h",
    "cr_h",
    "mn_h",
    "fe_h",
    "co_h",
    "ni_h",
)

pairwise = select_pairwise_combinations("AstroNN.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("AstroNN_corrections.pkl", pairwise, overwrite=overwrite)



'''
from astra.models import Corv, BossVisitSpectrum

pipeline_model = Corv

q = (
    pipeline_model
    .select(
        BossVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff,
        pipeline_model.logg,
        pipeline_model.e_teff,
        pipeline_model.e_logg,
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(BossVisitSpectrum, on=(BossVisitSpectrum.spectrum_pk == pipeline_model.spectrum_pk))
    .where(BossVisitSpectrum.run2d == "v6_1_3")
    #.where(pipeline_model.result_flags == 0)
    .dicts()
)
field_names = ("teff", "logg")

pairwise = select_pairwise_combinations("Corv.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("Corv_corrections.pkl", pairwise, overwrite=overwrite)
'''

from peewee import fn
from astra.models import SnowWhite, BossVisitSpectrum

pipeline_model = SnowWhite

q = (
    pipeline_model
    .select(
        BossVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff.alias("teff"),
        pipeline_model.logg.alias("logg"),
        pipeline_model.e_teff.alias("e_teff"),
        pipeline_model.e_logg.alias("e_logg"),
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(BossVisitSpectrum, on=(BossVisitSpectrum.spectrum_pk == pipeline_model.spectrum_pk))
    .where(
        pipeline_model.teff.is_null(False)
    &   (pipeline_model.logg > 7)
    &   (9.5 > pipeline_model.logg)
    &   (fn.abs(pipeline_model.teff - 13_000) > 250) # lots of bunching up at the nodes and edges
    )
    .where(pipeline_model.logg > 7)
    .where(BossVisitSpectrum.run2d == "v6_1_3")
    .where(pipeline_model.v_astra == "0.6.0")
    .dicts()
)

field_names = ("teff", "logg")

pairwise = select_pairwise_combinations("SnowWhite.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("SnowWhite_corrections.pkl", pairwise, overwrite=overwrite)

# ThePayne

from astra.models import ThePayne, ApogeeVisitSpectrumInApStar

pipeline_model = ThePayne

q = (
    pipeline_model
    .select(
        ApogeeVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff.alias("teff"),
        pipeline_model.e_teff.alias("e_teff"),
        pipeline_model.v_turb.alias("v_turb"),
        pipeline_model.e_v_turb.alias("e_v_turb"),
        pipeline_model.logg.alias("logg"),
        pipeline_model.e_logg.alias("e_logg"),
        pipeline_model.c_h.alias("c_h"),	
        pipeline_model.e_c_h.alias("e_c_h"),	
        pipeline_model.n_h.alias("n_h"),	
        pipeline_model.e_n_h.alias("e_n_h"),	
        pipeline_model.o_h.alias("o_h"),	
        pipeline_model.e_o_h.alias("e_o_h"),	
        pipeline_model.na_h.alias("na_h"),	
        pipeline_model.e_na_h.alias("e_na_h"),	
        pipeline_model.mg_h.alias("mg_h"),	
        pipeline_model.e_mg_h.alias("e_mg_h"),	
        pipeline_model.al_h.alias("al_h"),	
        pipeline_model.e_al_h.alias("e_al_h"),	
        pipeline_model.si_h.alias("si_h"),	
        pipeline_model.e_si_h.alias("e_si_h"),	
        pipeline_model.p_h.alias("p_h"),	
        pipeline_model.e_p_h.alias("e_p_h"),	
        pipeline_model.s_h.alias("s_h"),	
        pipeline_model.e_s_h.alias("e_s_h"),	
        pipeline_model.k_h.alias("k_h"),	
        pipeline_model.e_k_h.alias("e_k_h"),	
        pipeline_model.ca_h.alias("ca_h"),	
        pipeline_model.e_ca_h.alias("e_ca_h"),	
        pipeline_model.ti_h.alias("ti_h"),	
        pipeline_model.e_ti_h.alias("e_ti_h"),	
        pipeline_model.v_h.alias("v_h"),	
        pipeline_model.e_v_h.alias("e_v_h"),	
        pipeline_model.cr_h.alias("cr_h"),	
        pipeline_model.e_cr_h.alias("e_cr_h"),	
        pipeline_model.mn_h.alias("mn_h"),	
        pipeline_model.e_mn_h.alias("e_mn_h"),	
        pipeline_model.fe_h.alias("fe_h"),
        pipeline_model.e_fe_h.alias("e_fe_h"),
        pipeline_model.co_h.alias("co_h"),	
        pipeline_model.e_co_h.alias("e_co_h"),	
        pipeline_model.ni_h.alias("ni_h"),	
        pipeline_model.e_ni_h.alias("e_ni_h"),	
        pipeline_model.cu_h.alias("cu_h"),	
        pipeline_model.e_cu_h.alias("e_cu_h"),	
        pipeline_model.ge_h.alias("ge_h"),	
        pipeline_model.e_ge_h.alias("e_ge_h"),
        pipeline_model.c12_c13.alias("c12_c13"),	
        pipeline_model.e_c12_c13.alias("e_c12_c13"),
        pipeline_model.v_macro.alias("v_macro"),
        pipeline_model.e_v_macro.alias("e_v_macro")
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(ApogeeVisitSpectrumInApStar, on=(ApogeeVisitSpectrumInApStar.spectrum_pk == pipeline_model.spectrum_pk))
    .join(ApogeeVisitSpectrum, on=(ApogeeVisitSpectrum.spectrum_pk == ApogeeVisitSpectrumInApStar.drp_spectrum_pk))
    .where(pipeline_model.result_flags == 0)
    .where(pipeline_model.v_astra == __version__)
    .where(ApogeeVisitSpectrum.apred.in_(("dr17", "1.3")))
    .dicts()
)

field_names = (
    "teff",
    "v_turb",
    "logg",
    "c_h",	
    "n_h",	
    "o_h",	
    "na_h",	
    "mg_h",	
    "al_h",	
    "si_h",	
    "p_h",	
    "s_h",	
    "k_h",	
    "ca_h",	
    "ti_h",	
    "v_h",	
    "cr_h",	
    "mn_h",	
    "fe_h",
    "co_h",	
    "ni_h",	
    "cu_h",	
    "ge_h",	
    "c12_c13",	
    "v_macro",	
)
pairwise = select_pairwise_combinations("ThePayne.pkl", q, field_names, overwrite=overwrite, limit=100_000)
corrections = compute_corrections("ThePayne_corrections.pkl", pairwise, overwrite=overwrite)

# TheCannon


from astra.models import TheCannon, ApogeeCoaddedSpectrumInApStar

pipeline_model = TheCannon

q = (
    pipeline_model
    .select(
        ApogeeCoaddedSpectrumInApStar.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff,
        pipeline_model.e_teff,
        pipeline_model.logg,
        pipeline_model.e_logg,
        pipeline_model.fe_h,
        pipeline_model.e_fe_h,
        pipeline_model.v_micro,
        pipeline_model.e_v_micro,
        pipeline_model.v_macro,
        pipeline_model.e_v_macro,
        pipeline_model.c_fe,
        pipeline_model.e_c_fe,
        pipeline_model.n_fe,
        pipeline_model.e_n_fe,
        pipeline_model.o_fe,
        pipeline_model.e_o_fe,
        pipeline_model.na_fe,
        pipeline_model.e_na_fe,
        pipeline_model.mg_fe,
        pipeline_model.e_mg_fe,
        pipeline_model.al_fe,
        pipeline_model.e_al_fe,
        pipeline_model.si_fe,
        pipeline_model.e_si_fe,
        pipeline_model.s_fe,
        pipeline_model.e_s_fe,
        pipeline_model.k_fe,
        pipeline_model.e_k_fe,
        pipeline_model.ca_fe,
        pipeline_model.e_ca_fe,
        pipeline_model.ti_fe,
        pipeline_model.e_ti_fe,
        pipeline_model.v_fe,
        pipeline_model.e_v_fe,
        pipeline_model.cr_fe,
        pipeline_model.e_cr_fe,
        pipeline_model.mn_fe,
        pipeline_model.e_mn_fe,
        pipeline_model.ni_fe,
        pipeline_model.e_ni_fe,
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(ApogeeCoaddedSpectrumInApStar, on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == pipeline_model.spectrum_pk))
    .where(pipeline_model.result_flags == 0)
    .where(ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))
    .dicts()
)

field_names = (
    "teff",
    "logg",
    "fe_h",
    "v_micro",
    "v_macro",
    "c_fe",
    "n_fe",
    "o_fe",
    "na_fe",
    "mg_fe",
    "al_fe",
    "si_fe",
    "s_fe",
    "k_fe",
    "ca_fe",
    "ti_fe",
    "v_fe",
    "cr_fe",
    "mn_fe",
    "ni_fe",
)

pairwise = select_pairwise_combinations("TheCannon.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("TheCannon_corrections.pkl", pairwise, overwrite=overwrite)





'''
from astra.models import HotPayne, BossVisitSpectrum


pipeline_model = HotPayne

q = (
    pipeline_model
    .select(
        BossVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff,
        pipeline_model.logg,
        pipeline_model.fe_h,
        pipeline_model.v_micro,
        pipeline_model.v_sini,
        pipeline_model.he_h,
        pipeline_model.c_h,
        pipeline_model.n_h,
        pipeline_model.o_h,
        pipeline_model.si_h,
        pipeline_model.s_h,
        pipeline_model.teff_fullspec,
        pipeline_model.logg_fullspec,
        pipeline_model.fe_h_fullspec,
        pipeline_model.v_micro_fullspec,
        pipeline_model.v_sini_fullspec,
        pipeline_model.he_h_fullspec,
        pipeline_model.c_h_fullspec,
        pipeline_model.n_h_fullspec,
        pipeline_model.o_h_fullspec,
        pipeline_model.si_h_fullspec,
        pipeline_model.s_h_fullspec,
        pipeline_model.teff_hmasked,
        pipeline_model.logg_hmasked,
        pipeline_model.fe_h_hmasked,
        pipeline_model.v_micro_hmasked,
        pipeline_model.v_sini_hmasked,
        pipeline_model.he_h_hmasked,
        pipeline_model.c_h_hmasked,
        pipeline_model.n_h_hmasked,
        pipeline_model.o_h_hmasked,
        pipeline_model.si_h_hmasked,
        pipeline_model.s_h_hmasked,
        pipeline_model.e_teff,
        pipeline_model.e_logg,
        pipeline_model.e_fe_h,
        pipeline_model.e_v_micro,
        pipeline_model.e_v_sini,
        pipeline_model.e_he_h,
        pipeline_model.e_c_h,
        pipeline_model.e_n_h,
        pipeline_model.e_o_h,
        pipeline_model.e_si_h,
        pipeline_model.e_s_h,
        pipeline_model.e_teff_fullspec,
        pipeline_model.e_logg_fullspec,
        pipeline_model.e_fe_h_fullspec,
        pipeline_model.e_v_micro_fullspec,
        pipeline_model.e_v_sini_fullspec,
        pipeline_model.e_he_h_fullspec,
        pipeline_model.e_c_h_fullspec,
        pipeline_model.e_n_h_fullspec,
        pipeline_model.e_o_h_fullspec,
        pipeline_model.e_si_h_fullspec,
        pipeline_model.e_s_h_fullspec,
        pipeline_model.e_teff_hmasked,
        pipeline_model.e_logg_hmasked,
        pipeline_model.e_fe_h_hmasked,
        pipeline_model.e_v_micro_hmasked,
        pipeline_model.e_v_sini_hmasked,
        pipeline_model.e_he_h_hmasked,
        pipeline_model.e_c_h_hmasked,
        pipeline_model.e_n_h_hmasked,
        pipeline_model.e_o_h_hmasked,
        pipeline_model.e_si_h_hmasked,
        pipeline_model.e_s_h_hmasked,
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(BossVisitSpectrum, on=(BossVisitSpectrum.spectrum_pk == pipeline_model.spectrum_pk))
    #.where(pipeline_model.result_flags == 0)
    .dicts()
)
field_names = (
    "teff",
    "logg",
    "fe_h",
    "v_micro",
    "v_sini",
    "he_h",
    "c_h",
    "n_h",
    "o_h",
    "si_h",
    "s_h",
    "teff_fullspec",
    "logg_fullspec",
    "fe_h_fullspec",
    "v_micro_fullspec",
    "v_sini_fullspec",
    "he_h_fullspec",
    "c_h_fullspec",
    "n_h_fullspec",
    "o_h_fullspec",
    "si_h_fullspec",
    "s_h_fullspec",
    "teff_hmasked",
    "logg_hmasked",
    "fe_h_hmasked",
    "v_micro_hmasked",
    "v_sini_hmasked",
    "he_h_hmasked",
    "c_h_hmasked",
    "n_h_hmasked",
    "o_h_hmasked",
    "si_h_hmasked",
    "s_h_hmasked",
)
pairwise = select_pairwise_combinations("HotPayne.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("HotPayne_corrections.pkl", pairwise, overwrite=overwrite)
'''

from astra.models import Slam, BossVisitSpectrum

pipeline_model = Slam

q = (
    pipeline_model
    .select(
        BossVisitSpectrum.snr,
        pipeline_model.source_pk,
        pipeline_model.task_pk,
        pipeline_model.spectrum_pk,
        pipeline_model.teff,
        pipeline_model.fe_h,
        pipeline_model.e_teff,
        pipeline_model.e_fe_h,
    )
    .distinct(pipeline_model.spectrum_pk)
    .join(BossVisitSpectrum, on=(BossVisitSpectrum.spectrum_pk == pipeline_model.spectrum_pk))
    #.where(pipeline_model.result_flags == 0)
    .dicts()
)
field_names = ("teff", "fe_h")

pairwise = select_pairwise_combinations("SLAM.pkl", q, field_names, overwrite=overwrite)
corrections = compute_corrections("SLAM_corrections.pkl", pairwise, overwrite=overwrite)

