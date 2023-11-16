import numpy as np
print("A")
from peewee import JOIN, fn, chunked
print("A")
from tqdm import tqdm
print("A")

from astra.models import ApogeeCoaddedSpectrumInApStar
from astra.models.source import Source
print("A")
from astra.models.aspcap import ASPCAP
print("A")
from sklearn.metrics import confusion_matrix
print("A")
from sklearn.ensemble import RandomForestClassifier
print("A")
from sklearn.linear_model import LinearRegression
print("A")
from astropy.table import Table
print("A")
from astra.utils import log, expand_path
print("A")

import astropy.units as u
import astropy.constants as c

def clear_corrections(batch_size: int = 500):
    """
    Clear any existing corrections.
    """
    # TODO: Have some reference columns of what fields we should use / update with raw_ prefixes?
    N_updated = 0
    for chunk in chunked(ASPCAP.select(), batch_size):
        for result in chunk:
            result.teff = result.raw_teff
            result.e_teff = result.raw_e_teff
            result.logg = result.raw_logg
            result.e_logg = result.raw_e_logg

        N_updated += (
            ASPCAP
            .bulk_update(
                chunk,
                fields=[
                    ASPCAP.teff,
                    ASPCAP.e_teff,
                    ASPCAP.logg,
                    ASPCAP.e_logg,
                ]
            )
        )

    return N_updated

import numpy as np
import matplotlib.pyplot as plt



def apply_flags():
    
    (
        ASPCAP
        .update(
            result_flags=ASPCAP.flag_high_v_sini.set()
        )
        .where(
            (ASPCAP.raw_teff < 5250)
        &   (ASPCAP.raw_v_sini > 3)
        &   (ASPCAP.raw_v_sini.is_null(False))
        &   (ASPCAP.raw_v_sini != 'NaN')
        )
        .execute()
    )
    (
        ASPCAP
        .update(result_flags=ASPCAP.flag_high_v_micro.set())
        .where(
            (ASPCAP.raw_logg <= 3)
        &   (ASPCAP.raw_v_micro > 3)
        )
        .execute()
    )
    (
        ASPCAP
        .update(
            result_flags=ASPCAP.flag_ferre_fail.set(),
        )
        .where(
            (ASPCAP.raw_teff <= 3000)
        |   (ASPCAP.raw_logg < -1)
        )
        .execute()
    )
    (
        ASPCAP
        .update(
            result_flags=ASPCAP.flag_low_snr.set()
        )
        .where(
            (ASPCAP.ferre_log_snr_sq < 20)
        )
        .execute()
    )
    (
        ASPCAP
        .update(
            result_flags=ASPCAP.flag_high_rchi2.set()
        )
        .where(ASPCAP.rchi2 > 1000)
        .execute()
    )
    q = (
        ASPCAP
        .select(
            ASPCAP,
            ApogeeCoaddedSpectrumInApStar.std_v_rad)
        .join(ApogeeCoaddedSpectrumInApStar, on=(ASPCAP.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
        .objects()
        
    )
    with tqdm() as pb:            
        for chunk in chunked(q, 1000):
            updated = []
            for item in chunk:
                if item.std_v_rad > 1:
                    item.flag_high_std_v_rad = True
                    updated.append(item)
                
            if len(updated) > 0:
                (
                    ASPCAP
                    .bulk_update(
                        updated,
                        fields=[
                            ASPCAP.result_flags
                        ]
                    )
                )
                pb.update(1000)
    
    '''
    (
        ASPCAP
        .update(
            result_flags=ASPCAP.flag_high_std_v_rad.set()
        )
        .where(
            (ASPCAP.std_v_rad > 1)
        )
    )
    '''

    
def apply_ipl3_irfm_corrections():
        
    (
        ASPCAP
        .update(
            teff=ASPCAP.raw_teff - (1.32607360e-04 * fn.pow(ASPCAP.raw_teff, 2) - 1.34550981e+00 * ASPCAP.raw_teff + 172.58758444 * ASPCAP.m_h_atm + 3362.5502957592807)
        )
        .where(
            (ASPCAP.raw_logg >= 3.8)
        &   (7000 >= ASPCAP.raw_teff) & (ASPCAP.raw_teff >= 4500)
        )
        .execute()
    )
    (
        ASPCAP
        .update(teff=ASPCAP.raw_teff)# #- 133.08899593327305 + 172.58758444 * ASPCAP.raw_m_h_atm + 49.75657881)
        .where(
            (ASPCAP.raw_logg >= 3.8)
        &   (ASPCAP.raw_teff > 7000)
        )   
        .execute()
    )
    (
        ASPCAP
        .update(teff=ASPCAP.raw_teff -6.944800552354991 + 172.58758444 * ASPCAP.raw_m_h_atm + 49.75657881)
        .where(
            (ASPCAP.raw_logg >= 3.8)
        &   (ASPCAP.raw_teff < 4500)
        )
        .execute()
    )

    (
        ASPCAP
        .update(
            teff=ASPCAP.raw_teff - (1.50192884e-04 * fn.pow(ASPCAP.raw_teff,2) - 1.34175322e+00 * ASPCAP.raw_teff + 89.91479362 * ASPCAP.m_h_atm + (2962.5628921867774 + 20.904245812524106))
        )
        .where(
            (ASPCAP.raw_logg < 3.8)
        &   (5500 >= ASPCAP.raw_teff) & (ASPCAP.raw_teff >= 3000)
        )
        .execute()
    )
    (
        ASPCAP
        .update(teff=ASPCAP.raw_teff - 126.25491258531883)
        .where(
            (ASPCAP.raw_logg < 3.8)
        &   (ASPCAP.raw_teff > 5500)
        )
        .execute()
    )
    '''
    (
        ASPCAP
        .update(teff=ASPCAP.raw_teff - 106.28944170213481)
        .where(
            (ASPCAP.raw_logg < 3.8)
        &   (ASPCAP.raw_teff < 3500)
        )
        .execute()
    )
    '''

# V - J: 2.840 −1.3453 0.3906 −0.0546 0.002913


def get_m_dwarf_callable():
    
    bp_rp_mann = np.array([3.245, -2.4309, 1.043, -0.2127, 0.01649])
    bp_rp_mann_met = np.array([2.835, -1.893, 0.7860, -0.1594, 0.01243, 0.04417])

    from peewee import fn
    from astra.models import Source, ASPCAP


    q = (
        ASPCAP
        .select(
            ASPCAP,
            Source
        )
        .join(Source)
        .where(
            ((Source.bp_mag - Source.rp_mag) > 0.983)
        &   (Source.plx.is_null(False))
        &   (Source.plx > 10)
        &   ((Source.g_mag + 5 * fn.log10(Source.plx/100)) > 5.553)
        &   (ASPCAP.v_sini != 'NaN') & (ASPCAP.v_sini >= 0)
        &   ~ASPCAP.flag_bad
        )
        .dicts()
    )
    #mask = (df.BP_MAG - df.RP_MAG>0.983)&(df.G_MAG+5*np.log10(df.PLX/100)>5.553)&(~df.BADFLAG)&(df.PLX>10)
    from astropy.table import Table
    

    results = Table(rows=list(q))
    bp_rp = results["bp_mag"] - results["rp_mag"]

    in_fit = (
        (results["raw_teff"] < 4100)
    &   ((results["bp_mag"] - results["rp_mag"]) > 1.5)
    &   ((results["bp_mag"] - results["rp_mag"]) < 3.5)
    ) * np.isfinite(bp_rp)
    from scipy.stats import binned_statistic

    d_teff = results["raw_teff"] - 3500 * np.polyval(bp_rp_mann[::-1], bp_rp)

    meds, edges, binmask = binned_statistic(
        bp_rp[in_fit],
        d_teff[in_fit],
        statistic="median", 
        bins=20
    )
    delta_teff = meds[np.searchsorted(edges, bp_rp[in_fit]) - 1]

    def get_teff_logg(bp_rp, k_mag, plx):
        
        delta_teff = (
            meds[np.clip(np.searchsorted(edges, bp_rp) - 1, 0, edges.size - 2)]
        )

        mann_mass = np.array([-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4])
        mann_radius = np.array([1.9515, -0.3520, 0.01680])

        #M_Ks = (df_.K_MAG + 5*np.log10(df_.PLX/100))
        M_Ks = float(k_mag) + 5 * np.log10(float(plx)/100)

        z = np.array([bp_rp]).flatten()
        log10mass = 0
        for i in range(len(mann_mass)):
            log10mass += mann_mass[i]*(M_Ks-7.5)**i
            
        radius = 0
        for i in range(len(mann_radius)):
            radius += mann_radius[i]*M_Ks**i

        logg = np.log10(((c.G*10**log10mass*u.solMass/(radius*u.solRad)**2).to(u.cm/u.s**2)).value)

        return (delta_teff, logg, log10mass, radius)

    return get_teff_logg


def apply_m_dwarf_logg_correction():
    theta = np.array([-8.59880910e-08,  3.54159094e-05,  1.48755241e+00])
        
    (
        ASPCAP
        .update(
            logg=ASPCAP.raw_logg + 1.48755241e+00 + 3.54159094e-05 * ASPCAP.raw_teff - 8.59880910e-08 * fn.pow(ASPCAP.raw_teff, 2)
        )
        .where(
            (ASPCAP.raw_logg >= 3.8)
        &   (ASPCAP.raw_teff < 4100)
        )
        .execute()
    )
    

def apply_solar_neighbourhood_abundance_corrections():
    
    where = (
            (ASPCAP.snr > 50)
        &   (~ASPCAP.flag_bad)
        &   (0.05 > ASPCAP.m_h_atm)
        &   (ASPCAP.m_h_atm > -0.05)  
        &   (Source.plx > 2)      
    )
    
    q_giants = Table(rows=list(
            ASPCAP
            .select()
            .join(Source, on=(ASPCAP.source_pk == Source.pk))
            .switch(ASPCAP)
            .join(ApogeeCoaddedSpectrumInApStar, on=(ASPCAP.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
            .where(
                where & (ASPCAP.logg < 3.8)
            )
            .dicts()
        )
    )
    
    q_dwarfs = Table(rows=list(
            ASPCAP
            .select()
            .join(Source, on=(ASPCAP.source_pk == Source.pk))
            .switch(ASPCAP)
            .join(ApogeeCoaddedSpectrumInApStar, on=(ASPCAP.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
            .where(
                where & (ASPCAP.logg >= 3.8)
            )
            .dicts()
        )    
    )
    print(len(q_giants), len(q_dwarfs))
    field_names = (
        "raw_m_h_atm",
        "raw_alpha_m_atm",
        "raw_al_h",
        "raw_ca_h",
        "raw_ce_h",
        "raw_c_1_h",
        "raw_c_h",
        "raw_co_h",
        "raw_cr_h",
        "raw_cu_h",
        "raw_fe_h",
        "raw_k_h",
        "raw_mg_h",
        "raw_mn_h",
        "raw_na_h",
        "raw_nd_h",
        "raw_ni_h",
        "raw_n_h",
        "raw_o_h",
        "raw_p_h",
        "raw_si_h",
        "raw_s_h",
        "raw_ti_h",
        "raw_ti_2_h",
        "raw_v_h",
    )
    giant_kwds = dict()
    dwarf_kwds = dict()    
    giant_vals = {}
    dwarf_vals = {}
    for field_name in field_names:
        giants = np.array(q_giants[field_name]).astype(float)
        dwarfs = np.array(q_dwarfs[field_name]).astype(float)
        giants = giants[giants > -500]        
        dwarfs = dwarfs[dwarfs > -500]
        print(f"{field_name} {np.nanmean(giants):.2f} {np.nanmean(dwarfs):.2f}")
    
        giant_vals[field_name] = np.nanmean(giants)
        dwarf_vals[field_name] = np.nanmean(dwarfs)
        
        giant_kwds[field_name[4:]] = getattr(ASPCAP, field_name) - giant_vals[field_name]
        dwarf_kwds[field_name[4:]] = getattr(ASPCAP, field_name) - dwarf_vals[field_name]
    
    import pickle
    from astra.utils import expand_path
    from astra import __version__
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/snc_corrections.pkl"), "wb") as fp:
        pickle.dump(dict(giants=giant_vals, dwarfs=dwarf_vals), fp)
        
    (
        ASPCAP
        .update(**giant_kwds)
        .where(ASPCAP.raw_logg < 3.8)
        .execute()
    )
    (
        ASPCAP
        .update(**dwarf_kwds)
        .where(ASPCAP.raw_logg >= 3.8)
        .execute()
    )


def apply_ipl3_logg_corrections(batch_size: int = 500, limit: int = None):
    """
    Apply the IPL-3 logg corrections to the ASPCAP data model.
    """

    # First let's construct a classifier for the RC/RGB stars.
    apokasc = Table.read(expand_path("$MWM_ASTRA/aux/external-catalogs/APOKASC_cat_v7.0.5.fits"))
    
    q = (
        ASPCAP
        .select(
            ASPCAP,
            Source.sdss4_apogee_id
        )
        .distinct(Source)
        .join(Source)
        .where(Source.sdss4_apogee_id.in_(list(apokasc["2MASS_ID"])))
        .objects()
    )
    
    fields = (
        ASPCAP.raw_teff,
        ASPCAP.raw_logg,
        ASPCAP.raw_m_h_atm,
        ASPCAP.raw_c_m_atm,
        ASPCAP.raw_n_m_atm,
        ASPCAP.raw_v_micro,
    )
    
    for field in fields:
        apokasc[field.name] = np.nan * np.ones(len(apokasc))
    
    for row in q:
        mask = (apokasc["2MASS_ID"] == row.sdss4_apogee_id)
        for field in fields:
            apokasc[field.name][mask] = getattr(row, field.name)
        
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

    X = np.array([
        apokasc["raw_teff"],
        apokasc["raw_logg"],
        apokasc["raw_m_h_atm"],
        apokasc["raw_c_m_atm"],
    ]).T
    y = np.array(apokasc["APOKASC3_EVSTATES"])
    
    finite = np.isfinite(X).all(axis=1)
    is_rc_or_rgb = (y == 1) | (y == 2)
    
    mask_clf = finite * is_rc_or_rgb
    
    clf.fit(X[mask_clf], y[mask_clf])
    
    cm = confusion_matrix(
        y[mask_clf],
        clf.predict(X[mask_clf]),
    )
    log.info(f"Confusion matrix:\n{cm}")
        
    norm_cm = confusion_matrix(
        y[mask_clf],
        clf.predict(X[mask_clf]),
        normalize='true'
    )
    log.info(f"Normalized confusion matrix:\n{norm_cm}")
    assert np.min(np.diag(cm)) > 0.95, "You can do better than that..."

    # For all the RC stars, construct a corrector.
    is_rc_for_fit = (
        (apokasc["APOKASC3_EVSTATES"] == 2)
    &   (apokasc["APOKASC3P_LOGG"] > 2.3)    
    &   (apokasc["APOKASC3P_LOGG"] < 2.5) # remove secondary red clump
    &   (apokasc["APOKASC3P_MASS"] >= 0) # remove other secondary red clump things    
    &   (np.abs(apokasc["APOKASC3P_LOGG"] - apokasc["raw_logg"] + 0.2) < 0.2) # coarse outlier removal
#    &   (apokasc["KALLINGER_EVSTATES"] != 2) # remove secondary red clump
    )
    
    X = np.array([
        apokasc["raw_logg"],
    ]).T
    
    y = apokasc["APOKASC3P_LOGG"]
    
    mask_lm_rc = np.isfinite(X).all(axis=1) * is_rc_for_fit
    
    rc_offset = np.median(apokasc["raw_logg"][mask_lm_rc] - y[mask_lm_rc])


    # For all the RGB stars, construct a corrector.
    is_rgb_for_fit = (
        (apokasc["APOKASC3_EVSTATES"] == 1)  
    &   (apokasc["APOKASC3P_LOGG"] > -10)
    &   (apokasc["APOKASC3P_LOGG"] < 3.25)
    &   (apokasc["raw_logg"] < 3.25)
    &   (
            (apokasc["APOKASC3P_MASS"] > 0)
        |   (apokasc["APOKASC3P_LOGG"] < 0.85)
        )
    )
    
    X = np.array([
        apokasc["raw_m_h_atm"],
        apokasc["raw_logg"],
        #apokasc["raw_logg"]**2,
    ]).T
    y = apokasc["APOKASC3P_LOGG"]
    
    mask_lm_rgb = np.isfinite(X).all(axis=1) * is_rgb_for_fit
    
    lm_rgb = LinearRegression()
    lm_rgb.fit(X[mask_lm_rgb], y[mask_lm_rgb])
    
    def logg_correction_dwarf(r):
        return r.raw_logg - (-0.947 + 1.886e-4 * r.raw_teff + 0.410 * r.raw_m_h_atm) # Eq 2.

    print("getting mdwarf callable")
    teff_logg_m_dwarfs = get_m_dwarf_callable()
    print("ok got it")

    # We're ready to apply corrections for RGB, RC, and MS stars.
    q = (
        ASPCAP
        .select(
            ASPCAP,
            Source.bp_mag,
            Source.rp_mag,
            Source.k_mag,
            Source.plx
        )
        .join(Source)
        .where(
            ASPCAP.raw_teff.is_null(False)
        &   ASPCAP.raw_logg.is_null(False)
        &   ASPCAP.raw_m_h_atm.is_null(False)
        )
        .objects()
        .limit(limit)
    )
    
    # this is the isochrone logg from Szabolcs
    #def isochrone_main_sequence_logg_delta(raw_teff):
    #    a, b, c = (-1.495465498565102e-07, 0.0016514861990204446, -4.613929004721487)
    #    return a * raw_teff**2 + b*raw_teff + c
    #results["bp_mag"] - results["rp_mag"])
    
    with tqdm(total=0, desc="Applying corrections") as pb:
        for chunk in chunked(q, batch_size):
            any_mass_radius = False
            updated = []
            for r in chunk:   
                r.calibrated_flags = 0 # remove any previous classifications     
                r.logg = r.raw_logg
                
                #if r.raw_logg >= 3.8 and r.raw_teff < 4800 and r.bp_mag is not None and r.rp_mag is not None and 5.5 > (r.bp_mag - r.rp_mag) and (r.bp_mag - r.rp_mag) > 1.5:
                #    # dwarf
                #    r.logg = r.raw_logg - isochrone_main_sequence_logg_delta(r.raw_teff)
                #    r.flag_as_dwarf_for_calibration = True
                if r.raw_logg >= 3.8:
                
                    if r.raw_teff > 4100:
                        r.logg = logg_correction_dwarf(r)
                        r.flag_as_dwarf_for_calibration = True

                    else:                                            
                        # M-dwarf
                        try:                        
                            bp_rp = r.bp_mag - r.rp_mag
                        except:
                            r.logg = logg_correction_dwarf(r)
                            r.flag_as_dwarf_for_calibration = True                            
                        else:
                            if (3.5 > bp_rp > 1.5) and np.all(np.isfinite(np.array([r.k_mag, r.plx]).astype(float))) and r.plx > 0:
                                r.flag_as_m_dwarf_for_calibration = True                        
                            else:
                                r.logg = logg_correction_dwarf(r)
                                r.flag_as_dwarf_for_calibration = True                            
                                                
                elif r.raw_logg < 3.8:
                    # evolved                            
                    X = np.atleast_2d([
                        r.raw_teff,
                        r.raw_logg,
                        r.raw_m_h_atm,
                        r.raw_c_m_atm,
                    ]).astype(float)
                    if np.isfinite(X).all():
                        predicted_class, = clf.predict(X)
                        if predicted_class == 1: # rgb
                            r.flag_as_giant_for_calibration = True
                            r.logg, = lm_rgb.predict(np.atleast_2d([
                                r.raw_m_h_atm,
                                r.raw_logg,                            
                            ]))
                        elif predicted_class == 2: # rc
                            r.flag_as_red_clump_for_calibration = True
                            r.logg = r.raw_logg - rc_offset
                        else:
                            raise ValueError("arrrgh")
                    else:
                        r.logg = r.raw_logg
                else:
                    r.logg = r.raw_logg
                    
                    
                if np.isfinite(r.logg):
                    updated.append(r)

            if updated:      
                fields = [ASPCAP.logg, ASPCAP.calibrated_flags]              
                if any_mass_radius:
                    fields.extend([
                        ASPCAP.teff,
                    ])
                #        ASPCAP.mass,
                #        ASPCAP.radius
                
                (
                    ASPCAP
                    .bulk_update(
                        updated,
                        fields=fields
                    )
                )
            pb.update(batch_size)
            
    return None    


def apply_dr16_parameter_corrections(batch_size: int = 500):
    """
    Apply the DR16 corrections from arXiv:2007.05537 to the ASPCAP table.
    """

    q = ASPCAP.select()

    def logg_correction_dwarf(r):
        return r.raw_logg - (-0.947 + 1.886e-4 * r.raw_teff + 0.410 * r.raw_m_h_atm) # Eq 2.
    
    def logg_correction_rgb(r):
        logg_prime = np.clip(r.raw_logg, 1.2795, None)
        m_h_atm_prime = np.clip(r.raw_m_h_atm, -2.5, 0.5)
        return r.raw_logg - (-0.441 + 0.7588 * logg_prime - 0.2667 * logg_prime**2 + 0.02819 * logg_prime**3 + 0.1346 * m_h_atm_prime) # Eq (4)

    def logg_correction_red_clump(r):
        return r.raw_logg - (-4.532 + 3.222 * r.raw_logg - 0.528 * r.raw_logg**2)

    def teff_correction(r):
        m_h_atm_prime = np.clip(r.raw_m_h_atm, -2.5, 0.75)
        teff_prime = np.clip(r.raw_teff, 4500, 7000)
        return r.raw_teff + 610.81 - 4.275 * m_h_atm_prime - 0.116 * teff_prime # Eq (1)

    def e_stellar_param_correction(r, theta):
        X = np.array([1, (r.raw_teff - 4500), np.clip(r.snr - 100, 100, None), r.raw_m_h_atm])
        return np.exp(np.atleast_2d(theta) @ X)

    def e_teff_correction(r):
        return e_stellar_param_correction(r, [4.583, 2.965e-4, -2.177e-3, -0.117])#, 98])

    def e_logg_correction_dwarf(r):
        return e_stellar_param_correction(r, [-2.327, -1.349e-4, 2.269e-4, -0.306])#, 0.10])
        
    def e_logg_correction_red_clump(r):
        return e_stellar_param_correction(r, [-3.444, 9.584e-4, -5.617e-4, -0.181])#, 0.03])
    
    def e_logg_correction_rgb(r):
        return e_stellar_param_correction(r, [-2.923, 2.296e-4, 6.900e-4, -0.277])#, 0.05])

    updated = []
    for r in tqdm(q, desc="Applying corrections"):

        r.teff = teff_correction(r)
        r.e_teff = e_teff_correction(r)
        dT = r.raw_teff - (4400 + 552.6 * (r.raw_logg - 2.5) - 324.6 * r.raw_m_h_atm)

        if r.raw_logg > 4 or r.raw_teff > 6000:
            # dwarf
            r.logg = logg_correction_dwarf(r) #
            r.e_logg = e_logg_correction_dwarf(r)
        elif 2.38 < r.raw_logg < 3.5 and ((r.raw_c_m_atm - r.raw_n_m_atm) > 0.04 - 0.46 * r.raw_m_h_atm - 0.0028 * dT):
            # red clump stars
            r.logg = logg_correction_red_clump(r) # Eq (3)
            r.e_logg = e_logg_correction_red_clump(r)
        elif r.raw_logg <= 3.5 and r.raw_teff <= 6000:
            # RGB stars
            r.logg = logg_correction_rgb(r)
            r.e_logg = e_logg_correction_rgb(r)
        elif 3.5 < r.raw_logg <= 4.0 and r.raw_teff <= 6000:
            # weighted correction
            w = (4 - r.raw_logg)/(4 - 3.5)
            # • for stars with uncalibrated 3.5 <log g< 4.0 and Teff< 6000 K, a correction is determined using both the RGB and dwarf corrections, and a weighted correction is adopted based on log g.
            r.logg = (w * logg_correction_dwarf(r) + (1 - w) * logg_correction_rgb(r))
            r.e_logg = (w * e_logg_correction_dwarf(r) + (1 - w) * e_logg_correction_rgb(r))
        else:
            raise ValueError("argh")

        # TODO: uncertainties in abundances.

        r.calibrated = True
        updated.append(r)
    
    N_updated = 0
    with tqdm(total=len(updated), desc="Saving") as pb:
        for chunk in chunked(updated, batch_size):
            N_updated += (
                ASPCAP
                .bulk_update(
                    chunk,
                    fields=[
                        ASPCAP.teff,
                        ASPCAP.logg,
                        ASPCAP.e_teff,
                        ASPCAP.e_logg,
                        ASPCAP.calibrated
                    ]
                )
            )
            pb.update(N_updated)
    
    return N_updated
