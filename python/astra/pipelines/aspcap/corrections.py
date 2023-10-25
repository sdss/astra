import numpy as np
from peewee import JOIN, fn, chunked
from tqdm import tqdm

from astra.models.source import Source
from astra.models.aspcap import ASPCAP
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from astropy.table import Table
from astra.utils import log, expand_path

def clear_corrections(batch_size: int = 500):
    """
    Clear any existing corrections.
    """
    # TODO: Have some reference columns of what fields we should use / update with raw_ prefixes?
    N_updated = 0
    for chunk in chunked(ASPCAP.select(), batch_size):
        for result in chunk:
            result.calibrated = False
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
                    ASPCAP.calibrated
                ]
            )
        )

    return N_updated


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
        #apokasc["raw_n_m_atm"],
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
#    &   (apokasc["KALLINGER_EVSTATES"] != 2) # remove secondary red clump
    &   (apokasc["APOKASC3P_MASS"] >= 0) # remove other secondary red clump things    
    &   (np.abs(apokasc["APOKASC3P_LOGG"] - apokasc["raw_logg"] + 0.2) < 0.2) # coarse outlier removal
    )
    
    X = np.array([
        apokasc["raw_logg"],
        #apokasc["raw_teff"],
        #apokasc["raw_m_h_atm"],
        #apokasc["raw_c_m_atm"],
    ]).T
    
    y = apokasc["APOKASC3P_LOGG"]
    
    mask_lm_rc = np.isfinite(X).all(axis=1) * is_rc_for_fit
    
    rc_offset = np.median(apokasc["raw_logg"][mask_lm_rc] - y[mask_lm_rc])
    #lm_rc = LinearRegression()
    #lm_rc.fit(X[mask_lm_rc], y[mask_lm_rc])
    
    '''
    dy = lm_rc.predict(X[mask_lm_rc]) - y[mask_lm_rc]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(
        #apokasc["raw_logg"][mask_lm_rc],
        #apokasc["APOKASC3P_LOGG"][mask_lm_rc] - apokasc["raw_logg"][mask_lm_rc] + 0.2,
        y[mask_lm_rc],
        dy,        
        s=1,
        #alpha=0.1,
        c=apokasc["APOKASC3P_MASS"][mask_lm_rc]
    )
    #ax.axhline(0, c="#666666", ls=":", zorder=-1, lw=0.5)
    plt.colorbar(scat)
    fig.savefig(expand_path("~/tmp.png"))    
    print(np.nanmedian(dy), np.nanstd(dy))
    '''

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
    
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scat = ax.scatter(
        y[mask_lm_rgb],
        lm_rgb.predict(X[mask_lm_rgb]) - y[mask_lm_rgb],        
        s=1,
        #c=apokasc["APOKASC3P_MASS"][mask_lm_rgb]
    )
    plt.colorbar(scat)
    fig.savefig(expand_path("~/tmp.png"))
    '''

    def logg_correction_dwarf(r):
        return r.raw_logg - (-0.947 + 1.886e-4 * r.raw_teff + 0.410 * r.raw_m_h_atm) # Eq 2.

    
    # We're ready to apply corrections for RGB, RC, and MS stars.
    q = (
        ASPCAP
        .select()
        .where(
            ASPCAP.raw_teff.is_null(False)
        &   ASPCAP.raw_logg.is_null(False)
        &   ASPCAP.raw_m_h_atm.is_null(False)
        )
        .limit(limit)
    )
    
    with tqdm(total=len(q), desc="Applying corrections") as pb:
        for chunk in chunked(q, batch_size):
            updated = []
            for r in chunk:   
                r.calibrated_flags = 0 # remove any previous classifications     
                if r.raw_logg > 4 or r.raw_teff > 6000:
                    # dwarf
                    r.flag_main_sequence = True
                    r.logg = logg_correction_dwarf(r) #
                elif r.raw_logg > -1:
                    # evolved
                    X = np.atleast_2d([
                        r.raw_teff,
                        r.raw_logg,
                        r.raw_m_h_atm,
                        r.raw_c_m_atm,
                        #r.raw_n_m_atm
                    ])            
                    predicted_class, = clf.predict(X)
                    if predicted_class == 1: # rgb
                        r.flag_red_giant_branch = True
                        r.logg, = lm_rgb.predict(np.atleast_2d([
                            r.raw_m_h_atm,
                            r.raw_logg,                            
                        ]))
                    elif predicted_class == 2: # rc
                        r.flag_red_clump = True
                        '''
                        r.logg, = lm_rc.predict(np.atleast_2d([
                            #r.raw_logg,
                            0,
                        ]))
                        '''
                        r.logg = r.raw_logg - rc_offset
                    else:
                        raise ValueError("arrrgh")
                        
                if np.isfinite(r.logg):
                    updated.append(r)

            if updated:                    
                (
                    ASPCAP
                    .bulk_update(
                        updated,
                        fields=[
                            ASPCAP.logg,
                            ASPCAP.calibrated_flags,
                        ]
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
