import numpy as np
from peewee import JOIN, fn, chunked
from tqdm import tqdm

from astra.models.aspcap import ASPCAP
    

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
        elif r.raw_logg < 3.5 and r.raw_teff < 6000:
            # RGB stars
            r.logg = logg_correction_rgb(r)
            r.e_logg = e_logg_correction_rgb(r)
        elif 3.5 < r.raw_logg < 4.0 and r.raw_teff < 6000:
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
