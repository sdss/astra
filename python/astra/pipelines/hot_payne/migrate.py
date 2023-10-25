import numpy as np
from astropy.io import fits
from astra.models.source import Source

from astra.utils import expand_path
from astra.models.boss import BossVisitSpectrum
from astra.models.hot_payne import HotPayne
from astropy.table import Table
from tqdm import tqdm
from peewee import chunked


def ingest_from_file(batch_size: int = 1000):
    """
    Ingest results from Hot Payne.
    """
    
    image = fits.open(expand_path("$MWM_ASTRA/aux/external-catalogs/SDSSVHotPayne.fits"))
    
    q = (
        Source
        .select(
            Source.pk,
            Source.catalogid,
            Source.catalogid21,
            Source.catalogid25,
            Source.catalogid31
        )
        .tuples()
    )
    
    lookup_source_pk_by_catalogid = {}
    for pk, *catalogids in q.iterator():
        for catalogid in catalogids:
            if catalogid is not None:
                lookup_source_pk_by_catalogid[catalogid] = pk
    q = (
        BossVisitSpectrum
        .select(
            BossVisitSpectrum.spectrum_pk,
            BossVisitSpectrum.mjd,
            BossVisitSpectrum.fieldid,
            BossVisitSpectrum.catalogid
        )
    )
    lookup_spectrum_pk_by_key = { (r.catalogid, r.mjd, r.fieldid): r.spectrum_pk for r in q.iterator() }
            
    run2d = "v6_1_1"
    path = f"$BOSS_SPECTRO_REDUX/{run2d}/spAll-{run2d}.fits.gz"

    spAll = Table.read(expand_path(path))
            
    result_data = []
    for i, row in enumerate(tqdm(image[1].data)):
        
        spAll_match = (
            (spAll["CATALOGID"] == row["CATALOGID"])
        |   (spAll["CATALOGID_V0"] == row["CATALOGID"])
        |   (spAll["CATALOGID_V0P5"] == row["CATALOGID"])
        )
        if np.sum(spAll_match) > 1:
            d = np.abs(spAll["Z"][spAll_match] - row["Z"])
            d_idx = np.argmin(d)
            index = np.where(spAll_match)[0][d_idx]        
        else:
            try:
                index = np.where(spAll_match)[0][0]
            except:
                print(f"Failed to match row index {i}")
                continue
        
        # Get spectrum
        key = (spAll["CATALOGID"][index], spAll["MJD"][index], spAll["FIELD"][index])
        
        result_row = {
            'spectrum_pk': lookup_spectrum_pk_by_key[key],
            'source_pk': lookup_source_pk_by_catalogid[row['CATALOGID']],
            'z': row['Z'],
            'z_err': row['Z_ERR'],
            'teff': row['TEFF'],
            'e_teff': row['E_TEFF'],
            'logg': row['LOGG'],
            'e_logg': row['E_LOGG'],
            'fe_h': row['FE_H'],
            'e_fe_h': row['E_FE_H'],
            'v_micro': row['VMIC'],
            'e_v_micro': row['E_VMIC'],
            'v_sini': row['VSINI'],
            'e_v_sini': row['E_VSINI'],
            'he_h': row['HE_ABUN'],
            'e_he_h': row['E_HE_ABUN'],
            'c_fe': row['C_FE'],
            'e_c_fe': row['E_C_FE'],
            'n_fe': row['N_FE'],
            'e_n_fe': row['E_N_FE'],
            'o_fe': row['O_FE'],
            'e_o_fe': row['E_O_FE'],
            'si_fe': row['SI_FE'],
            'e_si_fe': row['E_SI_FE'],
            's_fe': row['S_FE'],
            'e_s_fe': row['E_S_FE'],
            'covar': row['COVAR'],
            'chi2': row['CHISQ'],
        }
        '''
        'teff_hydrogen_mask': row['TEFF2'],
        'e_teff_hydrogen_mask': row['E_TEFF2'],
        'logg_hydrogen_mask': row['LOGG2'],
        'e_logg_hydrogen_mask': row['E_LOGG2'],
        'fe_h_hydrogen_mask': row['FE_H2'],
        'e_fe_h_hydrogen_mask': row['E_FE_H2'],
        'v_micro_hydrogen_mask': row['VMIC2'],
        'e_v_micro_hydrogen_mask': row['E_VMIC2'],
        'v_sini_hydrogen_mask': row['VSINI2'],
        'e_v_sini_hydrogen_mask': row['E_VSINI2'],
        'he_h_hydrogen_mask': row['HE_ABUN2'],
        'e_he_h_hydrogen_mask': row['E_HE_ABUN2'],
        'c_h_hydrogen_mask': row['C_H2'],
        'e_c_h_hydrogen_mask': row['E_C_H2'],
        'n_h_hydrogen_mask': row['N_H2'],
        'e_n_h_hydrogen_mask': row['E_N_H2'],
        'o_h_hydrogen_mask': row['O_H2'],
        'e_o_h_hydrogen_mask': row['E_O_H2'],
        'si_h_hydrogen_mask': row['SI_H2'],
        'e_si_h_hydrogen_mask': row['E_SI_H2'],
        's_h_hydrogen_mask': row['S_H2'],
        'e_s_h_hydrogen_mask': row['E_S_H2'],
        'covar_hydrogen_mask': row['COVAR2'],
        'chi2_hydrogen_mask': row['CHISQ2'],
        '''
        result_data.append(result_row)
        
    n_updated = 0
    for chunk in tqdm(chunked(result_data, batch_size), desc="Inserting chunks"):
        n_updated += (
            HotPayne
            .insert_many(chunk)
            .execute()
        )
        
    return n_updated
        

    
    