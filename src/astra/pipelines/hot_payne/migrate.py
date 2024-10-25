from typing import Iterable, Optional
from astra import task
from astra.utils import expand_path, log
from astra.models.boss import BossVisitSpectrum
from astra.models.hot_payne import HotPayne
from astropy.io import fits
from peewee import chunked
from tqdm import tqdm

@task
def ingest_from_file(path: Optional[str] = "$MWM_ASTRA/aux/external-catalogs/SDSSVHotPayne-combined-20231107.fits") -> Iterable[HotPayne]:
    """
    Ingest results from Hot Payne.
    """
    
    translate_keys = {
        'teff': None, 
        'e_teff': None, 
        'logg': None, 
        'e_logg': None, 
        'fe_h': None, 
        'e_fe_h': None, 
        'vmic': 'v_micro', 
        'e_vmic': 'e_v_micro', 
        'vsini': 'v_sini', 
        'e_vsini': 'e_v_sini', 
        'he_abun': 'he_h', 
        'e_he_abun': 'e_he_h', 
        'c_h': None, 
        'e_c_h': None, 
        'n_h': None, 
        'e_n_h': None, 
        'o_h': None, 
        'e_o_h': None, 
        'si_h': None, 
        'e_si_h': None, 
        's_h': None, 
        'e_s_h': None, 
        'covar': None, 
        'chisq': 'chi2', 
        
        'teff_fullspec': None, 
        'e_teff_fullspec': None, 
        'logg_fullspec': None, 
        'e_logg_fullspec': None, 
        'feh_fullspec': 'fe_h_fullspec', 
        'e_feh_fullspec': 'e_fe_h_fullspec', 
        'vmic_fullspec': 'v_micro_fullspec', 
        'e_vmic_fullspec': 'e_v_micro_fullspec', 
        'vsini_fullspec': 'v_sini_fullspec', 
        'e_vsini_fullspec': 'e_v_sini_fullspec', 
        'he_abun_fullspec': 'he_h_fullspec', 
        'e_he_abun_fullspec': 'e_he_h_fullspec', 
        'ch_fullspec': 'c_h_fullspec', 
        'e_ch_fullspec': 'e_c_h_fullspec', 
        'nh_fullspec': 'n_h_fullspec', 
        'e_nh_fullspec': 'e_n_h_fullspec', 
        'oh_fullspec': 'o_h_fullspec', 
        'e_oh_fullspec': 'e_o_h_fullspec', 
        'sih_fullspec': 'si_h_fullspec', 
        'e_sih_fullspec': 'e_si_h_fullspec', 
        'sh_fullspec': 's_h_fullspec', 
        'e_sh_fullspec': 'e_s_h_fullspec', 
        'covar_fullspec': 'covar_fullspec', 
        'chisq_fullspec': 'chi2_fullspec', 

        'teff_hmasked': None, 
        'e_teff_hmasked': None, 
        'logg_hmasked': None, 
        'e_logg_hmasked': None, 
        'feh_hmasked': 'fe_h_hmasked', 
        'e_feh_hmasked': 'e_fe_h_hmasked',
        'vmic_hmasked': 'v_micro_hmasked',
        'e_vmic_hmasked': 'e_v_micro_hmasked',
        'vsini_hmasked': 'v_sini_hmasked',
        'e_vsini_hmasked': 'e_v_sini_hmasked',
        'he_abun_hmasked': 'he_h_hmasked',
        'e_he_abun_hmasked': 'e_he_h_hmasked',
        'ch_hmasked': 'c_h_hmasked',
        'e_ch_hmasked': 'e_c_h_hmasked',
        'nh_hmasked': 'n_h_hmasked',
        'e_nh_hmasked': 'e_n_h_hmasked',
        'oh_hmasked': 'o_h_hmasked',
        'e_oh_hmasked': 'e_o_h_hmasked',
        'sih_hmasked': 'si_h_hmasked',
        'e_sih_hmasked': 'e_si_h_hmasked',
        'sh_hmasked': 's_h_hmasked',
        'e_sh_hmasked': 'e_s_h_hmasked',
        'covar_hmasked': None,
        'chisq_hmasked': 'chi2_hmasked'
    }
    
    n_errors = 0
    with fits.open(expand_path(path)) as image:
        
        data = image[1].data
            
        q = (
            BossVisitSpectrum
            .select(
                BossVisitSpectrum.spectrum_pk,
                BossVisitSpectrum.source_pk,
                BossVisitSpectrum.mjd,
                BossVisitSpectrum.fieldid,
                BossVisitSpectrum.catalogid
            )
            .where(BossVisitSpectrum.catalogid.in_(tuple(set(data["CATALOGID"]))))
        )
        lookup_pks = { (r.catalogid, r.mjd, r.fieldid): (r.spectrum_pk, r.source_pk) for r in q.iterator() }
                
        for i, row in tqdm(enumerate(image[1].data)):
            
            key = (row["CATALOGID"], row["MJD"], row["FIELD"])
            try:
                spectrum_pk, source_pk = lookup_pks[key]
            except:
                log.warning(f"No spectrum found with {key}")
                n_errors += 1
                continue
            
            result = dict([((v or k), row[k]) for k, v in translate_keys.items()])
            result.update(spectrum_pk=spectrum_pk, source_pk=source_pk)
            
            for k in result.keys():
                if k.startswith("covar"):
                    result[k] = list(map(float, result[k].flatten()))
            
            yield HotPayne(**result)
            
    if n_errors > 0:
        log.warning(f"There were {n_errors} missing spectra")            
            

        
        