import numpy as np
import pickle
from tqdm import tqdm
from astra.utils import expand_path
#from astra.specutils.continuum.nmf import ApogeeContinuum
from astropy.table import Table
import os
from astra.utils import log
from astra.models.base import database
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
from astra.specutils.continuum.nmf.apogee import ApogeeNMFContinuum

from astra.models.nmf_rectify import NMFRectify
#from astra.models.the_cannon import TrainingSet, TrainingSetSpectrum
import warnings


def create_training_set_from_sdss4_dr17_apogee_subset(
    name,
    path,
):
    
    prefix = expand_path(f"$MWM_ASTRA/pipelines/TheCannon/{name}")
    data_path = f"{prefix}.pkl"
    labels_path = f"{prefix}.fits"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if os.path.exists(data_path) and not overwrite:
        raise OSError(f"Path exists and won't overwrite: {data_path}")    

    q = (
        ApogeeCoaddedSpectrumInApStar
        .select(
            ApogeeCoaddedSpectrumInApStar,
            NMFRectify.continuum_theta   
        )
        .join(NMFRectify, on=(NMFRectify.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
        .where(ApogeeCoaddedSpectrumInApStar.release == "dr17")
        .objects()
    )
    
    labelled_set = Table.read(path)

    model = ApogeeNMFContinuum()

    spectra = {}
    for spectrum in q:
        spectra[f"{spectrum.telescope}/{spectrum.field}/{spectrum.obj}"] = spectrum

    has_spectra = np.zeros(len(labelled_set), dtype=bool)
    for i, row in enumerate(labelled_set):
        key = "/".join([row[k.upper()].strip() for k in ("telescope", "field", "apogee_id")])
        try:
            spectrum = spectra[key]
        except:
            has_spectra[i] = False
        else:
            has_spectra[i] = True    
    
    use_labelled_set = labelled_set[has_spectra]

    pks, flux, ivar, rows, labels = ([], [], [], [], [])
    for i, row in enumerate(tqdm(use_labelled_set)):

        key = "/".join([row[k.upper()].strip() for k in ("telescope", "field", "apogee_id")])
        try:
            spectrum = spectra[key]
        except:
            log.warning(f"Could not find spectrum for row key {key}")
            continue

        continuum = model.continuum(spectrum.wavelength, spectrum.continuum_theta)[0]
        
        pks.append((spectrum.source_pk, spectrum.spectrum_pk))
        flux.append(spectrum.flux / continuum)
        ivar.append(spectrum.ivar * continuum**2)
        rows.append(row)    
    
    rows_table = Table(rows=rows, names=rows[0].dtype.names)
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
    training_set = dict(
        flux=flux,
        ivar=ivar,
        pks=pks
    )

    with open(data_path, "wb") as fp:
        pickle.dump(training_set, fp)
        
    rows_table.write(labels_path, overwrite=overwrite)    



def create_training_set_for_apogee_coadded_spectra_from_sdss4_dr17_apogee(
    name,
    mask_callable,
    label_names=None,
    overwrite=False,
    batch_size=1000,
) -> None:

    prefix = expand_path(f"$MWM_ASTRA/pipelines/TheCannon/{name}")
    data_path = f"{prefix}.pkl"
    labels_path = f"{prefix}.fits"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if os.path.exists(data_path) and not overwrite:
        raise OSError(f"Path exists and won't overwrite: {data_path}")
    
    dr17_allStar = Table.read(expand_path("$SAS_BASE_DIR/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits"))

    q = (
        ApogeeCoaddedSpectrumInApStar
        .select(
            ApogeeCoaddedSpectrumInApStar,
            NMFRectify.continuum_theta   
        )
        .join(NMFRectify, on=(NMFRectify.spectrum_pk == ApogeeCoaddedSpectrumInApStar.spectrum_pk))
        .where(ApogeeCoaddedSpectrumInApStar.release == "dr17")
        .objects()
    )

    model = ApogeeNMFContinuum()

    spectra = {}
    for spectrum in q:
        spectra[f"{spectrum.telescope}/{spectrum.field}/{spectrum.obj}"] = spectrum

    has_spectra = np.zeros(len(dr17_allStar), dtype=bool)
    for i, row in enumerate(dr17_allStar):
        key = "/".join([row[k.upper()].strip() for k in ("telescope", "field", "apogee_id")])
        try:
            spectrum = spectra[key]
        except:
            has_spectra[i] = False
        else:
            has_spectra[i] = True    

    dr17_rows = mask_callable(dr17_allStar[has_spectra])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        pks, flux, ivar, rows, labels = ([], [], [], [], [])
        for i, row in enumerate(tqdm(dr17_rows)):

            key = "/".join([row[k.upper()].strip() for k in ("telescope", "field", "apogee_id")])
            try:
                spectrum = spectra[key]
            except:
                log.warning(f"Could not find spectrum for row key {key}")
                continue

            # TODO: make this faster by getting the join with the original spectra
            '''
            try:
                result = NMFRectify.get(spectrum_pk=spectrum.spectrum_pk)
            except:
                args = tuple(map(np.atleast_2d, (spectrum.flux, spectrum.ivar)))
                x0 = model.get_initial_guess_with_small_W(*args)
                try:
                    continuum, result = model.fit(*args, x0=x0, full_output=True)          
                except:
                    log.warning(f"Could not fit continuum for row key {key}")
                    continue
            else:
                continuum = model.continuum(spectrum.wavelength, result.continuum_theta)[0]
            '''
            
            continuum = model.continuum(spectrum.wavelength, spectrum.continuum_theta)[0]
            
            pks.append((spectrum.source_pk, spectrum.spectrum_pk))
            flux.append(spectrum.flux / continuum)
            ivar.append(spectrum.ivar * continuum**2)
            rows.append(row)

    rows_table = Table(rows=rows, names=rows[0].dtype.names)
    flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
    if label_names is None:
        labels = None
    else:
        labels = np.array([rows_table[ln] for ln in label_names])
    training_set = dict(
        flux=flux,
        ivar=ivar,
        labels=labels,
        pks=pks
    )

    with open(data_path, "wb") as fp:
        pickle.dump(training_set, fp)
        
    rows_table.write(labels_path, overwrite=overwrite)
    
    '''
    raise a

    ts = TrainingSet.create(
        name=name,
        description="APOGEE DR17 co-added spectra using DR17 ASPCAP labels",
        n_labels=len(label_names),
        n_spectra=len(flux)
    )

    TrainingSetSpectrum.create_table()

    with database.atomic():
        q = (
            TrainingSetSpectrum
            .insert_many(
                [
                dict(training_set_pk=1, source_pk=source_pk, spectrum_pk=spectrum_pk) \
                    for source_pk, spectrum_pk in pks
                ],
            )
            .execute()
        )

    # Create a flat file from the rows used
    reference_table = Table(rows=rows_used, names=rows_used[0].dtype.names)
    reference_table.write(expand_path(f"$MWM_ASTRA/pipelines/TheCannon/{name}.fits"))
    '''
    return (data_path, labels_path)

    #return (ts, path)

    



if __name__ == "__main__":


    label_names = (
        "TEFF",
        "LOGG",
        "FE_H",
        "VMICRO",
        "VMACRO",
        #"VSINI",
        "C_FE",
        "N_FE",
        "O_FE",
        "NA_FE",
        "MG_FE",
        "AL_FE",
        "SI_FE",
        #"P_FE",
        "S_FE",
        "K_FE",
        "CA_FE",
        "TI_FE",
        "V_FE",
        "CR_FE",
        "MN_FE",
        #"CO_FE",
        "NI_FE",
        #"CU_FE",
        #"CE_FE",
        #"YB_FE"
    )
        
    def mask_callable(dr17_allstar, seed=17, size=5_000):
        np.random.seed(seed)
        meets_qc = (
            (dr17_allstar["SNR"] > 200)
        &   (dr17_allstar["ASPCAPFLAG"] == 0)
        &   (dr17_allstar["TEFF"] <= 8000)
        )
        for ln in label_names:
            new_mask = np.isfinite(dr17_allstar[ln])
            print(f"{ln} from {np.sum(meets_qc)} -> {np.sum(meets_qc * new_mask)}")
            meets_qc *= new_mask

        N = np.sum(meets_qc)
        print(f"{N} rows meet constraints")

        indices = np.random.choice(np.where(meets_qc)[0], size=min(size, N), replace=False)
        return dr17_allstar[indices]
    

    create_training_set_for_apogee_coadded_spectra_from_sdss4_dr17_apogee(
        "alpha-0",
        label_names, 
        mask_callable=mask_callable
    )