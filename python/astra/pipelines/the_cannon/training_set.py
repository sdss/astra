import numpy as np
import pickle
from tqdm import tqdm
from astra.utils import expand_path
from astra.specutils.continuum.nmf import ApogeeContinuum
from astropy.table import Table
import os
from astra.utils import log
from astra.models.base import database
from astra.models.apogee import ApogeeCoaddedSpectrumInApStar
from astra.models.the_cannon import TrainingSet, TrainingSetSpectrum
import warnings


def create_training_set_for_apogee_coadded_spectra_from_sdss4_dr17_apogee(
    name,
    label_names,
    mask_callable,
    continuum_model_class=ApogeeContinuum,
    overwrite=False,
    batch_size=1000,
) -> None:

    path = expand_path(f"$MWM_ASTRA/pipelines/TheCannon/{name}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and not overwrite:
        raise OSError(f"Path exists and won't overwrite: {path}")

    TrainingSet.create_table()

    if TrainingSet.select().where(TrainingSet.name == name).exists():
        raise ValueError(f"Training set with name `{name}` already exists")
    
    dr17_allStar = Table.read(expand_path("$SAS_BASE_DIR/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits"))

    q = (
        ApogeeCoaddedSpectrumInApStar
        .select()
        .where(ApogeeCoaddedSpectrumInApStar.release == "dr17")
    )

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

    rows = mask_callable(dr17_allStar[has_spectra])

    continuum_model = continuum_model_class()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        pks, labels, flux, ivar, rectified_model_flux, rows_used = ([], [], [], [], [], [])
        for i, row in enumerate(tqdm(rows)):

            key = "/".join([row[k.upper()].strip() for k in ("telescope", "field", "apogee_id")])
            try:
                spectrum = spectra[key]
            except:
                log.warning(f"Could not find spectrum for row key {key}")
                continue
            
            try:
                continuum, continuum_meta = continuum_model.fit(spectrum.flux, spectrum.ivar)
            except:
                log.exception(f"Could not fit continuum model to spectrum {spectrum}")
                continue

            pks.append((spectrum.source_pk, spectrum.spectrum_pk))
            flux.append(spectrum.flux / continuum)
            ivar.append(continuum * spectrum.ivar * continuum)
            rectified_model_flux.append(continuum_meta["rectified_model_flux"])
            labels.append([row[ln] for ln in label_names])
            rows_used.append(row)

    training_set = dict(
        labels=np.array(labels),
        flux=np.array(flux),
        ivar=np.array(ivar),
        rectified_model_flux=np.array(rectified_model_flux),
        label_names=label_names,
        pks=pks
    )

    with open(path, "wb") as fp:
        pickle.dump(training_set, fp)
    

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

    return (ts, path)

    



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