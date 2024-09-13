
# The DR19 (v0.6.0) release should restrict BOSS spectra to things up and until 60130.
import os
import pickle
from tqdm import tqdm
from astra.models import BossVisitSpectrum, BossCombinedSpectrum, Source


# For all these spectra, we need to do the following things:

# [X] 1. Delete the affected BossVisitSpectrum objects.

# [X] 2. Delete the affected BossRestFrameVisitSpectrum objects.

boss_visit_spectrum_path = "20240830_boss_spectra.pkl"
if os.path.exists(boss_visit_spectrum_path):
    with open(boss_visit_spectrum_path, "rb") as f:
        pks = pickle.load(f)
else:
    spectra = list(
        BossVisitSpectrum
        .select()
        .where(
            (BossVisitSpectrum.run2d == 'v6_1_3')
        &   (BossVisitSpectrum.mjd > 60130)
        )
    )
    # This first instance is one I deleted by hand to test the cascading.
    pks = {"source_pk": [10345532, None, 9337795, 1379544], "spectrum_pk": [36841167, 36887682, 35276769, 36134659], "boss_visit_spectrum_pk": [5038130, 5744638, 7637413, 4180240]}
    for spectrum in spectra:
        pks["source_pk"].append(spectrum.source_pk)
        pks["spectrum_pk"].append(spectrum.spectrum_pk_id)
        pks["boss_visit_spectrum_pk"].append(spectrum.pk)

    with open(boss_visit_spectrum_path, "wb") as f:
        pickle.dump(pks, f)

    # Delete them with cascade, which impacts the BossRestFrameVisitSpectrum reference.
    for spectrum in tqdm(spectra):
        spectrum.delete_instance(recursive=True)
    
# [X] 3. Delete the affected BossCombinedSpectrum objects.
if False:
    source_pks = list(set(pks["source_pk"]).difference({None}))
    q = (
        BossCombinedSpectrum
        .delete()
        .where(
            (BossCombinedSpectrum.run2d == 'v6_1_3')
        &   (BossCombinedSpectrum.v_astra == '0.6.0')
        &   BossCombinedSpectrum.source_pk.in_(source_pks)
        )
        .execute()
    )

# [X] 4. Delete the affected mwmVisit and mwmStar files.
#       -> 0 mwmStar
#       -> 0 mwmVisit
if False:
    sdss_ids = list(
        Source
        .select(Source.sdss_id)
        .where(Source.pk.in_(source_pks))
        .tuples()
    )
    for (sdss_id, ) in tqdm(sdss_ids):

        num = (f"{sdss_id}")[-4:]
        if len(num) < 4:
            num = ("0" + (4 - len(num))) + num
        
        u, d = (num[:2], num[2:])

        mwmStar_path = f"/uufs/chpc.utah.edu/common/home/sdss51/sdsswork/mwm/spectro/astra/0.6.0/star/{u}/{d}/mwmStar-0.6.0-{sdss_id}.fits"
        mwmVisit_path = f"/uufs/chpc.utah.edu/common/home/sdss51/sdsswork/mwm/spectro/astra/0.6.0/visit/{u}/{d}/mwmVisit-0.6.0-{sdss_id}.fits"

        if os.path.exists(mwmStar_path):
            os.system(f"mv {mwmStar_path} {mwmStar_path}.for_removal")
        if os.path.exists(mwmVisit_path):
            os.system(f"mv {mwmVisit_path} {mwmVisit_path}.for_removal")

    
# [X] 5. Recreate the mwmVisit and mwmStar files for the affected sources.

# [X] 6. Go through all the pipeline tables and remove the results for the affected BOSS visit spectra.
from astra.models import (BossNet, SnowWhite, LineForest, MDwarfType, Corv)

if False:        
    boss_pipeline_models = (BossNet, SnowWhite, LineForest, MDwarfType, Corv)
    for model in boss_pipeline_models:
        n = (
            model
            .delete()
            .where(
                (model.v_astra == "0.6.0")
            &   (model.spectrum_pk.in_(pks["spectrum_pk"]))
            )
            .execute()
        )
        print(model, n)

    '''
    <Model: BossNet> 1737800
    <Model: SnowWhite> 33930
    <Model: LineForest> 365344
    <Model: MDwarfType> 0
    <Model: Corv> 15435
    '''

# [X] Update spectrum counts, min/max mjds.
if False:
    from astra.migrations.misc import update_visit_spectra_counts
    from astra.models import ApogeeVisitSpectrum, BossVisitSpectrum
    update_visit_spectra_counts(
        apogee_where=ApogeeVisitSpectrum.apred.in_(("dr17", "1.3")),
        boss_where=(BossVisitSpectrum.run2d == "v6_1_3")
    )


# 7. Re-build pipeline-level files for affected SnowWhite/Corv things.
if False:
        
    from astra.products.pipeline import create_visit_pipeline_product

    q = (
        Source
        .select()
        .distinct(Source.sdss_id)
        .join(BossVisitSpectrum, on=(BossVisitSpectrum.source_pk == Source.pk))
        .where(
            Source.assigned_to_program("mwm_wd")
        &   Source.pk.in_(pks["source_pk"])
        )
    )
    from tqdm import tqdm

    for source in tqdm(q):
        try:
            create_visit_pipeline_product(
                source,
                SnowWhite,
                overwrite=True
            )
        except:
            print("fail")
        try:
            Corv.get(source_pk=source.pk)
        except:
            None
        else:
            create_visit_pipeline_product(
                source,
                Corv,
                overwrite=True
            )

# 7. Delete any sources that don't have any APOGEE spectra or BOSS spectra. -> Just restrict mwmTargets/mwmStar etc to those with n_apogee_visits > 0 or n_boss_visits >

# 8. Construct new summary files for any affected pipelines.
