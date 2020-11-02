from astra.tasks.daily import GetDailyApStarFiles, get_visits
from astra.contrib.classifier.tasks.test import ClassifySourceGivenApVisitFile



mjd = 59146

#daily_apstar_files = GetDailyApStarFiles(mjd=mjd)
#daily_apstar_kwds = get_visits(mjd=mjd)


directory = "../astra-components/astra_classifier/data"
common_kwds = dict(
    training_spectra_path=f"{directory}/nir/nir_clean_spectra_shfl.npy",
    training_labels_path=f"{directory}/nir/nir_clean_labels_shfl.npy",
    validation_spectra_path=f"{directory}/nir/nir_clean_spectra_valid_shfl.npy",
    validation_labels_path=f"{directory}/nir/nir_clean_labels_valid_shfl.npy",
    test_spectra_path=f"{directory}/nir/nir_clean_spectra_test_shfl.npy",
    test_labels_path=f"{directory}/nir/nir_clean_labels_test_shfl.npy",
    class_names=["FGKM", "HotStar", "SB2", "YSO"]
)


# Merge keywords and create task.
tasks = []
for kwds in get_visits(mjd=mjd):
    task = ClassifySourceGivenApVisitFile(**{ **common_kwds, **kwds })
    tasks.append(task)

