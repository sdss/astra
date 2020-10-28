
import astra
from astra.tasks.io import AllVisitSum
from astra.contrib.classifier.tasks.test import ClassifySourceGivenApVisitFile

from astropy.table import Table
from collections import defaultdict
from tqdm import tqdm

# Load the DR16 AllVisit file to get the parameters we need to locate ApVisit files.
# Notes: In future we will do this from a SQL query.
#        If you don't have the AllVisitSum file then you can use this to get it:
#           
#          AllVisitSum(release=release, apred=apred, use_remote=True).run()

release, apred = ("dr16", "r12")
all_visits = Table.read(AllVisitSum(release=release, apred=apred).local_path)

def get_kwds(row):
    """
    Return the ApVisit keys we need from a row in the AllVisitSum file.
    """
    return dict(
        telescope=row["TELESCOPE"],
        field=row["FIELD"].lstrip(),
        plate=row["PLATE"].lstrip(),
        mjd=str(row["MJD"]),
        prefix=row["FILE"][:2],
        fiber=str(row["FIBERID"]),
        apred=apred
    )

# We will run this in batch mode.
# (We could hard-code keys here but it's good coding practice to have a single place of truth)
apvisit_kwds = { key: [] for key in get_kwds(defaultdict(lambda: "")).keys() }

# Add all visits to the keywords.
N_max = 10 # Use this to set a maximum number of ApVisits to analyse.
for i, row in tqdm(enumerate(all_visits, start=1)):
    if N_max is not None and i >= N_max: break
    for k, v in get_kwds(row).items():
        apvisit_kwds[k].append(v)

# Keywords that are needed to train the NIR classifier.
directory = "../astra-components/astra_classifier/data"
common_kwds = dict(
    release=release,
    training_spectra_path=f"{directory}/nir/nir_clean_spectra_shfl.npy",
    training_labels_path=f"{directory}/nir/nir_clean_labels_shfl.npy",
    validation_spectra_path=f"{directory}/nir/nir_clean_spectra_valid_shfl.npy",
    validation_labels_path=f"{directory}/nir/nir_clean_labels_valid_shfl.npy",
    test_spectra_path=f"{directory}/nir/nir_clean_spectra_test_shfl.npy",
    test_labels_path=f"{directory}/nir/nir_clean_labels_test_shfl.npy",
    class_names=["FGKM", "HotStar", "SB2", "YSO"]
)

# Merge keywords and create task.
task = ClassifySourceGivenApVisitFile(**{ **common_kwds, **apvisit_kwds })

# Get Astra to build the acyclic graph and schedule tasks.
astra.build(
    [task],
    local_scheduler=True
)