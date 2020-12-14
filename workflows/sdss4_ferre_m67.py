

import matplotlib.pyplot as plt
import numpy as np
import astropy.table
import luigi

from tqdm import tqdm

import astra
from astra.tasks.io.sdss4 import ApStarFile
from astra.contrib.ferre.tasks.aspcap import (
    EstimateChemicalAbundanceGivenApStarFile, EstimateChemicalAbundancesGivenApStarFile
)

calibration_set = astropy.table.Table.read("/home/andy/data/sdss/apogeework/apogee/spectro/aspcap/r12/l33/allCal-r12-l33.fits")

# Only need unique entries of these to start.
calibration_set = astropy.table.unique(calibration_set, ("TELESCOPE", "FIELD", "APOGEE_ID"))

# Only choose things marked as M67.
is_m67 = calibration_set["FIELD"] == "M67"
calibration_set = calibration_set[is_m67]

# Limit the sample size for debugging purposes?
N_sample = None

batch_kwds = {}
for parameter_name in ApStarFile.batch_param_names():
    batch_kwds[parameter_name] = []

for i, row in enumerate(calibration_set):
    if i == 0:
        continue
    
    kv_pair = {
        "apred": "r12",
        "apstar": "stars",
        "prefix": row["FILE"][:2],
        "telescope": row["TELESCOPE"],
        "field": row["FIELD"],
        "obj": row["APOGEE_ID"],
    }
    for k, v in kv_pair.items():
        batch_kwds[k].append(v)

    if N_sample is not None and i >= N_sample:
        break


for parameter_name in ApStarFile.batch_param_names():
    batch_kwds[parameter_name] = tuple(batch_kwds[parameter_name])


workflow_keywords = dict(
    connection_string="sqlite:////home/ubuntu/data/sdss/astra/m67.db",
    
    # Analysis keywords.
    interpolation_order=1,
    continuum_flag=1,
    continuum_order=4,
    continuum_reject=0.1,
    continuum_observations_flag=1,
    error_algorithm_flag=1,
    optimization_algorithm_flag=3,
    wavelength_interpolation_flag=0,
    input_weights_path="/home/ubuntu/data/sdss/astra-components/astra_ferre/python/astra_ferre/core/global_mask_v02.txt",
    pca_project=False,
    pca_chi=False,
    directory_kwds=dict(dir="/home/ubuntu/data/sdss/astra-components/astra_ferre/tmp/"),
    n_threads=4,
    debug=True,
    use_direct_access=True
)


dispatch = EstimateChemicalAbundancesGivenApStarFile(
    **workflow_keywords,
    **batch_kwds
)


import astra

astra.build(
    [dispatch],
    workers=1,
    local_scheduler=True,
    detailed_summary=True

)