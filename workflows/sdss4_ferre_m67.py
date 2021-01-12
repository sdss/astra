

import matplotlib.pyplot as plt
import numpy as np
import astropy.table
import luigi

from tqdm import tqdm

import astra
import astra.utils
from astra.tasks.io.sdss4 import ApStarFile
from astra.contrib.ferre.tasks.aspcap import (
    EstimateChemicalAbundanceGivenApStarFile, EstimateChemicalAbundancesGivenApStarFile
)

calibration_set = astropy.table.Table.read("/uufs/chpc.utah.edu/common/home/sdss50/dr16/apogee/spectro/aspcap/r12/l33/allCal-r12-l33.fits")

# Only need unique entries of these to start.
calibration_set = astropy.table.unique(calibration_set, ("TELESCOPE", "FIELD", "APOGEE_ID"))

# Only choose things marked as M67.
is_m67 = calibration_set["FIELD"] == "M67"
calibration_set = calibration_set[is_m67]

# Limit the sample size for debugging purposes?
N_sample = 5

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
    # Analysis keywords.
    interpolation_order=1,
    continuum_flag=1,
    continuum_order=4,
    continuum_reject=0.1,
    continuum_observations_flag=1,
    error_algorithm_flag=1,
    optimization_algorithm_flag=3,
    wavelength_interpolation_flag=0,
    input_weights_path="/uufs/chpc.utah.edu/common/home/u6020307/astra-component-data/FERRE/masks/global_mask_v02.txt",
    pca_project=False,
    pca_chi=False,
    #directory_kwds=dict(dir="/home/ubuntu/data/sdss/astra-components/astra_ferre/tmp/"),
    n_threads=64,
    debug=True,
    use_direct_access=False,
#    use_slurm=True,
)


from astra.contrib.ferre.tasks.aspcap import dispatch_apstars_for_analysis

#config = luigi.configuration.get_config()
#config.read("utah.cfg")


dispatch = EstimateChemicalAbundancesGivenApStarFile(
    **workflow_keywords,
    **batch_kwds
)

batch_kwds = list(dispatch.get_batch_task_kwds(False))
batch_kwds[-1].update(telescope="lco25m", prefix="as", obj="2M08532214+1112230")

raise a
moo = dispatch_apstars_for_analysis(batch_kwds, "../astra-component-data/FERRE/grid_header_paths.utah.list", release="DR16")
foo = [ea[1] for ea in moo]


batch_kwds = astra.utils.batcher(foo, unique=True)

dispatch = EstimateChemicalAbundancesGivenApStarFile(
    **workflow_keywords,
    **batch_kwds
)


import astra

astra.build(
    [dispatch],
    # Workers sets the number of simultaneous workers, but these are bound by number of active
    # Slurm jobs because we interactively wait for a Slurm job to finish.
    local_scheduler=True,
    workers=24,
#    detailed_summary=True
)