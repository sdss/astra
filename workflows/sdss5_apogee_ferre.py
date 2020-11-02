

import matplotlib.pyplot as plt
import numpy as np
import astropy.table
import luigi

from tqdm import tqdm

import astra
from astra.tasks.io.sdss4 import ApStarFile
from astra.contrib.ferre.tasks.aspcap import IterativeEstimateOfStellarParametersGivenApStarFile

calibration_set = astropy.table.Table.read("/home/andy/data/sdss/apogeework/apogee/spectro/aspcap/r12/l33/allCal-r12-l33.fits")

# Only need unique entries of these to start.
calibration_set = astropy.table.unique(calibration_set, ("TELESCOPE", "FIELD", "APOGEE_ID"))

# Limit the sample size for debugging purposes
N_sample = 5

batch_kwds = {}
for parameter_name in ApStarFile.batch_param_names():
    batch_kwds[parameter_name] = []

for i, row in enumerate(calibration_set):
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


workflow_keywords = dict(
    connection_string="sqlite:////home/ubuntu/data/sdss/astra/tmp.db",
    
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
)


dispatch = IterativeEstimateOfStellarParametersGivenApStarFile(
    **workflow_keywords,
    **batch_kwds
)

astra.build(
    [dispatch],
    workers=1,
    local_scheduler=True,
    detailed_summary=True
)

raise a



# For each task, get the expected values.
_expected = []
for task in tqdm(dispatcher.requires(), desc="Getting expected values"):

    # Match task expected values based on apStar keywords.
    mask = (calibration_set["FIELD"] == task.field) \
         * (calibration_set["TELESCOPE"] == task.telescope) \
         * (calibration_set["APOGEE_ID"] == task.obj)

    assert mask.sum() > 0

    # Take the first one, which I hope is the one with the
    rows = calibration_set[mask]

    assert all(rows["NVISITS"][0] >= rows["NVISITS"][1:])

    #assert len(list(set(rows["CLASS"]))) == 1


    # From https://www.sdss.org/dr16/irspec/parameters/
    # [ 
    #   0: effective temperature: Teff, 
    #   1: surface gravity: log g, 
    #   2: microturbulence: vmicro, 
    #   3: overall metal abundance: [M/H], 
    #   4: carbon abundance: [C/M], 
    #   5: nitrogen abundance: [N/M], & 
    #   6: alpha-element abundance: [α/M], 
    #   7: vsini/vmacro
    # ]
    # 
    # So to match our param_names: TEFF, LOGG, METALS, O Mg Si S Ca Ti, N, C, LOG10VDOP, LGVSINI, log_snr_sq
    # [0, 1, 3, 6, 5, 4, 7]

    _expected.append([
        rows["FPARAM"][0][[0, 1, 3, 6, 5, 4, 7]]
    ])

expected = np.array(_expected)[:, 0, :]

N, P = expected.shape

offset = 1

actual = np.nan * np.ones((N, P + 1))
for i, task in enumerate(tqdm(dispatcher.requires(), desc="Getting actuals", total=N)):
    row = task.output()["database"].read()
    offset = 1
    for j, value in enumerate(row[offset:offset + P + 1]):
        if value is not None:
            actual[i, j] = value

# TODO: We record log10VDOP but FPARAM stores VDOP.

import matplotlib.pyplot as plt
param_names = list(task.output()["database"].read(as_dict=True).keys())[offset:offset + P]
for i, param_name in enumerate(param_names):

    fig, ax = plt.subplots()
    ax.scatter(
        expected[:, i],
        actual[:, i],
        s=5
    )
    ax.set_title(param_name)

    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(
        limits,
        limits,
        c="#666666",
        ls=":",
        zorder=-1,
        lw=0.5
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    fig.savefig(f"dr16-r12-l33-{param_name.replace(' ', '-')}.png", dpi=300)

