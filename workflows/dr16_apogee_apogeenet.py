import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt


import astra
from astra.contrib.apogeenet.tasks import (
    TrainedAPOGEENetModel,
    EstimateStellarParametersGivenSDSS4ApStarFile
)

component_data_dir = os.path.expanduser("~/astra-component-data/APOGEENet/")

# Common keywords for all analyses.
kwds = dict(
    release="dr16",
    model_path=os.path.join(component_data_dir, "APOGEE_NET.pt"),
    # Download the file from the SDSS SAS if we don't have it.
)

# Load the sources.
sources = Table.read(os.path.join(component_data_dir, "astra-yso-test.fits"))

# Here we could create one task per source, but it is slightly
# faster for us to explicitly tell astra to run all sources in
# batch mode.
source_keys = ("apstar", "apred", "field", "prefix", "telescope", "obj")
for key in source_keys:
    kwds.setdefault(key, [])

for source in sources:
    for key in source_keys:
        kwds[key].append(source[key])

task = EstimateStellarParametersGivenSDSS4ApStarFile(**kwds)


# Build the acyclic graph and execute tasks as required.
astra.build(
    [task],
    local_scheduler=True
)


# Let's make a plot comparing the outputs to what we expected.
param_names = ("Teff", "Logg", "FeH")
N, P = shape = (len(sources), 3) # three parameters estimated.

X = np.empty(shape)
X_err = np.empty(shape)
Y = np.empty(shape)
Y_err = np.empty(shape)

for i, (source, output) in enumerate(zip(sources, task.output())):

    # I am not sure whether this is the right column to use or not...
    # TODO: Check columns against original
    X[i] = [source[f"APOGEE_Net_{pn}"] for pn in param_names]
    X_err[i] = np.nan
    

    result = output["database"].read()

    # Just return the first result for each source
    # (the one from the highest S/N spectrum)
    Y[i] = [result.teff[0], result.logg[0], result.fe_h[0]]
    Y_err[i] = [result.u_teff[0], result.u_logg[0], result.u_fe_h[0]]


# Plot the results.
label_names = ("teff", "logg", "fe_h")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):

    residual = X[:, i] - Y[:, i]

    ax.errorbar(
        X[:, i],
        Y[:, i],
        xerr=X_err[:, i],
        yerr=Y_err[:, i],
        fmt="o",
        c="#000000",
        markersize=5,
        linewidth=1
    )

    mu, std = (np.nanmean(residual), np.nanstd(residual))
    ax.set_title(f"{label_names[i]}: ({mu:.2f}, {std:.2f})")

    limits = np.array([
        ax.get_xlim(),
        ax.get_ylim()
    ])
    limits = (np.min(limits), np.max(limits))
    ax.plot(
        limits,
        limits,
        c="#666666",
        ls=":",
        zorder=-1
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)

fig.tight_layout()

fig.savefig(f"{__file__[:-3]}.png")