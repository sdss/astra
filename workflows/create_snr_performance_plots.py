

# Let's make a plot of parameter vs SNR for APOGEENet.

import numpy as np
import astra
import sqlalchemy
from sqlalchemy import create_engine
from tqdm import tqdm
from astra.tasks.base import BaseTask
from astra.tasks.io import (ApStarFile, ApVisitFile)
from astra.utils import get_default

def get_apstar_results(table_name):

    connection_string = get_default(BaseTask, "connection_string")

    engine = create_engine(connection_string)

    connection = engine.connect()
    metadata = sqlalchemy.MetaData(schema="astra")

    table = sqlalchemy.Table(
        table_name,
        metadata,
        autoload=True,
        autoload_with=connection
    )

    # Any limits to place on the query?
    q = sqlalchemy.select([table])

    rows = engine.execute(q).fetchall()

    column_names = tuple([column.name for column in table.columns])
    return (len(rows), column_names, rows)



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_binned_statistic(x, y, z, bins=100, function=np.nanmedian,
                          xlabel=None, ylabel=None, zlabel=None,
                          ax=None, colorbar=False, figsize=(8, 8),
                          vmin=None, vmax=None, min_entries_per_bin=None,
                          subsample=None, mask=None, full_output=False, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    finite = np.isfinite(x * y * z)
    if mask is not None:
        finite *= mask
    if subsample is not None:
        idx = np.where(finite)[0]
        if subsample < 1:
            subsample *= idx.size
        if int(subsample) > idx.size:
            finite = idx
        else:
            finite = np.random.choice(idx, int(subsample), replace=False)
    
    H, xedges, yedges, binnumber = binned_statistic_2d(
        x[finite], y[finite], z[finite],
        statistic=function, bins=bins)

    if min_entries_per_bin is not None:
        if function != "count":
            H_count, xedges, yedges, binnumber = binned_statistic_2d(
                x[finite], y[finite], z[finite],
                statistic="count", bins=bins)

        else:
            H_count = H

        H[H_count < min_entries_per_bin] = np.nan


    if (vmin is None or vmax is None) and "norm" not in kwargs:
        vmin_default, med, vmax_default = np.nanpercentile(H, kwargs.pop("norm_percentiles", [5, 50, 95]))
        if vmin is None:
            vmin = vmin_default
        if vmax is None:
            vmax = vmax_default
    
    imshow_kwds = dict(
        vmin=vmin, vmax=vmax,
        aspect=np.ptp(xedges)/np.ptp(yedges), 
        extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
        cmap="inferno",
        interpolation="bilinear")
    imshow_kwds.update(kwargs)
    
    image = ax.imshow(H.T, **imshow_kwds)
    if colorbar:
        cbar = plt.colorbar(image, ax=ax)
        if zlabel is not None:
            cbar.set_label(zlabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    #fig.tight_layout()

    return (fig, image) if full_output else fig


kwds = {

    "thepayne_apstar": [
        'teff',
        'logg',
        'v_turb',
        'c_h',
        'n_h',
        'o_h',
        'na_h',
        'mg_h',
        'al_h',
        'si_h',
        'p_h',
        's_h',
        'k_h',
        'ca_h',
        'ti_h',
        'v_h',
        'cr_h',
        'mn_h',
        'fe_h',
        'co_h',
        'ni_h',
        'cu_h',
        'ge_h',
        'c12_c13',
        'v_macro',
    ],
    "apogeenet": [
        "teff", "logg", "fe_h"
    ]

}

for table_name, plot_column_names in kwds.items():
        

    N, column_names, rows = get_apstar_results(table_name=table_name)

    snr_index = column_names.index("snr")
    max_cardinality = max([len(row[snr_index]) for row in rows])

    shape = (N, max_cardinality)

    #keys = ("snr", "teff", "logg", "fe_h", "u_teff", "u_logg", "u_fe_h")
    keys = ["snr"] + list(plot_column_names)
    indices = [column_names.index(key) for key in keys]

    data = { key: np.nan * np.ones(shape) for key in keys }
    cardinality = np.ones(N, dtype=int)

    for i, row in enumerate(tqdm(rows)):
        M = cardinality[i] = len(row[snr_index])
        for j, key in zip(indices, keys):
            data[key][i, :M] = row[j]

    # Which stars have multiple visits?
    has_multiple_visits = np.isfinite(data["snr"][:, 1])

    # Now we want to calculate some metric relative to the high S/N case.
    x = []
    diff = { key: [] for key in keys }
    reference_indices = []

    for i, j in enumerate(cardinality):
        n_visits = j - 2 if j > 1 else j
        if n_visits == 1: continue

        x.extend(data["snr"][i, 2:2+n_visits])
        reference_indices.extend([i] * n_visits)
        
        for key in keys:
            reference_value = data[key][i, 0]
            individual_values = data[key][i, 2:2+n_visits]

            diff[key].extend(reference_value - individual_values)

    # Array-ify!
    x = np.array(x)
    reference_indices = np.array(reference_indices, dtype=int)
    for key in keys:
        diff[key] = np.array(diff[key])


    # Calculate some statistic in each bin.
    bin_edge_params = [
        (0, 50, 10),
        (50, 200, 20),
        (200, 1500, 50)
    ]
    bin_edges = np.unique(np.hstack([np.arange(start, end + 1, step) for start, end, step in bin_edge_params]))

    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    y = { key: np.ones(bin_centers.size) for key in keys }

    # Restrict the set somehow?
    #global_mask = data["snr"][reference_indices, 0] > 100
    global_mask = True

    # Now calculate statistic for the bins.
    for i, left_edge in enumerate(bin_edges[:-1]):
        right_edge = bin_edges[i + 1]

        mask = global_mask \
            * (right_edge >= x) * (x >= left_edge)
        
        for key in keys:
            values = diff[key][mask]
            #y[key][i] = np.sqrt(np.sum(values**2)/values.size)
            y[key][i] = np.nanmedian(np.abs(values))



    N = int(np.ceil((len(keys) - 1)/3))

    fig, axes = plt.subplots(3, N, figsize=(3 * N, 9))

    label_limits = {
        "teff": 1000
    }

    max_snr = 250
    axes = np.hstack(axes).flatten()

    for ax, label_name in zip(axes, plot_column_names):

        max_ylim = label_limits.get(label_name, 1)

        y_abs = np.abs(diff[label_name])

        plot_binned_statistic(
            x,
            y_abs,
            y_abs,
            function="count",
            ax=ax,
            bins=50,
            interpolation="none",
            mask=(y_abs < max_ylim) * (x < max_snr),
            vmin=1,
            norm=LogNorm(),
            cmap="Greys"
        )

        """
        ax.scatter(
            x,
            np.abs(diff["teff"]),
            s=1,
            c="#666666",
            alpha=0.3,
            rasterized=True
        )
        """
        ax.plot(
            bin_centers,
            y[label_name],
            'o-',
            markersize=5,
            zorder=10,
        )
        ax.set_xlim(0, max_snr)
        ax.set_ylim(0, max_ylim)

        if ax.is_last_row():
            ax.set_xlabel("S/N")

        ax.set_ylabel(f"|{label_name}|")

    axes[0].text(
        0.95, 
        0.90,
        f"N_visits = {x.size}",    
        transform=axes[0].transAxes,
        horizontalalignment="right"
    )
    axes[0].text(
        0.95,
        0.80,
        f"N_stars = {has_multiple_visits.sum()}",
        transform=axes[0].transAxes,
        horizontalalignment="right"
    )

    for ax in axes[len(plot_column_names):]:
        ax.set_visible(False)

    fig.tight_layout()
    path = f"/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/astra/0.1.11/snr-response-{table_name}.png"
    fig.savefig(path, dpi=300)

