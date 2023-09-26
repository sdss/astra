
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from glob import glob
from astra.utils import log, expand_path
from astra.pipelines.ferre.utils import (
    read_ferre_headers,
    read_control_file, 
    read_input_parameter_file,
    read_output_parameter_file,
    read_and_sort_output_data_file,
    parse_ferre_spectrum_name,
    parse_header_path,
    get_apogee_segment_indices,
    read_file_with_name_and_data,
    read_input_data_file,
    get_apogee_pixel_mask,
    TRANSLATE_LABELS
)

MASK = get_apogee_pixel_mask()

def unmasked_pixel_array(a):
    b = np.nan * np.ones(MASK.shape, dtype=float)
    b[MASK] = a
    return b


# TODO: Put this utility elsewhere
def fill_between_steps(ax, x, y1, y2=0, h_align='mid', **kwargs):
    """
    Fill between for step plots in matplotlib.

    **kwargs will be passed to the matplotlib fill_between() function.
    """

    # If no Axes opject given, grab the current one:

    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx = xx - xstep / 2.
    elif h_align == 'right':
        xx = xx - xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)


def plot_ferre_result(
    name, 
    rectified_flux, 
    rectified_e_flux,
    rectified_model_flux,
    input_params,
    output_params,
    synthfile_headers=None,
    all_input_parameters=None,
    all_output_parameters=None,
):

    meta = parse_ferre_spectrum_name(name)
    

    wavelength = 10**(4.179 + 6e-6 * np.arange(8575))
    fig = plt.figure(figsize=(10, 4))

    x_index, y_index = (-1, -2)
    #z_index = -3 # microturbulence
    z_index = -4 # metallicity
    z_label = "[Fe/H]"

    K, L, M = (3, 4, 2)
    spectrum_axes = [plt.subplot2grid((K, L), (i, 0), colspan=L-M) for i in range(K)]
    hrd_axes = plt.subplot2grid((K, L), (0, L-M), rowspan=K, colspan=M)
    #fig, axes = plt.subplots(3, 1)
    for i, (si, n) in enumerate(zip(*get_apogee_segment_indices())):
        ax = spectrum_axes[i]

        ax.plot(
            wavelength[si:si+n],
            unmasked_pixel_array(rectified_model_flux)[si:si+n],
            c="tab:red",
            zorder=3,
        )
        ylim = np.array(ax.get_ylim())
        # Add 10% either side.
        ylim = [ylim[0] - 0.05 * np.ptp(ylim), ylim[1] + 0.05 * np.ptp(ylim)]

        y = unmasked_pixel_array(rectified_flux)[si:si+n]
        y_err = unmasked_pixel_array(rectified_e_flux)[si:si+n]
        ax.plot(
            wavelength[si:si+n], 
            y,
            c='k',
            drawstyle="steps-mid",
            zorder=2,
        )

        fill_between_steps(
            ax,
            wavelength[si:si+n],
            y - y_err,
            y + y_err,
            facecolor="#cccccc",
            zorder=-1
        )

        ax.set_ylim(ylim)
        #ax.set_ylim(0, 1.5)
    

        ax.axhline(1.0, c="#666666", ls=":", lw=1, zorder=0)
        if ax.is_last_row():
            ax.set_xlabel(f"Wavelength [A]")

        xp = [input_params[x_index], output_params[x_index]]
        yp = [input_params[y_index], output_params[y_index]]
        hrd_axes.scatter(
            xp,
            yp,
            c="tab:red",
            s=[5, 30],
            zorder=2,
        )
        hrd_axes.plot(xp, yp, c="tab:red", ls=":", lw=0.5, zorder=2)
        

        ax.yaxis.set_major_locator(MaxNLocator(3))
    
    if synthfile_headers is not None:
        patches = create_patches_from_bounds(synthfile_headers["LLIMITS"], synthfile_headers["ULIMITS"], x_index, y_index, synthfile_headers["STEPS"])
        for patch in patches:
            hrd_axes.add_patch(patch)

    if all_input_parameters is not None:

        x, y = (all_input_parameters[:, x_index], all_input_parameters[:, y_index])
        show = (x > -999) & (y > -999)
        hrd_axes.scatter(
            x[show], y[show],
            facecolor="#cccccc",
            s=5,
            zorder=-10
        )

    if all_output_parameters is not None:
        x, y = (all_output_parameters[:, x_index], all_output_parameters[:, y_index])
        z = all_output_parameters[:, z_index]
        show = (x > -999) & (y > -999) & (z > -999)
        z = all_output_parameters[:, z_index]
         
        scat = hrd_axes.scatter(
            x[show],
            y[show],
            #facecolor="#666666",
            c=z[show],
            alpha=0.5,
            s=5,
            zorder=1
        )
        cbar = plt.colorbar(scat)
        cbar.set_label(z_label)

    if all_input_parameters is not None and all_output_parameters is not None:
        xi, yi = (all_input_parameters[:, x_index], all_input_parameters[:, y_index])
        xo, yo = (all_output_parameters[:, x_index], all_output_parameters[:, y_index])

        xi[xi < -999] = np.nan
        yi[yi < -999] = np.nan
        xo[xo < -999] = np.nan
        yo[yo < -999] = np.nan
        for j in range(xi.size):
            hrd_axes.plot(
                [xi[j], xo[j]],
                [yi[j], yo[j]],
                c="#cccccc",
                lw=0.5,
                ls=":",
                zorder=-1
            )

    hrd_axes.set_ylabel("Surface gravity")
    hrd_axes.set_xlabel("Effective temperature [K]")
    hrd_axes.set_xlim(hrd_axes.get_xlim()[::-1])
    hrd_axes.set_ylim(hrd_axes.get_ylim()[::-1])

    output_dict = dict(zip(synthfile_headers["LABEL"], output_params))
    title = " / ".join([f"{k.lower()}={v:.2f}" for k, v in output_dict.items()][::-1])

    fig.suptitle(
        f"{title} / initial_flags={meta['initial_flags']}"
    )
    #fig.suptitle(
    #    f"source_id:{meta['source_id']} / spectrum_id:{meta['spectrum_id']} / upstream_id:{meta['upstream_id']} / initial_flags:{meta['initial_flags']}"
    #)
    fig.tight_layout()
    
    return fig
    

def create_patches_from_bounds(lower_limits, upper_limits, x_index, y_index, steps, **kwargs):
    
    colors = ("k", "#666666", "#CCCCCC")
    linestyles = (":", ":", ":")
    offsets = (np.zeros_like(steps), steps / 8, steps)

    patches = []
    for offset, color, linestyle in zip(offsets, colors, linestyles):

        patches.append(
            Rectangle(
                (lower_limits[x_index] + offset[x_index], lower_limits[y_index] + offset[y_index]),
                upper_limits[x_index] - lower_limits[x_index] - offset[x_index] * 2,
                upper_limits[y_index] - lower_limits[y_index] - offset[y_index] * 2,
                fill=None, 
                edgecolor=color,
                linestyle=linestyle,
                zorder=-10
            )
        )
    return patches

def _create_quick_look_plots(input_path, plot_dir):

    short_grid_name = input_path.split("/")[-2]
    pwd = os.path.dirname(input_path)

    log.info(f"Creating figures from {pwd}")
    control_kwds = read_control_file(input_path)

    # Load input files.
    input_names, input_parameters = read_input_parameter_file(pwd, control_kwds)   
    log.info(f"There are {len(input_names)} sources")

    parameters, e_parameters, meta, names_with_missing_outputs = read_output_parameter_file(pwd, control_kwds, input_names)
    flux = read_input_data_file(os.path.join(pwd, control_kwds["FFILE"]))
    e_flux = read_input_data_file(os.path.join(pwd, control_kwds["ERFILE"]))

    rectified_flux, names_with_missing_rectified_flux, output_rectified_model_flux_indices = read_and_sort_output_data_file(
        os.path.join(pwd, control_kwds["SFFILE"]),
        input_names
    )        
    rectified_e_flux = e_flux / (flux / rectified_flux)
    rectified_model_flux, names_with_missing_model_flux, output_model_flux_indices = read_and_sort_output_data_file(
        os.path.join(pwd, "rectified_model_flux.output"), 
        input_names
    )

    # Get limits 
    headers, *segment_headers = read_ferre_headers(expand_path(control_kwds["SYNTHFILE(1)"]))
    
    # For each input name, make a plot
    for j, input_name in enumerate(input_names):
        meta = parse_ferre_spectrum_name(input_name)
        fig = plot_ferre_result(
            input_name,
            rectified_flux[j],
            rectified_e_flux[j],
            rectified_model_flux[j],
            input_parameters[j],
            parameters[j],
            synthfile_headers=headers,
            all_input_parameters=input_parameters,
            all_output_parameters=parameters,
        )
        fig.savefig(f"{plot_dir}/{meta['source_id']}_{meta['spectrum_id']}_{short_grid_name}_{meta['index']}.png")
        del fig
        plt.close("all")


def create_quick_look_plots_for_aspcap_stage(stage_dir):
    """
    Create quick look plots for an ASPCAP stage.
    
    :param stage_dir:
        The directory of the ASPCAP stage (e.g., some/path/coarse or some/path/params)
    """
    
    # Plot spectrum vs final spectrum (both rectified)
    plot_dir = os.path.join(f"{stage_dir}/plots")
    os.makedirs(plot_dir, exist_ok=True)

    input_paths = glob(f"{stage_dir}/*/input.nml")
    for input_path in input_paths:
        try:
            _create_quick_look_plots(input_path, plot_dir)
        except:
            log.exception(f"Exception when creating quick look plots for {input_path}")
            continue
        


if __name__ == "__main__":

    create_quick_look_plots_for_aspcap_stage("/uufs/chpc.utah.edu/common/home/u6020307/20230605_ngc188/params/")

