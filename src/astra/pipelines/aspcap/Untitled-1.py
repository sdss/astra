# "$MWM_ASTRA/pipelines/aspcap/synspec_dr17_marcs_header_paths.list"

import os
import numpy as np
import re
import h5py as h5
from astropy.io import fits
from glob import glob
from tqdm import tqdm
from scipy.interpolate import splrep, splev

PATTERN = re.compile("a(?P<alpha_m>[pm]\d+)c(?P<c_m>[pm]\d+)n(?P<n_m>[pm]\d+)v(?P<v_dop>[pm]\d+)")

def _path_float(v):
    sign = {
        "m": -1,
        "p": +1
    }[v[0]]
    return sign * float(v[1:])/10.0

def _get_grid_points(path):
    return tuple(map(_path_float, re.search(PATTERN, os.path.basename(path)).groups()))
    
    

def read_ferre_grid(folder, mask):
    
    paths = glob(f"{folder}/a*.fits")
    
    path_grid_points = np.array(list(map(_get_grid_points, paths)))
    
    grid_points = list(map(np.unique, path_grid_points.T))
    label_names = ["alpha_m", "c_m", "n_m", "v_dop"]
    
    # open the first file to get the header
    with fits.open(paths[0]) as image:
        for n in (4, 3, 2):
            grid_points.append(image[0].header[f"CRVAL{n}"] + np.arange(image[0].header[f"NAXIS{n}"]) * image[0].header[f"CDELT{n}"])
            label_names.append(image[0].header[f"CTYPE{n}"].lower())

    pixels = np.sum(mask)    
    shape = list(map(len, grid_points)) + [pixels]
    model_flux = np.empty(shape)
    
    for (path, grid_point) in zip(tqdm(paths), path_grid_points):
        with fits.open(path) as image:
            model_flux[tuple(map(np.searchsorted, grid_points, grid_point))] = image[0].data[..., mask]
    
    return (tuple(label_names), tuple(grid_points), model_flux)
    

def write_grid(grid, path):
    with h5.File(path, "w") as fp:
        label_names, grid_points, model_flux = grid
        fp.create_dataset("label_names", data=np.array(label_names, dtype="S"))
        gp = fp.create_group("grid_points")
        for label, points in zip(label_names, grid_points):
            gp.create_dataset(label, data=points)
        fp.create_dataset("model_flux", data=model_flux)
    
    return True

def read_grid(path):
    with h5.File(path, "r") as fp:
        grid_points = tuple([fp[f"grid_points/{label}"] for label in fp["grid_points"]])
        
        return (tuple(fp["label_names"]), grid_points, fp["model_flux"])

    

# get some random interpolated spectra 


def estimate_stellar_parameters(
    grid, 
    flux, 
    ivar, 
    mask, 
    refinement_levels=(3, 2, 1), 
    overlap=0,
    include_end_point_in_first_refinement=True,
    no_slice_on=None,
):
    
    label_names, grid_points, model_flux = grid
    
    # do the iterative refinement
    L = len(label_names)
    #refinement_levels = (2**np.arange(refinements)[::-1])
    s_indices = np.zeros(L, dtype=int)
    e_indices = model_flux.shape[:L]
    
    get_offset = lambda s, o: int(np.ceil((1 + o) * s))

    mask_flux, mask_ivar = (flux[mask], ivar[mask])
    last_chi2 = None
    for i, s in enumerate(refinement_levels):
        is_sliced = np.ones(L, dtype=bool)
        if no_slice_on is not None:
            sliced = []
            for j, (si, ei) in enumerate(zip(s_indices, e_indices)):
                if j in no_slice_on:
                    sliced.append(slice(None))
                    is_sliced[j] = False
                else:
                    sliced.append(slice(si, ei, s))
            sliced = tuple(sliced)
        else:        
            sliced = tuple([slice(si, ei, s) for si, ei in zip(s_indices, e_indices)])
        
        chi2 = np.sum((model_flux[sliced] - mask_flux)**2 * mask_ivar, axis=-1)
        index = np.array(np.unravel_index(np.argmin(chi2), chi2.shape))
        index[is_sliced] = index[is_sliced] * s + s_indices[is_sliced]
        
        print(s, index, chi2.shape, np.min(chi2), s_indices, e_indices)
        #assert last_chi2 is None or np.min(chi2) <= last_chi2
        #last_chi2 = np.min(chi2)
        if s > 1:            
            offset = get_offset(s, overlap)
            s_indices = np.maximum(0, index - offset)
            e_indices = np.minimum(model_flux.shape[:L], index + offset + 1)
            
            if no_slice_on is not None:
                for j in no_slice_on:
                    s_indices[j] = 0
                    e_indices[j] = model_flux.shape[j]
            
            assert np.all(e_indices > s_indices)

    # Do nD quadratic case.


    # Now do 1D case.

    stuff = []
    point = []
    for l in range(L):
        axis = list(range(L))
        axis.remove(l)
        
        x = grid_points[l][s_indices[l]:e_indices[l]]
        y = np.min(chi2, axis=tuple(axis))
                
        idx = np.argmin(y)
        si, ei = (max(0, idx - 1), min(idx + 2, len(x)))
        xf, yf = (x[si:ei], y[si:ei])
        
        if xf.size < 3:
            # include the next best point
            idx = list(range(si, ei))
            for v in np.argsort(y):
                if v not in idx:
                    idx.append(v)
                    idx = np.sort(idx)
                    break
                
            xf, yf = (x[idx], y[idx])
                
        a, b, c = coeff = np.polyfit(xf, yf, 2)
        # ax^2 + bx + c = 0
        # dy/dx = 2ax + b
        # 2ax + b = 0
        x_min = -0.5 * b/a
        x_min = np.clip(x_min, *x[[0, -1]])        
        point.append(x_min)
        stuff.append((x, y, xf, yf, coeff, np.polyval(coeff, x_min)))
        #if label_names[l] == "c_m":
        #    raise a
        
        '''
        if label_names[l] == "logg":
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            xi = np.linspace(x[0], x[-1], 1000)
            yi = np.polyval(coeff, xi)
            ax.scatter(x, y)        
            ax.plot(xi, yi)
            raise a
        '''
        
    
    def forward_model_2d(x, y,
        c0,
        a1, a2, a3,
        b1, b2, b3,
        c1, c2, c3
    ):
        "non-mixing test plynomial"
        out = c0
        out += a1 * x**1 + a2 * x**2 + a3 * x**3
        out += b1 * y**1 + b2 * y**2 + b3 * y**3
        out += c1 * x*y + c2 * x**2*y + c3 * x*y**2
        return out
    
    xi = list(label_names).index("logg")
    yi = list(label_names).index("c_m")
    
    axis = tuple(set(list(range(L))).difference({xi, yi}))
    x = grid_points[xi][s_indices[xi]:e_indices[xi]]
    y = grid_points[yi][s_indices[yi]:e_indices[yi]]
    z = np.min(chi2, axis=axis)
    
    def cost(p, x, y, z):
        return np.sum((forward_model_2d(x, y, *p) - z)**2)
    
    X, Y = np.meshgrid(x, y)
    from scipy import optimize as op
    p_opt = op.minimize(
        cost,
        np.zeros(10),
        (X, Y, z)
    )
    
    raise a
            
        
    p_opt_dict = dict(zip(label_names, point))
    return (p_opt_dict, chi2, s_indices, e_indices, stuff)

    


if __name__ == "__main__":
    
    from astra.utils import expand_path
    from astra.models import FerreStellarParameters
    
    short_grid_name = "apo25m_a_GKg"
    folder = "/uufs/chpc.utah.edu/common/home/sdss50/dr17/apogee/spectro/speclib/synth/synspec/marcs/giantisotopes/sgGK_200921nlte_lsfa"

    r = list(
        FerreStellarParameters
        .select()
        .where(FerreStellarParameters.short_grid_name == short_grid_name)
        .where(FerreStellarParameters.ferre_flags == 0)
        .limit(10)
    )[4]
    
    flux = r.unmask(r.model_flux)
    ivar = np.ones_like(flux)
    
    finite = np.isfinite(flux)
    flux[~finite] = 0
    ivar[~finite] = 0 
    
    
    grid_path = expand_path(f"~/sas_home/{short_grid_name}.h5")
    grid = read_ferre_grid(folder, finite)
    
    """
    grid = list(grid)
    grid[1] = list(grid[1])
    grid[1][3] = 10**grid[1][3]
    """
    
    from time import time
    
    p_input = dict(
        teff=r.teff,
        logg=r.logg,
        m_h=r.m_h,
        alpha_m=r.alpha_m,
        c_m=r.c_m,
        n_m=r.n_m,
        v_dop=10**r.log10_v_micro
    )
    
    t_init = time()
    (p_opt_dict, chi2, s_indices, e_indices, stuff) = estimate_stellar_parameters(grid, flux, ivar, finite, refinement_levels=(3, 2, 1))
    

    
        
    def make_big_plot(grid, p_opt_dict, p_input, chi2, s_indices, e_indices, stuff):
            
        
        
        # pre-compute vmin, vmax to keep consistent colours
        vmin, vmax = (None, None)
        ymin, ymax = (None, None)
        for i in range(L):
            for j in range(L):
                if j > i: continue            
                xi, yi = (j, i + 1)
                axis = tuple(set(list(range(L))).difference({xi, yi}))
                chi2_surf = np.min(chi2, axis=axis)
                
                if vmin is None:
                    vmin, vmax = (np.min(chi2_surf), np.max(chi2_surf))
                else:
                    vmin = min(vmin, np.min(chi2_surf))
                    vmax = max(vmax, np.max(chi2_surf))
                    
            axis = tuple(set(list(range(L))).difference({i}))
            chi2_line = np.min(chi2, axis=axis)
            x, y, xf, yf, coeff, _ = stuff[i]
            xc = np.linspace(x[0], x[-1], 1000)
            yc = np.polyval(coeff, xc)
                    
            if ymin is None:
                ymin, ymax = (np.min(yc), np.mean(chi2_line))
            else:
                ymin = min(ymin, np.min(yc))
                ymax = max(ymax, np.mean(chi2_line))
            
        
        # add a fraction
        f = 0.05 * np.ptp(vmax - vmin)
        vmin, vmax = (vmin - f, vmax + f)
        f = 0.05 * np.ptp(ymax - ymin)
        ymin, ymax = (ymin - f, ymax + f)
                
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(L, L)
        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                if j > i:
                    ax.set_visible(False)
                    continue
                            
                xi, yi = (j, i)
                #ax.set_title(f"{xi}, {yi}")
                #continue
                
                if j == i:
                    # Draw a big arse corner plot
                    x, y, xf, yf, coeff, _ = stuff[xi]
                    xc = np.linspace(x[0], x[-1], 1000)
                    ax.plot(xc, np.polyval(coeff, xc), c="#666666")
                    ax.axvline(p_opt_dict[grid[0][i]], c="tab:red", ls="--")        
                    ax.axvline(p_input[grid[0][i]], c="tab:blue", ls="--")
                    ax.scatter(x, y, c="#666666", s=5, zorder=10, ec="k", lw=0.5)
                    ax.scatter(xf, yf, c="k", s=5, zorder=10)
                    
                    ax.set_ylim(ymin, ymax)
                    
                    if ax.is_first_col():
                        ax.set_ylabel(r"\chi^2")
                    else:
                        ax.set_yticks([])
                    
                    if ax.is_last_row():
                        ax.set_xlabel(grid[0][xi])
                    else:
                        ax.set_xticks([])                
                    continue
                
                axis = tuple(set(list(range(L))).difference({xi, yi}))
                chi2_surf = np.min(chi2, axis=axis)
                
                
                x = grid[1][xi][s_indices[xi]:e_indices[xi]]
                y = grid[1][yi][s_indices[yi]:e_indices[yi]]
                xlabel, ylabel = (grid[0][xi], grid[0][yi])
                
                ax.imshow(
                    chi2_surf.T,
                    extent=(x[0], x[-1], y[0], y[-1]),
                    aspect="auto",
                    origin="lower",
                    interpolation="none",
                    vmin=vmin,
                    vmax=vmax
                )
                if ax.is_last_row():
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xticks([])
                
                if ax.is_first_col():
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_yticks([])
                
                ax.scatter(
                    [p_opt_dict[xlabel]],
                    [p_opt_dict[ylabel]],
                    facecolor="tab:red",
                    s=10,
                    alpha=0.5
                )
                ax.scatter(
                    [p_input[xlabel]],
                    [p_input[ylabel]],
                    facecolor="tab:blue",
                    s=10,
                    alpha=0.5
                )
        
        return fig
            
        
    
    '''
    if not os.path.exists(grid_path):
        grid = read_ferre_grid(folder, finite)
        raise a
        write_grid(grid, grid_path)
    else:
        grid = read_grid(grid_path)
    '''
    
    r = list(
        FerreStellarParameters
        .select()
        .where(FerreStellarParameters.short_grid_name == short_grid_name)
        .where(FerreStellarParameters.ferre_flags == 0)
        .limit(100)
    )
    
    # subset of grid
    '''
    grid_subset = (
        ("m_h", "logg", "teff"),
        (grid[1][-3], grid[1][-2], grid[1][-1]),
        grid[2][3, 6, 1, 2]
    )
    '''
    
    x = []
    y = []
    chis = []
    
    for k, r_ in enumerate(tqdm(r)):
        
        flux = r_.unmask(r_.model_flux)
        ivar = np.ones_like(flux)
        
        finite = np.isfinite(flux)
        flux[~finite] = 0
        ivar[~finite] = 0         
        #p = estimate_stellar_parameters(grid, flux, ivar, finite)
        (p, chi2, s_indices, e_indices, stuff) = estimate_stellar_parameters(grid, flux, ivar, finite, refinement_levels=(2, 1), no_slice_on=(3, ))

        p_input = dict(
            teff=r_.teff,
            logg=r_.logg,
            m_h=r_.m_h,
            alpha_m=r_.alpha_m,
            c_m=r_.c_m,
            n_m=r_.n_m,
            v_dop=10**r_.log10_v_micro
        )
        fig = make_big_plot(grid, p, p_input, chi2, s_indices, e_indices, stuff)
        fig.savefig(expand_path(f"~/sas_home/20240216_aspcap_self/{r_.task_pk}.png"))
        plt.close("all")
        
        #x.append((r_.m_h, r_.logg, r_.teff))
        #y.append((p["m_h"], p["logg"], p["teff"]))
        x.append((r_.alpha_m, r_.c_m, r_.n_m, 10**r_.log10_v_micro, r_.m_h, r_.logg, r_.teff))
        y.append((p.get("alpha_m", np.nan), p.get("c_m", np.nan), p.get("n_m", np.nan), p.get("v_dop", np.nan), p.get("m_h", np.nan), p.get("logg", np.nan), p.get("teff", np.nan)))
        #y.append((p["teff"], p["logg"], p.get("m_h", np.nan), p.get("alpha_m", np.nan), p.get("c_m", np.nan), p.get("n_m", np.nan), p.get("v_dop", np.nan)))
        chis.append([ea[-1] for ea in stuff])
        

    x = np.array(x)
    y = np.array(y)
    chis = np.array(chis)
    
    vmin, vmax = np.min(chis), np.max(chis)
    
    
    import matplotlib.pyplot as plt
    L = len(grid[0])
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    
    for i, ax in enumerate(axes.flat):
        try:
            x[:,i]
        except:
            ax.set_visible(False)
            continue
            
        scat = ax.scatter(x[:, i], y[:, i], c=y[:,0] - x[:,0])#c=chis[:, i], vmin=vmin, vmax=vmax)
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))
        ax.plot(limits, limits, c="#666666", ls=":", zorder=-1, lw=0.5)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        diff = y[:, i] - x[:, i]
        ax.set_title(f"{np.nanmean(diff):.2f} +/- {np.nanstd(diff):.2f} ({np.isfinite(diff).sum()})")
        ax.set_xlabel(grid[0][i])
    
    cbar = plt.colorbar(scat)
    fig.tight_layout()
    fig.savefig("/uufs/chpc.utah.edu/common/home/u6020307/20240216_aspcap_self2.png", dpi=600)