import pickle
import numpy as np
from scipy import (interpolate, optimize as op)
from scipy.signal import find_peaks, peak_widths
from astropy.nddata import StdDevUncertainty, InverseVariance
from astropy.io import fits
from astra import log
from astra.base import TaskInstance, Parameter, DictParameter
from astra.utils import expand_path, executable, flatten, dict_to_list
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
        
from astra.tools.continuum import Sinusoids


from astra.database.astradb import database, InitialKorgOutput, ApogeeNetOutput, AspcapOutput


class APOGEEInitialEstimate(TaskInstance):

    grid_path = Parameter(default="$MWM_ASTRA/component_data/korg/initial_apogee_convolved_roq_20221110.pkl", bundled=True)

    def execute(self):

        with open(expand_path(self.grid_path), "rb") as fp:
            model = pickle.load(fp)

        rectified_model_flux = model["node_flux"] @ model["coefficients"].T

        # Prepare for continuum modelling.
        deg, L = (2, 1400)

        # TODO: Should we just use a single region for all chips? It seems to look "OK" in some scenarios to just use one region,
        #       and in some spectra where the chip edges are problematic, it actually works better to use a single region.
        regions = [
            (15_100.0, 15_832.0), 
            (15_832.0, 16_454.0), 
            (16_454.0, 17_000.0)
        ]
        #regions = [(15_100, 17_000)]
        N, P = model["parameter_values"].shape
        continuum_kwds = dict(regions=regions, deg=deg, L=L)        

        for task, (data_product, ), parameters in self.iterable():
            for spectrum in SpectrumList.read(data_product.path):
                if not spectrum_overlaps(spectrum, 16_500):
                    continue

                # Only using the first spectrum index.
                flux = spectrum.flux.value[0]
                ivar = spectrum.uncertainty.represent_as(InverseVariance).array[0]

                pseudo_continuum_flux = flux / rectified_model_flux

                # Use the minimum pseudo continuum flux array to find skylines.
                # (This is the lowest possible continuum value at each pixel, over all models.)
                min_pseudo_continuum_flux = np.min(pseudo_continuum_flux, axis=0)

                # Inflate errors for chip edges, then create the pseudo continuum inverse variance array.
                ivar_multiplier = inverse_variance_multiplier_for_apogee_chip_edges(spectrum)[0]
                pseudo_continuum_ivar = rectified_model_flux * (ivar * ivar_multiplier) * rectified_model_flux

                is_finite = np.isfinite(min_pseudo_continuum_flux)

                # Set a conservative initial height for finding sky lines.
                height = np.nanmedian(min_pseudo_continuum_flux) + 5 * np.nanstd(min_pseudo_continuum_flux)
                is_sky_line = create_sky_line_mask(min_pseudo_continuum_flux, height=height)

                mask = is_sky_line | ~is_finite

                theta, evaluated_continuum, continuum_args = evaluate_continuum(
                    spectrum, 
                    pseudo_continuum_flux, 
                    pseudo_continuum_ivar, 
                    mask=mask,
                    **continuum_kwds
                )
                
                # Evaluate the \chi^2.
                model_flux = evaluated_continuum * rectified_model_flux
                pixel_chi_sq = (model_flux - flux)**2 * ivar
                chi_sq = np.sum(pixel_chi_sq[:, ~mask], axis=1)
                reduced_chi_sq = chi_sq / (np.sum(~mask) - L - 1)
                print(np.nanmin(reduced_chi_sq))

                # Use the best continuum to recompute the sky line mask.
                best_index = np.argmin(reduced_chi_sq)
                pseudo_continuum_rectified_flux = flux / evaluated_continuum[best_index]
                height = 1 + 5 * np.nanstd(pseudo_continuum_rectified_flux[~mask])
                is_sky_line = create_sky_line_mask(pseudo_continuum_rectified_flux, height=height)

                updated_mask = is_sky_line | ~is_finite
                if not np.all(updated_mask == mask):
                    theta, evaluated_continuum, continuum_args = evaluate_continuum(
                        spectrum, 
                        pseudo_continuum_flux, 
                        pseudo_continuum_ivar, 
                        mask=mask,
                        **continuum_kwds
                    )
                    # Evaluate the \chi^2 (again)
                    model_flux = evaluated_continuum * rectified_model_flux
                    pixel_chi_sq = (model_flux - flux)**2 * ivar
                    chi_sq = np.sum(pixel_chi_sq[:, ~mask], axis=1)
                    reduced_chi_sq = chi_sq / (np.sum(~mask) - L - 1)

                    print(np.nanmin(reduced_chi_sq))

                #import matplotlib.pyplot as plt
                #fig, axes = plt.subplots(2, 2)
                p_opt = np.zeros(P)
                for index in range(P):
                    x_unique = np.sort(np.unique(model["parameter_values"][:, index]))
                    y_min = np.array([np.min([reduced_chi_sq[model["parameter_values"][:, index] == xu]]) for xu in x_unique])

                    tck = interpolate.splrep(x_unique, y_min)
                    x_i = np.linspace(*x_unique[[0, -1]], 1000) 
                    y_i = interpolate.splev(x_i, tck)
                    p_opt[index] = x_i[np.argmin(y_i)]

                    #axes.flat[index].scatter(x_unique, y_min)
                    #axes.flat[index].plot(x_i, y_i)

                # Interpolate the theta coefficients.
                f = interpolate.LinearNDInterpolator(
                    model["parameter_values"],
                    theta.reshape((N, -1)),
                    rescale=True
                )
                theta_opt = f(p_opt).reshape((len(regions), -1))
                #continuum_opt = np.empty_like(flux)
                #for j, ((lower, upper), indices, M_region, M_continuum) in enumerate(zip(*continuum_args)):
                #    continuum_opt[slice(lower, upper)] = M_region.T @ theta_opt[j]
                
                # Save the theta coefficients and the stellar parameters.

                result = dict(
                    teff=p_opt[0],
                    logg=p_opt[1],
                    microturbulence=p_opt[2],
                    m_h=p_opt[3],
                    reduced_chi_sq=reduced_chi_sq[np.argmin(reduced_chi_sq)],
                    theta=theta_opt.flatten(),
                    parent_data_product_id=data_product.id
                )

                # Create or update rows.
                with database.atomic():
                    task.create_or_update_outputs(InitialKorgOutput, [result])



class APOGEEConvolvedInitialEstimate(TaskInstance):

    grid_path = Parameter(default="$MWM_ASTRA/component_data/korg/initial_apogee_convolved_roq_20221110.pkl", bundled=True)

    continuum_method = Parameter(default="astra.tools.continuum.Sinusoids", bundled=True)
    continuum_kwargs = DictParameter(
        default=dict(
            deg=3,
            regions=[(15_100.0, 15_793.0), (15_880.0, 16_417.0), (16_499.0, 17_000.0)],
            mask="$MWM_ASTRA/component_data/ThePayne/cannon_apogee_pixels.npy",
        ),
        bundled=True,
    )

    def execute(self):

        with open(expand_path(self.grid_path), "rb") as fp:
            model = pickle.load(fp)


        A = model["coefficients"].T
        node_indices = model["node_indices"]
        min_bound, max_bound = (
            np.min(model["parameter_values"], axis=0),
            np.max(model["parameter_values"], axis=0),
        )
        ptp_bound = max_bound - min_bound

        whiten = lambda x: (x - min_bound)/ptp_bound
        darken = lambda x: x*ptp_bound + min_bound
        
        interpolator = interpolate.LinearNDInterpolator(
            whiten(model["parameter_values"]),
            model["node_flux"],
            rescale=False
        )

        from astra.contrib.aspcap.utils import approximate_log10_microturbulence
        f_continuum = executable(self.continuum_method)(**self.continuum_kwargs)

        for task, (data_product, ), parameters in self.iterable():
            # TODO: this is assuming only APOGEE data in one HDU, and only doing the first (stacked) spectrum.
            # TODO: Only initializing once from median of grid.
            for spectrum in SpectrumList.read(data_product.path):
                if not spectrum_overlaps(spectrum, 16_500):
                    continue

                continuum = f_continuum.fit(spectrum)(spectrum)
                
                # assume we're only doing stacked spectra for now
                flux = spectrum.flux.value / continuum
                e_flux = spectrum.uncertainty.represent_as(StdDevUncertainty).array / continuum

                flux, e_flux = (flux[0], e_flux[0])
                finite = np.isfinite(flux) * np.isfinite(e_flux)

                # Get some initial guess
                initial_chi_sq = ((flux[node_indices] - model["node_flux"])/e_flux[node_indices])**2
                initial_chi_sq = np.nansum(initial_chi_sq, axis=1)

                p0 = model["parameter_values"][np.argmin(initial_chi_sq)]

                '''
                def f(x, teff, logg, vt, m_h):
                    #print(darken([teff, logg, vt, m_h]))
                    return (interpolator([teff, logg, vt, m_h]) @ A)[0, finite]

                whitened_p_opt, whitened_p_cov = op.curve_fit(
                    f=f,
                    xdata=None,
                    ydata=flux[finite],
                    sigma=e_flux[finite],
                    p0=whiten(p0),
                    absolute_sigma=True,
                    maxfev=10_000,
                    factor=100,
                    epsfcn=1e-6,
                    xtol=1e-10,
                    ftol=1e-10
                )
                model_flux = (interpolator(whitened_p_opt) @ A)[0]
                p_opt = darken(whitened_p_opt)
                '''

                def cost(theta):
                    teff, logg, m_h = theta
                    xi = 10**approximate_log10_microturbulence(logg * ptp_bound[1] + min_bound[1])
                    xi = (xi - min_bound[2])/ptp_bound[2]
                    chi_sq = (((interpolator([teff, logg, xi, m_h]) @ A)[0] - flux) / e_flux)**2
                    if not np.any(np.isfinite(chi_sq)):
                        return np.inf
                    return np.sum(chi_sq[finite])

                opts = []
                for index in np.argsort(initial_chi_sq)[:10]:
                    opts.append(
                        op.minimize(
                            cost, 
                            whiten(model["parameter_values"][index])[[0, 1, 3]],
                            method="Nelder-Mead", 
                            options=dict(maxiter=10_000)
                        )
                    )
                vals = [ea.fun for ea in opts]
                opt = opts[np.argmin(vals)]
                raise a
                alt_model_flux = (interpolator(opt.x) @ A)[0]
                alt_p_opt = darken(opt.x)

                print(np.argmin(vals))
                raise a


                parent_data_product_id = spectrum.meta.get("DATA_PRODUCT_ID", None)
                if parent_data_product_id is None or len(parent_data_product_id) == 0:
                    parent_data_product_id = data_product.id

                result = dict(
                    snr=flatten(spectrum.meta["SNR"])[0],
                    source_id=spectrum.meta.get("SDSS_ID", None),
                    parent_data_product_id=parent_data_product_id,
                )
                result.update(dict(zip(model["parameter_names"], p_opt)))
                result.update(dict(zip([f"e_{k}" for k in model["parameter_names"]], darken(np.sqrt(np.diag(whitened_p_cov))))))

                chi_sq = ((model_flux[finite] - flux[finite])/e_flux[finite])**2
                reduced_chi_sq = chi_sq / (finite.sum() - p_opt.size - 1)

                result.update(chi_sq=chi_sq, reduced_chi_sq=reduced_chi_sq)

                # Create or update rows.
                with database.atomic():
                    task.create_or_update_outputs(KorgOutput, [result])


                #alt_opt = op.minimize(cost, p_opt, p0=whiten(p0), method="Nelder-Mead")

                raise a

                """
                # Get alternative
                q = AspcapOutput.select().where(AspcapOutput.parent_data_product_id == data_product.id).first()
                '''
                alt_p_opt = [q.teff, q.logg, 10**q.log10vdop, q.metals]
                alt_model_flux = (interpolator(alt_p_opt) @ A)[0]
                '''
                image = fits.open(q.task.output_data_products[0].path)
                alt_model_flux = (image[3].data["MODEL_FLUX"]/image[3].data["CONTINUUM"])[0]
                        

                alt_residual = ((alt_model_flux - flux) / e_flux)**2
                alt_chi_sq = np.sum(alt_residual[finite])
                alt_reduced_chi_sq = alt_chi_sq / (alt_residual.size - p_opt.size - 1)

                assert reduced_chi_sq < alt_reduced_chi_sq or not np.all(np.isfinite([reduced_chi_sq, alt_reduced_chi_sq]))
                import matplotlib.pyplot as plt
                L = 5
                fig, axes = plt.subplots(L, 1, figsize=(10, 10))

                # hide non-data chunks
                missing = ~np.isfinite(flux)
                model_flux[missing] = np.nan
                alt_model_flux[missing] = np.nan
            
                chunk = np.ptp(spectrum.wavelength.value[~missing]) / L
                start = spectrum.wavelength.value[~missing][0]
                for i, ax in enumerate(axes):
                    ax.plot(spectrum.wavelength.value, flux, c='k')
                    ax.plot(spectrum.wavelength.value, model_flux, c="tab:red")
                    ax.plot(spectrum.wavelength.value, alt_model_flux, c="tab:blue")
                    ax.plot(spectrum.wavelength.value, model_flux - flux, c="tab:red")
                    ax.plot(spectrum.wavelength.value, alt_model_flux - flux, c="tab:blue")
                    ax.set_ylim(-0.1, 1.2)
                    ax.axhline(0, c="#666666", ls=":", zorder=-1, lw=0.5)
                    ax.set_xlim(start + i * chunk, start + (i + 1) * chunk)
                
                axes[0].set_title(f"korg-chisq: {reduced_chi_sq:.2e} ({p_opt[0]:.0f}/{p_opt[1]:.1f}/{p_opt[2]:.1f}/{p_opt[3]:.1f}), alt-chisq: {alt_reduced_chi_sq:.2e} ({alt_p_opt[0]:.0f}/{alt_p_opt[1]:.1f}/{alt_p_opt[2]:.1f}/{alt_p_opt[3]:.1f}")
                fig.tight_layout()
                fig.savefig(f"/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/astra/korg-tests/{data_product.id}.png", dpi=300)

                """


def evaluate_continuum(spectrum, pseudo_continuum_flux, pseudo_continuum_ivar, **kwargs):
    # mask, deg, L, regions):
    N, P = pseudo_continuum_flux.shape    
    continuum_args = Sinusoids(**kwargs)._initialize(spectrum)
    
    num_regions = len(continuum_args[0]) 
    num_coefficients = continuum_args[3][0].shape[0]
    theta = np.empty((N, num_regions, num_coefficients))
    evaluated_continuum = np.empty_like(pseudo_continuum_flux)

    for i, (c_flux, c_ivar) in enumerate(zip(pseudo_continuum_flux, pseudo_continuum_ivar)):
        for j, ((lower, upper), indices, M_region, M_continuum) in enumerate(zip(*continuum_args)):
            MTM = M_continuum @ (c_ivar[indices][:, None] * M_continuum.T)
            MTy = M_continuum @ (c_ivar[indices] * c_flux[indices]).T

            eigenvalues = np.linalg.eigvalsh(MTM)
            MTM[np.diag_indices(len(MTM))] += 1e-6 * np.max(eigenvalues)
            # eigenvalues = np.linalg.eigvalsh(MTM)
            # condition_number = max(eigenvalues) / min(eigenvalues)
            # TODO: warn on high condition number
            theta[i, j] = np.linalg.solve(MTM, MTy)
            evaluated_continuum[i, slice(lower, upper)] = M_region.T @ theta[i, j]

    return (theta, evaluated_continuum, continuum_args)


def create_sky_line_mask(flux, height=None, width_buffer=2):
    is_sky_line = np.zeros(flux.shape, dtype=bool)
    if height is None:
        height = np.nanmedian(flux) + np.nanstd(flux)
    indices, _ = find_peaks(flux, height=height)
    widths, *_ = peak_widths(flux, indices)
    for peak_index, peak_width in zip(indices, widths):
        wi = int(np.ceil(width_buffer * peak_width))
        is_sky_line[peak_index - wi:peak_index + wi] = True
    return is_sky_line


def inverse_variance_multiplier_for_apogee_chip_edges(
        spectrum, 
        scalar=1e-2, 
        N_pixels_per_edge=10, 
        chip_edges=(15832, 16454)
    ):
    # Some things here are hard-coded in, simply because they haven't changed in 10 years.
    regions = np.hstack([15_000, np.repeat(chip_edges, 2), 17_000]).reshape((-1, 2))
    N, P = spectrum.flux.shape
    ivar_multiplier = np.ones((N, P))
    for wl_lower, wl_upper in regions:
        lower, upper = spectrum.wavelength.value.searchsorted([wl_lower, wl_upper])
        for i in range(N):
            first_finite, last_finite = np.where(np.isfinite(spectrum.flux.value[i, slice(lower, upper)]))[0][[0, -1]]
            ivar_multiplier[i, lower + first_finite:lower + first_finite + N_pixels_per_edge] *= scalar
            ivar_multiplier[i, lower + last_finite - N_pixels_per_edge:lower + last_finite] *= scalar
    return ivar_multiplier


