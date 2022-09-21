import numpy as np
import os
import scipy.optimize as op

from astra import log, __version__
from astropy.nddata import StdDevUncertainty, InverseVariance
from astra.utils import expand_path
from astra.base import TaskInstance, TupleParameter, Parameter
from astra.database.astradb import WhiteDwarfOutput, WhiteDwarfLineRatiosOutput
from astra.tools.spectrum import Spectrum1D
from astra.contrib.wd.utils import line_features

from astra.sdss.datamodels.base import get_extname
from astra.sdss.datamodels.pipeline import create_pipeline_product


from astra.contrib.wd.fitting import (
    norm_spectra,
    fit_line,
    fit_func,
    err_func,
    hot_vs_cold,
)

data_dir = expand_path("$MWM_ASTRA/component_data/wd/")

class LineRatios(TaskInstance):

    # Don't want to store large tuples in the database for every single task.
    # Instead we will default to None and specify defaults.
    wavelength_regions = TupleParameter(default=None)
    polyfit_regions = TupleParameter(default=None)
    polyfit_order = Parameter(default=5)

    def execute(self):

        default_wavelength_regions = (
            [3860, 3900],  # Balmer line
            [3950, 4000],  # Balmer line
            [4085, 4120],  # Balmer line
            [4320, 4360],  # Balmer line
            [4840, 4880],  # Balmer line
            [6540, 6580],  # Balmer line
            [3880, 3905],  # He I/II line
            [3955, 3975],  # He I/II line
            [3990, 4056],  # He I/II line
            [4110, 4140],  # He I/II line
            [4370, 4410],  # He I/II line
            [4450, 4485],  # He I/II line
            [4705, 4725],  # He I/II line
            [4900, 4950],  # He I/II line
            [5000, 5030],  # He I/II line
            [5860, 5890],  # He I/II line
            [6670, 6700],  # He I/II line
            [7050, 7090],  # He I/II line
            [7265, 7300],  # He I/II line
            [4600, 4750],  # Molecular C absorption band
            [5000, 5160],  # Molecular C absorption band
            [3925, 3940],  # Ca H/K line
            [3960, 3975],  # Ca H/K line
        )
        default_polyfit_regions = (
            [3850, 3870],
            [4220, 4245],
            [5250, 5400],
            [6100, 6470],
            [7100, 9000],
        )

        for task, data_products, parameters in self.iterable():
            wavelength_regions = parameters["wavelength_regions"] or default_wavelength_regions
            polyfit_regions = parameters["polyfit_regions"] or default_polyfit_regions
            polyfit_order = parameters["polyfit_order"]

            all_features = []
            for data_product in data_products:
                # Only use BOSS APO spectra.
                # TODO: Revise when we have BOSS LCO spectra.
                spectrum = Spectrum1D.read(data_product.path, hdu=1)

                all_line_ratios = line_features(
                    spectrum,
                    wavelength_regions=wavelength_regions,
                    polyfit_regions=polyfit_regions,
                    polyfit_order=polyfit_order,
                )
                for line_ratios in all_line_ratios:
                    for (wl_start, wl_end), line_ratio in zip(wavelength_regions, line_ratios):
                        all_features.append(
                            dict(
                                wavelength_start=wl_start,
                                wavelength_end=wl_end,
                                line_ratio=line_ratio
                            )
                        )

            # Create outputs: one for every line feature.
            task.create_or_update_outputs(WhiteDwarfLineRatiosOutput, all_features)
            
        return None

    

class WhiteDwarfStellarParameters(TaskInstance):

    """
    Estimate effective temperature and surface gravity for DA-type white dwarfs.
    """

    model_grid = Parameter(default="da2014", bundled=True)
    parallax = Parameter(default=None)
    phot_g_mean_mag = Parameter(default=None)
    plot = Parameter(default=False, bundled=True)

    def execute(self):

        for task, (data_product, ), parameters in self.iterable():

            parallax = parameters["parallax"]
            phot_g_mean_mag = parameters["phot_g_mean_mag"]
                
            # Only ever on BOSS data.
            assert data_product.filetype == "mwmStar"
            spectrum = Spectrum1D.read(data_product.path, hdu=1)

            if parallax is None or phot_g_mean_mag is None:
                parallax = parallax or spectrum.meta["PLX"]
                phot_g_mean_mag = phot_g_mean_mag or spectrum.meta["G_MAG"]

            log.info(
                f"{self} on {data_product} using parallax = {parallax:.2f} and phot_g_mean_mag = {phot_g_mean_mag:.2f}"
            )

            # TODO: some better classifier
            wd_type = "DA"

            wl = spectrum.wavelength.value
            flux = spectrum.flux.value
            flux_err = spectrum.uncertainty.represent_as(StdDevUncertainty).array

            if flux.ndim == 2:
                # TODO: Not yet handle mwmVisit
                flux = flux[0]
                flux_err = flux_err[0] 
            

            data = np.vstack([wl, flux, flux_err]).T
            data = data[
                (np.isnan(data[:, 1]) == False) & (data[:, 0] > 3500) & (data[:, 0] < 8000)
            ]
            # normalize data
            spec_n, cont_flux = norm_spectra(data, type=wd_type)

            # crop the  relevant lines for initial guess
            line_crop = np.loadtxt(data_dir + "/line_crop.dat")
            l_crop = line_crop[(line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())]

            # fit entire grid to find good starting point
            best = fit_line(
                spec_n, l_crop, model_in=None, quick=True, model=self.model_grid
            )
            first_T, first_g = best[2][0], best[2][1]
            all_chi, all_TL = best[4], best[3]

            # Teff values determine what line ranges to use precise fit
            if first_T >= 16000 and first_T <= 40000:
                line_crop = np.loadtxt(data_dir + "/line_crop.dat")
                l_crop = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]
            elif first_T >= 8000 and first_T < 16000:
                line_crop = np.loadtxt(data_dir + "/line_crop_cool.dat")
                l_crop = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]
            elif first_T < 8000:
                line_crop = np.loadtxt(data_dir + "/line_crop_vcool.dat")
                l_crop = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]
            elif first_T > 40000:
                line_crop = np.loadtxt(data_dir + "/line_crop_hot.dat")
                l_crop = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]

            # fit the spectrum
            new_best = op.minimize(
                fit_func,
                (first_T, first_g, 10.0),
                bounds=((6000, 89000), (650, 899), (None, None)),
                args=(spec_n, l_crop, self.model_grid, 0),
                method="L-BFGS-B",
            )
            other_T = op.minimize(
                err_func,
                (first_T, first_g),
                bounds=((6000, 89000), (650, 899)),
                args=(new_best.x[2], new_best.fun, spec_n, l_crop, self.model_grid),
                method="L-BFGS-B",
            )
            Teff, Teff_err, logg, logg_err, rv = (
                new_best.x[0],
                abs(new_best.x[0] - other_T.x[0]),
                new_best.x[1],
                abs((new_best.x[1] - other_T.x[1]) / 100),
                new_best.x[2],
            )

            # Repeat everything for second solution
            if first_T <= 13000.0:
                other_TL, other_chi = (
                    all_TL[all_TL[:, 0] > 13000.0],
                    all_chi[all_TL[:, 0] > 13000.0],
                )
                other_sol = other_TL[other_chi == np.min(other_chi)]
            elif first_T > 13000.0:
                other_TL, other_chi = (
                    all_TL[all_TL[:, 0] <= 13000.0],
                    all_chi[all_TL[:, 0] <= 13000.0],
                )
                other_sol = other_TL[other_chi == np.min(other_chi)]

            if other_sol[0][0] >= 16000 and other_sol[0][0] <= 40000:
                line_crop = np.loadtxt(data_dir + "/line_crop.dat")
                l_crop_s = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]
            elif other_sol[0][0] >= 8000 and other_sol[0][0] < 16000:
                line_crop = np.loadtxt(data_dir + "/line_crop_cool.dat")
                l_crop_s = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]
            elif other_sol[0][0] < 8000:  # first_T<other_sol[0][0]:
                line_crop = np.loadtxt(data_dir + "/line_crop_vcool.dat")
                l_crop_s = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]
            elif other_sol[0][0] > 40000:  # first_T>other_sol[0][0]:
                line_crop = np.loadtxt(data_dir + "/line_crop_hot.dat")
                l_crop_s = line_crop[
                    (line_crop[:, 0] > wl.min()) & (line_crop[:, 1] < wl.max())
                ]

            sec_best = op.minimize(
                fit_func,
                (other_sol[0][0], other_sol[0][1], new_best.x[2]),
                bounds=((6000, 89000), (650, 899), (None, None)),
                args=(spec_n, l_crop, self.model_grid, 0),
                method="L-BFGS-B",
            )  # xtol=1. ftol=1.
            other_T2 = op.minimize(
                err_func,
                (other_sol[0][0], other_sol[0][1]),
                bounds=((6000, 89000), (650, 899)),
                args=(new_best.x[2], sec_best.fun, spec_n, l_crop, self.model_grid),
                method="L-BFGS-B",
            )

            Teff2, Teff_err2, logg2, logg_err2 = (
                sec_best.x[0],
                abs(sec_best.x[0] - other_T2.x[0]),
                sec_best.x[1],
                abs((sec_best.x[1] - other_T2.x[1]) / 100),
            )

            # Use Gaia parallax and photometry to determine best solution
            final_T, final_T_err, final_g, final_g_err = hot_vs_cold(
                Teff,
                Teff_err,
                logg,
                logg_err,
                Teff2,
                Teff_err2,
                logg2,
                logg_err2,
                parallax,
                phot_g_mean_mag,
                model=self.model_grid,
            )

            lines_s, lines_m, mod_n = fit_func(
                (final_T, 100 * final_g, rv), spec_n, l_crop, models=self.model_grid, mode=1
            )
            spec_w = data[:, 0]
            mod_n[np.isnan(mod_n)] = 0.0
            check_f_spec = data[:, 1][(spec_w > 4500.0) & (spec_w < 4700.0)]
            check_f_model = mod_n[:, 1][(mod_n[:, 0] > 4500.0) & (mod_n[:, 0] < 4700.0)]
            adjust = np.average(check_f_model) / np.average(check_f_spec)    

            # Resample model flux to match spectrum
            resampled_model_flux = np.interp(
                spectrum.wavelength.value,
                mod_n[:, 0] * (rv + 299792.458) / 299792.458,
                mod_n[:, 1] / adjust,
                left=np.nan,
                right=np.nan
            )
            
            # Let's only consider chi-sq between say 3750 - 8000
            chisq_mask = (8000 >= spectrum.wavelength.value) * (spectrum.wavelength.value >= 3750)
            ivar = spectrum.uncertainty.represent_as(InverseVariance).array.flatten()
            pixel_chi_sq = (resampled_model_flux - spectrum.flux.value.flatten())**2 * ivar
            chi_sq = np.nansum(pixel_chi_sq[chisq_mask])
            reduced_chi_sq = (chi_sq / (np.sum(np.isfinite(pixel_chi_sq[chisq_mask])) - 3))

            result = dict(
                snr=spectrum.meta["SNR"][0], # TODO: Not yet handle mwmVisit
                wd_type=wd_type,
                teff=final_T,
                e_teff=final_T_err,
                logg=final_g,
                e_logg=final_g_err,
                v_rel=rv,
                conditioned_on_parallax=parallax,
                conditioned_on_phot_g_mean_mag=phot_g_mean_mag,
                chi_sq=chi_sq,
                reduced_chi_sq=reduced_chi_sq
            )

            log.info(
                f"{self} with {self.input_data_products[0]}: Teff = {result['teff']} +/- {result['e_teff']}, logg = {result['logg']} +/- {result['e_logg']}"
            )

            # Create result row in the database.
            task.create_or_update_outputs(WhiteDwarfOutput, [result])
            
            # Create the astraStar/astraVisit object
            result.update(
                spectral_axis=spectrum.spectral_axis,
                model_flux=np.atleast_2d(resampled_model_flux),
                chi_sq=chi_sq,
                reduced_chi_sq=reduced_chi_sq,
                continuum=adjust,
            )
            create_pipeline_product(
                task, 
                data_product, 
                {
                    get_extname(spectrum, data_product): result
                }
            )

            if self.plot:
                
                import matplotlib.pyplot as plt

                lines_s_o, lines_m_o, mod_n_o = fit_func(
                    (Teff2, logg2, rv), spec_n, l_crop_s, models=self.model_grid, mode=1
                )
                lines_s_final, lines_m_final, mod_n_final = fit_func(
                    (final_T, 100 * final_g, rv),
                    spec_n,
                    l_crop,
                    models=self.model_grid,
                    mode=1,
                )

                fig = plt.figure(figsize=(11.5, 3.75))
                ax1 = plt.subplot2grid((1, 4), (0, 3))
                step = 0
                for i in range(0, 4):  # plots Halpha (i=0) to H6 (i=5)
                    min_p = lines_s[i][:, 0][lines_s[i][:, 1] == np.min(lines_s[i][:, 1])][
                        0
                    ]
                    min_p_o = lines_s_o[i][:, 0][
                        lines_s_o[i][:, 1] == np.min(lines_s_o[i][:, 1])
                    ][0]
                    min_p_final = lines_s_final[i][:, 0][
                        lines_s_final[i][:, 1] == np.min(lines_s_final[i][:, 1])
                    ][0]
                    ax1.plot(lines_s[i][:, 0] - min_p, lines_s[i][:, 1] + step, color="k")
                    ax1.plot(lines_s[i][:, 0] - min_p, lines_m[i] + step, color="r")
                    ax1.plot(lines_s_o[i][:, 0] - min_p_o, lines_m_o[i] + step, color="g")
                    ax1.plot(
                        lines_s_final[i][:, 0] - min_p_final,
                        lines_m_final[i] + step,
                        color="tab:blue",
                    )

                    step += 0.5
                xticks = ax1.xaxis.get_major_ticks()
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

                ax2 = plt.subplot2grid((1, 4), (0, 0), colspan=3)

                # full_spec_w=fitting_scripts.vac_to_air(full_spec[:,0])
                spec_w = data[:, 0]
                ax2.plot(data[:, 0], data[:, 1], color="k")
                # Adjust the flux of models to match the spectrum
                mod_n[np.isnan(mod_n)], mod_n_o[np.isnan(mod_n_o)] = 0.0, 0.0
                check_f_spec = data[:, 1][(spec_w > 4500.0) & (spec_w < 4700.0)]
                check_f_model = mod_n[:, 1][(mod_n[:, 0] > 4500.0) & (mod_n[:, 0] < 4700.0)]
                adjust = np.average(check_f_model) / np.average(check_f_spec)
                ax2.plot(
                    mod_n[:, 0] * (rv + 299792.458) / 299792.458,
                    mod_n[:, 1] / adjust,
                    color="r",
                )
                check_f_model_o = mod_n_o[:, 1][
                    (mod_n_o[:, 0] > 4500.0) & (mod_n_o[:, 0] < 4700.0)
                ]
                adjust_o = np.average(check_f_model_o) / np.average(check_f_spec)

                check_f_model_final = mod_n_final[:, 1][
                    (mod_n_final[:, 0] > 4500.0) & (mod_n_final[:, 0] < 4700.0)
                ]
                adjust_final = np.average(check_f_model_final) / np.average(check_f_spec)

                ax2.plot(
                    mod_n_o[:, 0] * (rv + 299792.458) / 299792.458,
                    mod_n_o[:, 1] / adjust_o,
                    color="g",
                )
                ax2.plot(
                    mod_n_final[:, 0] * (rv + 299792.458) / 299792.458,
                    mod_n_final[:, 1] / adjust_final,
                    color="tab:blue",
                )

                m = ((spectrum.flux.value * spectrum.uncertainty.array) > 3)[0]
                wls = spectrum.wavelength.value[m]
                fls = spectrum.flux.value[0, m]

                ax2.set_xlim(3000, 10_000)
                ax2.set_ylim(0, np.max(mod_n[:, 1] / adjust))

                ax2.set_ylabel(
                    r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1} \AA^{-1}$]", fontsize=12
                )
                ax2.set_xlabel(r"Wavelength $(\AA)$", fontsize=12)
                fig.tight_layout()

                basename = os.path.basename(data_product.path[:-5])
                fig_path = expand_path(f"$MWM_ASTRA/{__version__}/{data_product.kwargs['run2d']}-{data_product.kwargs['apred']}/plots/wd/{basename}.png")
                os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                fig.savefig(fig_path, dpi=300)
                log.info(f"Created figure {fig_path}")

                del fig
                plt.close("all")
