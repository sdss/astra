import numpy as np
import os
import scipy.optimize as op

from astra import log
from astra.base import ExecutableTask, Parameter
from astra.database.astradb import database, Output, TaskOutput, WhiteDwarfOutput
from astra.tools.spectrum import Spectrum1D

from astra.sdss.catalog import get_gaia_dr2_photometry

from astra.contrib.wd.fitting import (
    norm_spectra,
    fit_line,
    fit_func,
    err_func,
    hot_vs_cold,
)


class WhiteDwarfStellarParameters(ExecutableTask):

    """
    Estimate effective temperature and surface gravity for DA-type white dwarfs.
    """

    model_grid = Parameter(default="da2014")
    parallax = Parameter(default=None)
    absolute_G_mag = Parameter(default=None)
    plot = Parameter(default=False)

    def execute(self):

        data_product = self.input_data_products[0]
        source = data_product.sources[0]

        parallax = self.parallax
        absolute_G_mag = self.absolute_G_mag

        if parallax is None or absolute_G_mag is None:
            log.info(f"Retrieving Gaia (DR2) photometry for {source}")
            gaia = get_gaia_dr2_photometry(source.catalogid)

            parallax = parallax or gaia["parallax"]
            absolute_G_mag = absolute_G_mag \
                or (gaia["phot_g_mean_mag"] + 5 * np.log10(gaia["parallax"]/100))

        log.info(f"{self} on {data_product} using parallax = {parallax:.2f} and absolute_G_mag = {absolute_G_mag:.2f}")

        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")

        spectrum = Spectrum1D.read(data_product.path)

        # Get Gaia (absolute) G magnitude and parallax.

        # TODO: some better classifier
        wd_type = "DA"

        wl = spectrum.wavelength.value
        flux = spectrum.flux.value
        flux_err = spectrum.uncertainty.array**-0.5

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
            absolute_G_mag,
            model=self.model_grid,
        )

        snr = flux / flux_err
        result = dict(
            snr=np.mean(snr[snr > 0]),
            wd_type=wd_type,
            teff=final_T,
            u_teff=final_T_err,
            logg=final_g,
            u_logg=final_g_err,
            conditioned_on_parallax=parallax,
            conditioned_on_absolute_G_mag=absolute_G_mag,
        )

        log.info(
            f"{self} with {self.input_data_products[0]}: Teff = {result['teff']} +/- {result['u_teff']}, logg = {result['logg']} +/- {result['u_logg']}"
        )

        # Create result row in the database.
        output, wd_output = self.create_output(WhiteDwarfOutput, result)
        log.info(f"Created database outputs {output} and {wd_output}")

        # TODO: Create a BossStar output file.
        

        if self.plot:

            import matplotlib.pyplot as plt

            lines_s,lines_m,mod_n=fit_func((Teff,logg,rv),spec_n,l_crop,models=self.model_grid,mode=1)
            lines_s_o,lines_m_o,mod_n_o=fit_func((Teff2,logg2,rv),spec_n,l_crop_s,models=self.model_grid, mode=1)
            lines_s_final,lines_m_final,mod_n_final=fit_func((final_T,100*final_g,rv),spec_n,l_crop,models=self.model_grid,mode=1)
            
            fig=plt.figure(figsize=(11.5,3.75))
            ax1 = plt.subplot2grid((1,4), (0, 3))
            step = 0
            for i in range(0,4): # plots Halpha (i=0) to H6 (i=5)
                min_p   = lines_s[i][:,0][lines_s[i][:,1]==np.min(lines_s[i][:,1])][0]
                min_p_o = lines_s_o[i][:,0][lines_s_o[i][:,1]==np.min(lines_s_o[i][:,1])][0]
                min_p_final = lines_s_final[i][:,0][lines_s_final[i][:,1]==np.min(lines_s_final[i][:,1])][0]
                ax1.plot(lines_s[i][:,0]-min_p,lines_s[i][:,1]+step,color='k')
                ax1.plot(lines_s[i][:,0]-min_p,lines_m[i]+step,color='r')
                ax1.plot(lines_s_o[i][:,0]-min_p_o,lines_m_o[i]+step,color='g')
                ax1.plot(lines_s_final[i][:,0]-min_p_final,lines_m_final[i]+step,color='tab:blue')
                
                step+=0.5
            xticks = ax1.xaxis.get_major_ticks()
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

            ax2 = plt.subplot2grid((1,4), (0, 0),colspan=3)

            #full_spec_w=fitting_scripts.vac_to_air(full_spec[:,0])
            spec_w=data[:,0]
            ax2.plot(data[:,0],data[:,1],color='k')
            #Adjust the flux of models to match the spectrum
            mod_n[np.isnan(mod_n)], mod_n_o[np.isnan(mod_n_o)] = 0.0, 0.0
            check_f_spec=data[:,1][(spec_w>4500.) & (spec_w<4700.)]
            check_f_model=mod_n[:,1][(mod_n[:,0]>4500.) & (mod_n[:,0]<4700.)]
            adjust=np.average(check_f_model)/np.average(check_f_spec)
            ax2.plot(mod_n[:,0]*(rv+299792.458)/299792.458,mod_n[:,1]/adjust,color='r')
            check_f_model_o=mod_n_o[:,1][(mod_n_o[:,0]>4500.) & (mod_n_o[:,0]<4700.)]
            adjust_o=np.average(check_f_model_o)/np.average(check_f_spec)

            check_f_model_final=mod_n_final[:,1][(mod_n_final[:,0]>4500.) & (mod_n_final[:,0]<4700.)]
            adjust_final=np.average(check_f_model_final)/np.average(check_f_spec)

            ax2.plot(mod_n_o[:,0]*(rv+299792.458)/299792.458,mod_n_o[:,1]/adjust_o,color='g')
            ax2.plot(mod_n_final[:,0]*(rv+299792.458)/299792.458,mod_n_final[:,1]/adjust_final,color='tab:blue')

            m = ((spectrum.flux.value * spectrum.uncertainty.array) > 3)[0]
            wls = spectrum.wavelength.value[m]
            fls = spectrum.flux.value[0, m]

            ax2.set_xlim(3000, 10_000)
            ax2.set_ylim(0, 1.2 * np.percentile(fls, 99.9))

            ax2.set_ylabel(r'F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1} \AA^{-1}$]',fontsize=12)
            ax2.set_xlabel(r'Wavelength $(\AA)$',fontsize=12)

            raise a 