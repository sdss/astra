import numpy as np
import pandas as pd
import os
from scipy.integrate import simps
from scipy import interpolate
from astropy.convolution import convolve, Gaussian1DKernel
from astra.utils import expand_path

data_dir = expand_path("$MWM_ASTRA/component_data/wd/")


def norm_models(quick=True, model="da2014", testing=False):
    """Import Normalised WD Models
    Optional arguments:
        quick=True   : Use presaved model array. Check is up to date
        model='da2014': Which model grid to use: List shown below in mode_list
        testing=False      : plot testing image
    Return [model_list,model_param,orig_model_wave,orig_model_flux,tck_model,r_model]
    """
    if quick:  # Use preloaded tables
        model_list = [
            "da2014",
            "pier",
            "pier3D",
            "pier3D_smooth",
            "pier_rad",
            "pier1D",
            "pier_smooth",
            "pier_rad_smooth",
            "pier_rad_fullres",
            "pier_fullres",
            "pier_IR",
        ]
        if model not in model_list:
            raise wdfitError('Unknown "model" in norm_models')
        if model == "pier_IR":
            fn, d = "/wdfit.pier.lst", "/WDModels_Koester.pier_npy/"
        else:
            fn, d = "/wdfit." + model + ".lst", "/WDModels_Koester." + model + "_npy/"
        model_list = np.loadtxt(data_dir + fn, usecols=[0], dtype=np.string_).astype(
            str
        )
        model_param = np.loadtxt(data_dir + fn, usecols=[1, 2])
        m_spec = np.load(data_dir + d + model_list[0])
        m_wave = m_spec[:, 0]
        if model == "pier_IR":
            out_m_wave = m_wave[(m_wave >= 3400) & (m_wave <= 22000)]  # 13000 22000
        else:
            out_m_wave = m_wave[(m_wave >= 3400) & (m_wave <= 13000)]  # 13000 22000
        norm_m_flux = np.load(data_dir + "/norm_m_flux." + model + ".npy")
        #
        if out_m_wave.shape[0] != norm_m_flux.shape[1]:
            raise wdfitError("l and f arrays not correct shape in norm_models")
    # Calculate
    else:
        # Load models from models()
        model_list, model_param, m_wave, m_flux, r_model = models(
            quick=quick, model=model
        )
        # Range over which DA is purely continuum
        norm_range = np.loadtxt(data_dir + "/wide_norm_range.dat", usecols=[0, 1])
        n_range_s = np.loadtxt(
            data_dir + "/wide_norm_range.dat", usecols=[2], dtype="string"
        )
        # for each line, for each model, hold l and f of continuum
        cont_l = np.empty([len(m_flux), len(norm_range)])
        cont_f = np.empty([len(m_flux), len(norm_range)])
        # for each zone

        for j in range(len(norm_range)):
            if (norm_range[j, 0] < m_wave.max()) & (norm_range[j, 1] > m_wave.min()):
                # crop
                _f = m_flux.transpose()[
                    (m_wave >= norm_range[j, 0]) & (m_wave <= norm_range[j, 1])
                ].transpose()
                _l = m_wave[(m_wave >= norm_range[j, 0]) & (m_wave <= norm_range[j, 1])]

                # interpolate region
                print(norm_range[j, 0], norm_range[j, 1])
                print(np.size(_l))

                tck = interpolate.interp1d(_l, _f, kind="cubic")
                # interpolate onto 10* resolution
                l = np.linspace(_l.min(), _l.max(), (len(_l) - 1) * 10 + 1)
                f = tck(l)
                # print f
                # find maxima and save
                if n_range_s[j] == "P":
                    for k in range(len(f)):
                        cont_l[k, j] = l[f[k] == f[k].max()][0]
                        cont_f[k, j] = f[k].max()
                # find mean and save
                elif n_range_s[j] == "M":
                    for k in range(len(f)):
                        cont_l[k, j] = np.mean(l)
                        cont_f[k, j] = np.mean(f[k])
                else:
                    print("Unknown n_range_s, ignoring")
        # Continuum
        if (norm_range.min() > 3400) & (norm_range.max() < 13000):
            out_m_wave = m_wave[(m_wave >= 3400) & (m_wave <= 13000)]
        else:
            raise wdfitError("Normalised models cropped to too small a region")
        cont_m_flux = np.empty([len(m_flux), len(out_m_wave)])
        for i in range(len(m_flux)):
            # not suitable for higher order fitting
            tck = interpolate.splrep(
                cont_l[i], cont_f[i], t=[3885, 4340, 4900, 6460], k=3
            )
            cont_m_flux[i] = interpolate.splev(out_m_wave, tck)
        # Normalised flux
        norm_m_flux = (
            m_flux.transpose()[(m_wave >= 3400) & (m_wave <= 13000)].transpose()
            / cont_m_flux
        )
        np.save(data_dir + "/norm_m_flux." + model + ".npy", norm_m_flux)
        #
        # testing
        if testing:
            import matplotlit.pyplot as plt

            def p():
                plt.figure(figsize=(7, 9))
                ax1 = pl.subplot(211)
                llst = [3885, 4340, 4900, 6460]
                for num in llst:
                    plt.axvline(num, color="g", zorder=1)
                plt.plot(m_wave, m_flux[i], color="grey", lw=0.8, zorder=2)
                plt.plot(out_m_wave, cont_m_flux[i], "b-", zorder=3)
                plt.scatter(
                    cont_l[i], cont_f[i], edgecolors="r", facecolors="none", zorder=20
                )
                ax2 = plt.subplot(212, sharex=ax1)
                plt.axhline([1], color="g")
                plt.plot(out_m_wave, norm_m_flux[i], "b-")
                plt.ylim([0, 2])
                plt.xlim([3400, 13000])
                plt.show()
                return

            for i in np.where(model_param[:, 1] == 8.0)[0][::8]:
                p()
    return [out_m_wave, norm_m_flux, model_param]


def models(quick=True, quiet=True, band="sdss_r", model="da2014"):
    """Import WD Models
    Optional:
        quick=True   : Use presaved model array. Check is up to date
        quiet=True   : verbose
        band='sdss_r': which mag band to calculate normalisation over (MEAN, not folded)
        model: Which model grid to use
    Return [model_list,model_param,orig_model_wave,orig_model_flux,tck_model,r_model]
    """
    model_list = [
        "da2014",
        "pier",
        "pier3D",
        "pier3D_smooth",
        "pier_rad",
        "pier1D",
        "pier_smooth",
        "pier_rad_smooth",
        "pier_rad_fullres",
        "pier_fullres",
    ]
    if model not in model_list:
        raise wdfitError('Unknown "model" in models')
    else:
        fn, d = "/wdfit." + model + ".lst", "/WDModels_Koester." + model + "/"
    # Load in table of all models
    model_list = np.loadtxt(data_dir + fn, usecols=[0], dtype=np.string_).astype(str)
    model_param = np.loadtxt(data_dir + fn, usecols=[1, 2])
    orig_model_wave = np.loadtxt(data_dir + d + model_list[0], usecols=[0])
    #
    if not quiet:
        print("Loading Models")
    if quick:
        orig_model_flux = np.load(data_dir + "/orig_model_flux." + model + ".npy")
        if orig_model_wave.shape[0] != orig_model_flux.shape[1]:
            raise wdfitError("l and f arrays not correct shape in models")
    else:
        orig_model_flux = np.empty([len(model_list), len(orig_model_wave)])
        if model != "db":
            for i in range(len(model_list)):
                if not quiet:
                    print(i)
                print(data_dir + d + model_list[i])
                orig_model_flux[i] = np.loadtxt(
                    data_dir + d + model_list[i], usecols=[1]
                )
        else:
            from jg import spectra as _s

            for i in range(len(model_list)):
                if not quiet:
                    print(i)
                tmp = _s.spectra(fn=data_dir + d + model_list[i], usecols=[0, 1])
                # Not uniform wavelength grid argh!
                tmp.interpolate(orig_model_wave, kind="linear", save_res=True)
                orig_model_flux[i] = tmp.f()
        np.save(data_dir + "/orig_model_flux." + model + ".npy", orig_model_flux)
    # Linearly interpolate Model onto Spectra points
    tck_model = interpolate.interp1d(orig_model_wave, orig_model_flux, kind="linear")
    # Only calculate r model once
    band_lims = _band_limits(band)
    r_model = np.mean(
        orig_model_flux.transpose()[
            ((orig_model_wave >= band_lims[0]) & (orig_model_wave <= band_lims[1]))
        ].transpose(),
        axis=1,
    )
    #
    print(orig_model_wave)
    print(orig_model_flux)
    return [model_list, model_param, orig_model_wave, orig_model_flux, r_model]


def corr3d(temperature, gravity, ml2a=0.8, testing=False):
    """Determines the 3D correction (Tremblay et al. 2013, 559, A104)
          from their atmospheric parameters
    Optional, measure in parsecs
    ML2/alpha = 0.8 is implemented
    """
    temperature, gravity = float(temperature), float(gravity)
    if temperature > 14500.0 or temperature < 6000.0 or gravity > 9.0 or gravity < 7.0:
        # print "no correction"
        return 0.0, 0.0
    if ml2a == 0.8:
        print("NEED TO CHANGE PATH AND GET DATA.")
        teff_corr = pylab.csv2rec("/home/nicola/DA_fitting/3D_corr/teff_corr.csv")
        logg_corr = pylab.csv2rec("/home/nicola/DA_fitting/3D_corr/logg_corr.csv")
        teff, logg = teff_corr["teff"], teff_corr["logg"]
        # indentifying ranges Teff_corr#
        t1, t2 = np.max(teff[teff <= temperature]), np.min(teff[teff >= temperature])
        g1, g2 = np.max(logg[logg <= gravity]), np.min(logg[logg >= gravity])
        if t1 != t2:
            t1, t2 = float(t1), float(t2)
            t = (temperature - t1) / (t2 - t1)
        else:
            t = 0.0
        if np.isnan(t) == True:
            t = 0.0
        if g1 != g2:
            g1, g2 = float(g1), float(g2)
            g = (gravity - g1) / (g2 - g1)
        else:
            g = 0
        if np.isnan(g) == True:
            g = 0
        m11 = teff_corr[np.logical_and(teff == t1, logg == g1)]["teff_corr"][0]
        m12 = teff_corr[np.logical_and(teff == t1, logg == g2)]["teff_corr"][0]
        m21 = teff_corr[np.logical_and(teff == t2, logg == g1)]["teff_corr"][0]
        m22 = teff_corr[np.logical_and(teff == t2, logg == g2)]["teff_corr"][0]
        teff_3dcorr = (
            (1 - t) * (1 - g) * m11
            + t * (1 - g) * m21
            + t * g * m22
            + (1 - t) * g * m12
        )

        m11 = logg_corr[np.logical_and(teff == t1, logg == g1)]["logg_corr"][0]
        m12 = logg_corr[np.logical_and(teff == t1, logg == g2)]["logg_corr"][0]
        m21 = logg_corr[np.logical_and(teff == t2, logg == g1)]["logg_corr"][0]
        m22 = logg_corr[np.logical_and(teff == t2, logg == g2)]["logg_corr"][0]
        logg_3dcorr = (
            (1 - t) * (1 - g) * m11
            + t * (1 - g) * m21
            + t * g * m22
            + (1 - t) * g * m12
        )

        if testing:
            plt.fig = figure(figsize=(7, 2.5))
            plt.fig.subplots_adjust(
                left=0.10, bottom=0.15, right=0.98, top=0.98, wspace=0.35
            )
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            tmp_g = [7.0, 7.5, 8.0, 8.5, 9.0]
            tmp_s = ["b-", "r-", "g-", "c-", "m-"]
            axes = [ax1, ax2]
            tmp_t = ["teff_corr", "logg_corr"]
            for i in range(len(axes)):
                for j in range(len(tmp_g)):
                    axes[i].plot(
                        teff[where(logg == tmp_g[j])],
                        teff_corr[tmp_t[i]][where(logg == tmp_g[j])],
                        tmp_s[j],
                        label="logg = %.1f" % (tmp_g[j]),
                        lw=0.5,
                    )
                axes[i].legend(numpoints=1, ncol=1, loc=3, fontsize=7)
                axes[i]._set_xlabel("Teff")
                axes[i]._set_ylabel(tmp_t[i])
                axes[i].set_xlim([6000, 14500])
            ax1.plot(temperature, teff_3dcorr, "ro", ms=2)
            ax2.plot(temperature, logg_3dcorr, "ro", ms=2)
            plt.show()
            plt.close()
        return np.round(teff_3dcorr, 0), np.round(logg_3dcorr, 2)
    elif ml2a == 0.8:
        print("to be implemented")
    elif ml2a == 0.7:
        print("to be implemented")


def vac_to_air(wavin):
    """Converts spectra from air wavelength to vacuum to compare to models"""
    return wavin / (1.0 + 2.735182e-4 + 131.4182 / wavin**2 + 2.76249e8 / wavin**4)


def air_to_vac(wlum):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006
    """
    return (1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)) * wlum


def _band_limits(band):
    """give mag band eg "sdss_r" & return outer limits for use in models etc"""
    mag = np.loadtxt(data_dir + "/sm/" + band + ".dat")
    mag = mag[mag[:, 1] > 0.05]
    return [mag[:, 0].min(), mag[:, 0].max()]


def norm_spectra(spectra, type="DA", add_infinity=False):
    """
    Normalised spectra by DA or DB WD continuum regions
    spectra of form array([wave,flux,error]) (err not necessary so works on models)
    only works on SDSS spectra region
    Optional:
        EDIT n_range_s to change whether region[j] is fitted for a peak or mean'd
        add_infinity=False : add a spline point at [inf,0]
    returns spectra, cont_flux
    """
    if type == "DA":
        # start_n=np.array([3770.,3796.,3835.,3895.,3995.,4130.,4490.,4620.,5070.,5200.,
        #              6000.,7000.,7550.,8400.])
        # end_n=np.array([3795.,3830.,3885.,3960.,4075.,4290.,4570.,4670.,5100.,5300.,
        #            6100.,7050.,7600.,8450.])
        # n_range_s=np.array(['P','P','P','P','P','P','M','M','M','M','M','M','M','M'])
        start_n = np.array(
            [
                3600,
                3770.0,
                3796.0,
                3835.0,
                3895.0,
                3995.0,
                4160,
                4490.0,
                4620.0,
                5070.0,
                5200.0,
                6000.0,
                7000.0,
                7550.0,
                8400.0,
            ]
        )
        end_n = np.array(
            [
                3650,
                3795.0,
                3830.0,
                3885.0,
                3960.0,
                4075.0,
                4210,
                4570.0,
                4670.0,
                5100.0,
                5300.0,
                6100.0,
                7050.0,
                7600.0,
                8450.0,
            ]
        )
        n_range_s = np.array(
            ["M", "P", "P", "P", "P", "P", "M", "M", "M", "M", "M", "M", "M", "M", "M"]
        )
    elif type == "DB":
        # start_n=np.array([3500,3650.,3755.,3920.,4150.,4225.,4665.,5060,5150.,5500.,6100.,6800.,7600.,8300.])#4095.
        # end_n=np.array([3525,3675.,3765.,3940.,4200,4250.,4690.,5080,5300.,5700.,6300.,7000.,7800.,8700.])#,4105.
        # n_range_s=np.array(['M','P','P','M','P','P','M','M','M','M','M','M','M','M','M'])#'P',
        # start_n=np.array([3500,3650.,3750.,3880.,4075.,4220.,4665.,4940,5060,5300.,5500.,6100.,6800.,7600.,8300.])#4095.
        # end_n=np.array([3525,3675.,3780.,3920.,4100.,4250.,4690.,5000,5080,5400.,5700.,6300.,7000.,7800.,8700.])#,4105.
        start_n = np.array(
            [
                3370,
                3500,
                3650.0,
                3750.0,
                3880.0,
                4075.0,
                4220.0,
                4630.0,
                4750,
                5060,
                5300.0,
                5500.0,
                6100.0,
                6800.0,
                7600.0,
                8300.0,
            ]
        )  # 4095.
        end_n = np.array(
            [
                3430,
                3525,
                3675.0,
                3780.0,
                3920.0,
                4100.0,
                4250.0,
                4670.0,
                4850,
                5080,
                5400.0,
                5700.0,
                6300.0,
                7000.0,
                7800.0,
                8700.0,
            ]
        )  # ,4105.
        n_range_s = np.array(
            [
                "M",
                "M",
                "P",
                "P",
                "P",
                "P",
                "M",
                "P",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
            ]
        )  #'P',

    elif type == "IR":
        # start_n=np.array([6900,7200,7400,7800,8400,8800,9400,10300,10600,11280,11700,12050,12420,13100,13280,13800,14200,15000,15500,16000])#4095.
        # end_n=np.array([6950, 7250,7450,7850,8450, 8850,9420,10350,10650,11320,11750,12100,12480,13150,13310,13850,14250,15050,15550,16050])#,4105.
        # n_range_s=np.array(['M','M','M','M','M','M','P','P','M','M','P','M','M','M','M','M','M','M','M','M','M','M','M'])#'P',
        # start_n=np.array([6800,7050,7400,7800,8300,8800,9050,9350,9900,10300,10750,11200,11800,12300,13000,14600,15500,16000])#4095.
        # end_n=np.array([6950, 7250,7450,7850,8500, 8850,9150,9450,9950,10500,10850,11300,12050,12400,13200,14800,15750,16050])#,4105.
        # n_range_s=np.array(['M','M','M','M','M','M','M','M','M','M','P','M','M','M','M','P','M','M','M'])#'P',
        start_n = np.array(
            [
                6800,
                7050,
                7400,
                7800,
                8300,
                8800,
                9340,
                9800,
                10300,
                11280,
                11800,
                13600,
                14200,
                15500,
                16000,
            ]
        )  # 4095. WF3
        end_n = np.array(
            [
                6950,
                7250,
                7450,
                7850,
                8500,
                8850,
                9400,
                9840,
                10650,
                11750,
                12150,
                13950,
                15050,
                15750,
                16250,
            ]
        )  # ,4105. WF3
        ##start_n=np.array([6800,7050,7400,7800,8300,8800,9000,9350,9700,10300,11280,11800,12400,13200,14200,15500,16000])#4095. Xshooter
        # end_n=np.array([6950, 7250,7450,7850,8500, 8850,9200,9400,9800,10650,11750,12000,12500,13300,15050,15750,16250])#,4105. Xshooter
        n_range_s = np.array(
            [
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
                "M",
            ]
        )
    else:
        print("Please provide a WD type: DA, DB or IR")

    if len(spectra[0]) > 2:
        snr = np.zeros([len(start_n), 3])
        spectra[:, 2][spectra[:, 2] == 0.0] = spectra[:, 2].max()
    else:
        snr = np.zeros([len(start_n), 2])
    wav = spectra[:, 0]
    for j in range(len(start_n)):
        if (start_n[j] < wav.max()) & (end_n[j] > wav.min()):
            _s = spectra[(wav >= start_n[j]) & (wav <= end_n[j])]
            _w = _s[:, 0]
            # Avoids gappy spectra
            k = 3  # Check if there are more points than 3
            if len(_s) > k:
                # interpolate onto 10* resolution
                l = np.linspace(_w.min(), _w.max(), (len(_s) - 1) * 10 + 1)
                if len(spectra[0]) > 2:
                    tck = interpolate.splrep(_w, _s[:, 1], w=1 / _s[:, 2], s=1000)
                    # median errors for max/mid point
                    snr[j, 2] = np.median(_s[:, 2]) / np.sqrt(len(_w))
                else:
                    tck = interpolate.splrep(_w, _s[:, 1], s=0.0)
                f = interpolate.splev(l, tck)
                # find maxima and save
                if n_range_s[j] == "P":
                    snr[j, 0], snr[j, 1] = l[f == f.max()][0], f.max()
                # find mean and save
                elif n_range_s[j] == "M":
                    snr[j, 0:2] = np.mean(l), np.mean(f)
                else:
                    print("Unknown n_range_s, ignoring")
    snr = snr[snr[:, 0] != 0]
    # t parameter chosen by eye. Position of knots.
    if type == "DA":
        if snr[:, 0].max() < 6460:
            knots = [3000, 4900, 4100, 4340, 4500, 4860, int(snr[:, 0].max() - 5)]
        else:
            knots = [3885, 4340, 4900, 6460]
        if snr[:, 0].min() > 3885:
            print(
                "Warning: knots used for spline norm unsuitable for high order fitting"
            )
            knots = knots[1:]
        if (snr[:, 0].min() > 4340) or (snr[:, 0].max() < 4901):
            knots = None  # 'Warning: knots used probably bad'
    elif type == "DB":
        if snr[:, 0].max() < 6460:
            knots = [3000, 4900, 4100, 4340, 4860, int(snr[:, 0].max() - 5)]
        else:
            knots = [3885, 4200, 4340, 4900, 6460]
        if snr[:, 0].min() > 3885:
            print(
                "Warning: knots used for spline norm unsuitable for high order fitting"
            )
            knots = knots[1:]
        if (snr[:, 0].min() > 4340) or (snr[:, 0].max() < 4901):
            knots = None  # 'Warning: knots used probably bad'

    elif type == "IR":

        knots = [5800, 9000, 10100, 11500, 12000, 13750, 15000, 16000]
    #

    if add_infinity:  # Adds points at inf & 0 for spline to fit to err = mean(spec err)
        if snr.shape[1] > 2:
            mean_snr = np.mean(snr[:, 2])
            snr = np.vstack([snr, np.array([90000.0, 0.0, mean_snr])])
            snr = np.vstack([snr, np.array([100000.0, 0.0, mean_snr])])
        else:
            snr = np.vstack([snr, np.array([90000.0, 0.0])])
            snr = np.vstack([snr, np.array([100000.0, 0.0])])
    try:  # weight by errors
        if len(spectra[0]) > 2:
            tck = interpolate.splrep(
                snr[:, 0], snr[:, 1], w=1 / snr[:, 2], t=knots, k=2
            )
        else:
            tck = interpolate.splrep(snr[:, 0], snr[:, 1], t=knots, k=2)
    except ValueError:
        knots = None
        if len(spectra[0]) > 2:
            tck = interpolate.splrep(
                snr[:, 0], snr[:, 1], w=1 / snr[:, 2], t=knots, k=2
            )
        else:
            tck = interpolate.splrep(snr[:, 0], snr[:, 1], t=knots, k=2)
    cont_flux = interpolate.splev(wav, tck).reshape(wav.size, 1)
    spectra_ret = np.copy(spectra)
    spectra_ret[:, 1:] = spectra_ret[:, 1:] / cont_flux
    # import matplotlib.pyplot as plt
    # print(spectra_ret)
    # plt.plot(spectra[:,0],spectra[:,1],zorder=1)
    # plt.plot(spectra[:,0],cont_flux,zorder=2)
    # plt.scatter(snr[:,0],snr[:,1],c="r",zorder=3)
    # plt.plot(spectra_ret[:,0],spectra_ret[:,1])
    # plt.show()
    return spectra_ret, cont_flux


def interpolating_model_DA(temp, grav, m_type="da2014"):
    """Interpolate model atmospheres given an input Teff and logg
    models are saved as a numpy array to increase speed"""
    # PARAMETERS #
    dir_models = data_dir + "/WDModels_Koester." + m_type + "_npy/"
    if m_type == "pier" or m_type == "pier_fullres":
        teff = np.array(
            [
                1500.0,
                1750.0,
                2000.0,
                2250.0,
                2500.0,
                2750.0,
                3000.0,
                3250.0,
                3500.0,
                3750.0,
                4000.0,
                4250.0,
                4500.0,
                4750.0,
                5000.0,
                5250.0,
                5500.0,
                6000.0,
                6500.0,
                7000.0,
                7500.0,
                8000.0,
                8500.0,
                9000.0,
                9500.0,
                10000.0,
                10500.0,
                11000.0,
                11500.0,
                12000.0,
                12500.0,
                13000.0,
                13500.0,
                14000.0,
                14500.0,
                15000.0,
                15500.0,
                16000.0,
                16500.0,
                17000.0,
                20000.0,
                25000.0,
                30000.0,
                35000.0,
                40000.0,
                45000.0,
                50000.0,
                55000.0,
                60000.0,
                65000.0,
                70000.0,
                75000.0,
                80000.0,
                85000.0,
                90000.0,
            ]
        )
        logg = np.array([6.50, 7.00, 7.50, 7.75, 8.00, 8.25, 8.50, 9.00, 9.50])

    elif m_type == "da2014":
        teff = np.array(
            [
                6000.0,
                6250.0,
                6500.0,
                6750.0,
                7000.0,
                7250.0,
                7500.0,
                7750.0,
                8000.0,
                8250.0,
                8500.0,
                8750.0,
                9000.0,
                9250.0,
                9500.0,
                9750.0,
                10000.0,
                10100.0,
                10200.0,
                10250.0,
                10300.0,
                10400.0,
                10500.0,
                10600.0,
                10700.0,
                10750.0,
                10800.0,
                10900.0,
                11000.0,
                11100.0,
                11200.0,
                11250.0,
                11300.0,
                11400.0,
                11500.0,
                11600.0,
                11700.0,
                11750.0,
                11800.0,
                11900.0,
                12000.0,
                12100.0,
                12200.0,
                12250.0,
                12300.0,
                12400.0,
                12500.0,
                12600.0,
                12700.0,
                12750.0,
                12800.0,
                12900.0,
                13000.0,
                13500.0,
                14000.0,
                14250.0,
                14500.0,
                14750.0,
                15000.0,
                15250.0,
                15500.0,
                15750.0,
                16000.0,
                16250.0,
                16500.0,
                16750.0,
                17000.0,
                17250.0,
                17500.0,
                17750.0,
                18000.0,
                18250.0,
                18500.0,
                18750.0,
                19000.0,
                19250.0,
                19500.0,
                19750.0,
                20000.0,
                21000.0,
                22000.0,
                23000.0,
                24000.0,
                25000.0,
                26000.0,
                27000.0,
                28000.0,
                29000.0,
                30000.0,
                35000.0,
                40000.0,
                45000.0,
                50000.0,
                55000.0,
                60000.0,
                65000.0,
                70000.0,
                75000.0,
                80000.0,
                90000.0,
                100000.0,
            ]
        )
        logg = np.array(
            [
                4.00,
                4.25,
                4.50,
                4.75,
                5.00,
                5.25,
                5.50,
                5.75,
                6.00,
                6.25,
                6.50,
                6.75,
                7.00,
                7.25,
                7.50,
                7.75,
                8.00,
                8.25,
                8.50,
                8.75,
                9.00,
                9.25,
                9.50,
            ]
        )
    elif (m_type == "pier") & (
        temp < 1500.0 or temp > 90000.0 or grav < 6.5 or grav > 9.0
    ):
        return [], []
    elif (m_type == "da2014") & (
        temp < 6000.0 or temp > 100000.0 or grav < 4.0 or grav > 9.5
    ):
        return [], []
    # INTERPOLATION #
    g1, g2 = np.max(logg[logg <= grav]), np.min(logg[logg >= grav])
    if g1 != g2:
        g = (grav - g1) / (g2 - g1)
    else:
        g = 0
    t1, t2 = np.max(teff[teff <= temp]), np.min(teff[teff >= temp])
    if t1 != t2:
        t = (temp - t1) / (t2 - t1)
    else:
        t = 0
    if m_type == "da2014":
        models = [
            "da%06d_%d_2.7.npy" % (i, j * 100) for i in [t1, t2] for j in [g1, g2]
        ]
    else:
        models = ["WD_%.2f_%d.0.npy" % (j, i) for i in [t1, t2] for j in [g1, g2]]
    try:
        m11, m12 = np.load(dir_models + models[0]), np.load(dir_models + models[1])
        m21, m22 = np.load(dir_models + models[2]), np.load(dir_models + models[3])
        flux_i = (
            (1 - t) * (1 - g) * m11[:, 1]
            + t * (1 - g) * m21[:, 1]
            + t * g * m22[:, 1]
            + (1 - t) * g * m12[:, 1]
        )
        return np.dstack((m11[:, 0], flux_i))[0]
    except:
        return [], []


def fit_line(_sn, l_crop, model_in=None, quick=True, model="sdss"):
    """
    Use norm models - can pass model_in from norm_models() with many spectra
    Input _sn, l_crop <- Normalised spectrum & a cropped line list
    Optional:
        model_in=None   : Given model array
        quick=True      : Use presaved model array. Check is up to date
        model='da2014'  : 'da2014' or 'pier'
    Calc and return chi2, list of arrays of spectra, and scaled models at lines
    """
    # load normalised models and linearly interp models onto spectrum wave
    if model_in == None:
        m_wave, m_flux_n, m_param = norm_models(quick=quick, model=model)
    else:
        m_wave, m_flux_n, m_param = model_in
    sn_w = _sn[:, 0]

    m_flux_n_i = interpolate.interp1d(m_wave, m_flux_n, kind="linear")(sn_w)
    # Crops models and spectra in a line region, renorms models, calculates chi2
    tmp_lines_m, lines_s, l_chi2 = [], [], []
    for i in range(len(l_crop)):
        l_c0, l_c1 = l_crop[i, 0], l_crop[i, 1]
        l_m = m_flux_n_i.transpose()[(sn_w >= l_c0) & (sn_w <= l_c1)].transpose()
        l_s = _sn[(sn_w >= l_c0) & (sn_w <= l_c1)]
        l_m = l_m * np.sum(l_s[:, 1]) / np.sum(l_m, axis=1).reshape([len(l_m), 1])
        l_chi2.append(np.sum(((l_s[:, 1] - l_m) / l_s[:, 2]) ** 2, axis=1))
        tmp_lines_m.append(l_m)
        lines_s.append(l_s)
    # mean chi2 over lines and stores best model lines for output
    lines_chi2, lines_m = np.sum(np.array(l_chi2), axis=0), []
    # print("boom",lines_chi2)
    is_best = lines_chi2 == lines_chi2.min()
    for i in range(len(l_crop)):
        lines_m.append(tmp_lines_m[i][is_best][0])
    best_TL = m_param[is_best][0]
    return lines_s, lines_m, best_TL, m_param, lines_chi2


def tmp_func(_T, _g, _rv, _sn, _l, _m):
    c = 299792.458  # Speed of light in km/s
    model = interpolating_model_DA(_T, (_g / 100), m_type=_m)
    # model=convolve_gaussian_R(model, 100)
    try:
        norm_model, m_cont_flux = norm_spectra(model)
    except:
        print("Could not load the model")
        return 1
    else:
        # interpolate normalised model and spectra onto same wavelength scale
        m_wave_n, m_flux_n, sn_w = (
            norm_model[:, 0] * (_rv + c) / c,
            norm_model[:, 1],
            _sn[:, 0],
        )
        if np.min(m_wave_n) > np.min(sn_w) or np.max(m_wave_n) < np.max(sn_w):
            print(
                _T, _g, np.min(m_wave_n), np.min(sn_w), np.max(m_wave_n), np.max(sn_w)
            )
            import matplotlib.pyplot as plt

            plt.plot(sn_w, _sn[:, 1], zorder=1)
            plt.plot(m_wave_n, m_flux_n, zorder=2)
            plt.show()
        m_flux_n_i = interpolate.interp1d(m_wave_n, m_flux_n, kind="linear")(sn_w)
        # Initialise: normalised models and spectra in line region, and chi2
        lines_m, lines_s, sum_l_chi2 = [], [], 0

        for i in range(len(_l)):
            # Crop model and spec to line
            l_c0, l_c1 = _l[i, 0], _l[i, 1]
            l_m = m_flux_n_i.transpose()[(sn_w >= l_c0) & (sn_w <= l_c1)].transpose()
            l_s = _sn[(sn_w >= l_c0) & (sn_w <= l_c1)]
            # renormalise models to spectra in line region & calculate chi2+sum
            l_m = l_m * np.sum(l_s[:, 1]) / np.sum(l_m)
            # l_m = l_m*np.sum(l_m)/np.sum(l_m)
            # l_s[:,1]= l_s[:,1]*np.sum(l_s[:,1])/np.sum(l_s[:,1])
            # if _T<15000:
            #   sum_l_chi2 += np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)/(np.size(l_s[:,1]))
            # else:
            sum_l_chi2 += np.sum(((l_s[:, 1] - l_m) / l_s[:, 2]) ** 2)
            # sum_l_chi2.append(np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)/(np.size(l_s[:,1])))
            lines_m.append(l_m), lines_s.append(l_s)
        return lines_s, lines_m, model, sum_l_chi2


def fit_func(x, specn, lcrop, models="da2014", mode=0):
    """Requires: x - initial guess of T, g, and rv
    specn/lcrop - normalised spectrum / list of cropped lines to fit
    mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model"""

    tmp = tmp_func(x[0], x[1], x[2], specn, lcrop, models)
    if tmp == 1:
        return 1e30
    elif mode == 0:
        return tmp[3]  # this is the quantity that gets minimized
    elif mode == 1:
        return tmp[0], tmp[1], tmp[2]  # ,tmp[3]


def err_func(x, rv, valore, specn, lcrop, models="da2014"):
    """Script finds errors by minimising function at chi+1 rather than chi
    Requires: x; rv - initial guess of T, g; rv
    valore - the chi value of the best fit
    specn/lcrop - normalised spectrum / list of cropped lines to fit"""
    if models == "pier_IR":
        tmp = tmp_func_IR(x[0], x[1], rv, specn, lcrop, models)
    else:
        tmp = tmp_func(x[0], x[1], rv, specn, lcrop, models)
    if tmp != 1:
        return abs(tmp[3] - (valore + 1.0))  # this is quantity that gets minimized
    else:
        return 1e30


def degrade_spec(spec, res=3.0):  # 5.5 3.
    sig = res / 2.355
    wave, flux = spec[:, 0], spec[:, 1]
    gauss_kernel = Gaussian1DKernel(sig)  # ,x_size=int(21*sig),mode='linear_interp')
    smooth_flux = convolve(flux, gauss_kernel)
    # new_wave=np.arange(wave.min(),wave.max(),(res/2))
    # new_flux= interpolate.interp1d(wave,smooth_flux,kind='linear')(new_wave)
    # return(np.dstack((new_wave,new_flux))[0])
    return np.dstack((wave, smooth_flux))[0]


def convolve_gaussian(spec, FWHM):
    """
    Convolve spectrum with a Gaussian with FWHM by oversampling and
    using an FFT approach. Wavelengths are assumed to be sorted,
    but uniform spacing is not required. Will cause wrap-around at
    the end of the spectrum.
    """
    sigma = FWHM / 2.355
    x = spec[:, 0]
    y = spec[:, 1]

    def next_pow_2(N_in):
        N_out = 1
        while N_out < N_in:
            N_out *= 2
        return N_out

    # oversample data by at least factor 10 (up to 20).
    xi = np.linspace(x[0], x[-1], next_pow_2(10 * len(x)))
    yi = interpolate.interp1d(x, y)(xi)

    yg = np.exp(-0.5 * ((xi - x[0]) / sigma) ** 2)  # half gaussian
    yg += yg[::-1]
    yg /= np.sum(yg)  # Norm kernel

    yiF = np.fft.fft(yi)
    ygF = np.fft.fft(yg)
    yic = np.fft.ifft(yiF * ygF).real
    new_spec = np.stack((x, interpolate.interp1d(xi, yic)(x)), axis=-1)
    return new_spec


#


def convolve_gaussian_R(spec, R):
    """
    Similar to convolve_gaussian, but convolves to a specified resolution
    rather than a specfied FWHM. Essentially this amounts to convolving
    along a log-uniform x-axis instead.
    """
    x = spec[:, 0]
    y = spec[:, 1]
    in_spec = np.stack((np.log(x), y), axis=-1)
    new_tmp = convolve_gaussian(in_spec, 1.0 / R)
    new_spec = np.stack((x, new_tmp[:, 1]), axis=-1)
    return new_spec


def logg_from_Teff_R(Teff, R):
    MGRID = pd.read_csv(data_dir + "/CO_thickH_processed.csv")
    logT = np.log10(Teff)
    logR = np.log10(R)
    logg = interpolate.griddata(
        (MGRID["logT"], MGRID["logR"]), MGRID["logg"], (logT, logR)
    )
    return logg


def R_from_Teff_logg(Teff, logg):
    MGRID = pd.read_csv(data_dir + "/CO_thickH_processed.csv")
    logT = np.log10(Teff)
    # logR=np.log10(R)
    logR = interpolate.griddata(
        (MGRID["logT"], MGRID["logg"]), MGRID["logR"], (logT, logg)
    )
    R = 10 ** (logR)
    return R


def mass_from_Teff_logg(Teff, logg):
    MGRID = pd.read_csv(data_dir + "/CO_thickH_processed.csv")
    logT = np.log10(Teff)
    # logR=np.log10(R)
    Mass = interpolate.griddata(
        (MGRID["logT"], MGRID["logg"]), MGRID["#Mass"], (logT, logg)
    )
    return Mass


def hot_vs_cold(
    T1, T1_err, g1, g1_err, T2, T2_err, g2, g2_err, parallax, GaiaG, model="pier"
):
    M_bol_sun, Teff_sun, Rsun_cm, R_sun_pc = (
        4.75,
        5780.0,
        69.5508e9,
        2.2539619954370203e-08,
    )
    R1 = R_from_Teff_logg(T1, g1 / 100)
    R2 = R_from_Teff_logg(T2, g2 / 100)
    print(T1, g1)
    mod1 = interpolating_model_DA(T1, g1 / 100, m_type=model)
    flux1 = (
        mod1[:, 1]
        / (((1000 / parallax) * 3.086e18) ** 2)
        * (np.pi * 4 * (R1 * Rsun_cm) ** 2)
    )
    wave1 = mod1[:, 0]
    mod2 = interpolating_model_DA(T2, g2 / 100, m_type=model)
    flux2 = (
        mod2[:, 1]
        / (((1000 / parallax) * 3.086e18) ** 2)
        * (np.pi * 4 * (R2 * Rsun_cm) ** 2)
    )
    wave2 = mod2[:, 0]
    flux_G1, mag_G1 = synth_G(wave1, flux1)
    flux_G2, mag_G2 = synth_G(wave2, flux2)
    print(mag_G1, mag_G2)
    if abs(mag_G1 - GaiaG) <= abs(mag_G2 - GaiaG):
        return (T1, T1_err, g1 / 100.0, g1_err / 100.0)
    else:
        return (T2, T2_err, g2 / 100.0, g2_err / 100.0)
    # return(wave2,flux2)


def synth_G(spectrum_w, spectrum_f):
    profile_file = os.path.join(data_dir, "GAIA_GAIA2r.G.dat")
    min = 3320.0
    max = 10828.0
    filter_w, filter_r = np.loadtxt(profile_file, usecols=(0, 1), unpack=True)
    norm_filter = simps(filter_r, filter_w)  # Integrate over wavelength
    filter_r = filter_r / norm_filter
    filter_r = filter_r[
        (filter_w > np.min(spectrum_w)) & (filter_w < np.max(spectrum_w))
    ]
    filter_w = filter_w[
        (filter_w > np.min(spectrum_w)) & (filter_w < np.max(spectrum_w))
    ]
    min = np.min(filter_w) - 100.0
    max = np.max(filter_w) + 100.0
    filt_spec2 = spectrum_f[np.logical_and(spectrum_w >= min, spectrum_w <= max)]
    filt_spec1 = spectrum_w[np.logical_and(spectrum_w >= min, spectrum_w <= max)]

    tck_filt = interpolate.interp1d(
        filt_spec1, filt_spec2, kind="linear", assume_sorted=True
    )
    spectral_filt = tck_filt(filter_w)
    to_integrate = spectral_filt * filter_r
    flux_filt = simps(to_integrate, filter_w)
    ew = 5836.0
    c = 2.99792458e10
    zp = 2.495e-9
    zp = zp * ew**2 * 1.0e15 / c
    flux_filt = flux_filt * (ew**2 * 1.0e-8 / c)
    mag = -2.5 * np.log10(flux_filt / (zp * 1.0e-23))
    return flux_filt, mag
