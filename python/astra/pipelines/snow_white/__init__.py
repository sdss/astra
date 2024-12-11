import scipy.optimize as op
from scipy import linalg, interpolate
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lmfit
from typing import Iterable, Optional
import pandas as pd

from astra import __version__, task
from astra.utils import log, expand_path
from peewee import JOIN
from astra.models.boss import BossVisitSpectrum
from astra.models.source import Source
from astra.models.spectrum import SpectrumMixin
from astra.models.snow_white import SnowWhite


PIPELINE_DATA_DIR = expand_path(f"$MWM_ASTRA/pipelines/snow_white")
LARGE = 1e3

@task
def snow_white(
    spectra: Optional[Iterable[SpectrumMixin]] = (
        BossVisitSpectrum
        .select()
        .join(Source)
        .switch(BossVisitSpectrum)
        .join(SnowWhite, JOIN.LEFT_OUTER, on=(SnowWhite.spectrum_pk == BossVisitSpectrum.spectrum_pk))
        .where(
            Source.assigned_to_program("mwm_wd")
        &   SnowWhite.spectrum_pk.is_null()
        )        
    ), 
    plot=True,
    debug=False
) -> Iterable[SnowWhite]:
    """
    Classify white dwarf types based on their spectra, and fit stellar parameters to DA-type white dwarfs.

    :param spectra:
        Input spectra.
    """
    
    from astra.pipelines.snow_white import get_line_info_v3, fitting_scripts

    #with open(os.path.join(PIPELINE_DATA_DIR, 'training_file_v3'), 'rb') as f:
    with open(os.path.join(PIPELINE_DATA_DIR, '20240801_training_file'), 'rb') as f:        
        kf = pickle._load(f, fix_imports=True)


    wref = np.load(os.path.join(PIPELINE_DATA_DIR, "wref.npy"))

    # Once again, we hhave to put this stupid hack in
    sys.path.insert(0, os.path.dirname(__file__))
    with open(os.path.join(PIPELINE_DATA_DIR, "emu_file"), 'rb') as pickle_file:
        emu = pickle.load(pickle_file)

    for spectrum in spectra:

        try:                

            if np.sum(spectrum.flux) == 0:
                yield SnowWhite(
                    source_pk=spectrum.source_pk,
                    spectrum_pk=spectrum.spectrum_pk,  
                    flag_no_flux=True
                )
                continue

            bad_pixel = (
                (spectrum.flux == 0)
            |   (spectrum.ivar == 0)
            |   (~np.isfinite(spectrum.flux))
            )
            flux = np.copy(spectrum.flux)
            flux[bad_pixel] = 0.01
            e_flux = np.copy(spectrum.e_flux)
            e_flux[bad_pixel] = LARGE

            data_args = (spectrum.wavelength, flux, e_flux)
            
            labels = get_line_info_v3.line_info(*data_args)
            predictions = kf.predict(labels.reshape(1, -1))
            probs = kf.predict_proba(labels.reshape(1, -1))
            
            first = probs[0][kf.classes_==predictions[0]]
            if first >= 0.5:
                classification = predictions[0]
            else:
                second = sorted(probs[0])[-2]
                if second/first>0.6:
                    classification = predictions[0]+"/"+kf.classes_[probs[0]==second]
                else:
                    classification = predictions[0]+":"

            result_kwds = dict(
                source_pk=spectrum.source_pk,
                spectrum_pk=spectrum.spectrum_pk,            
                classification=classification,
            )
            result_kwds.update(
                dict(zip([f"p_{class_name.lower()}" for class_name in kf.classes_], probs[0]))
            )

            if classification not in ("DA", "DA:"):                
                result = SnowWhite(**result_kwds)

            else:
                # Fit DA-type
                spectra=np.stack((data_args),axis=-1)
                spectra = spectra[(np.isnan(spectra[:,1])==False) & (spectra[:,0]>3600)& (spectra[:,0]<9800)]
                spec_w=spectrum.wavelength


                #normilize spectrum
                spec_n, cont_flux = fitting_scripts.norm_spectra(spectra,mod=False)
                #load lines to fit and crops them
                line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop.dat'))
                l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]

                #fit entire grid to find good starting point
                lines_sp,lines_mod,best_grid,grid_param,grid_chi=fitting_scripts.fit_grid(spec_n,line_crop)

                first_T=grid_param[grid_chi==np.min(grid_chi)][0][0]
                first_g=800
                initial=0
                tl= pd.read_csv(os.path.join(PIPELINE_DATA_DIR, 'reference_phot_tlogg.csv'))
                sourceID=np.array(tl[u'source_id']).astype(str)
                T_H=np.array(tl[u'teff_H']).astype(float)
                log_H=np.array(tl[u'logg_H']).astype(float)
                eT_H=np.array(tl[u'eteff_H']).astype(float)
                elog_H=np.array(tl[u'elogg_H']).astype(float)
                GaiaID=str(spectrum.source.gaia_dr3_source_id)
                if GaiaID in sourceID: #if there is a photometric solution use that as starting point
                    first_T=T_H[GaiaID][0]
                    first_g=log_H[GaiaID][0]*100
                    initial=1
                if first_T > 80000:
                    first_T=80000
                if first_g < 701:
                    first_g=701
                if first_T>=16000 and first_T<=40000:
                    line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop.dat'),skiprows=1) #exclude Halpha. It is needed in exception
                elif first_T>=8000 and first_T<16000:
                    line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_cool.dat'),skiprows=1)
                elif first_T<8000:
                    line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_vcool.dat'),skiprows=1)
                elif first_T>40000:
                    line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_hot.dat'),skiprows=1)
                l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
                
                    
                # initiate parameters for the fit
                fit_params = lmfit.Parameters()
                fit_params['teff'] = lmfit.Parameter(name="teff",value=first_T,min=3000,max=80000)
                fit_params['logg'] = lmfit.Parameter(name="logg",value=first_g,min=701,max=949)
                fit_params['rv'] = lmfit.Parameter(name="rv",value=0.2, min=-80, max=80) #this is a wavelenght shift not a radial velocity. since a eparate module finds rv

                #new normalization rotine working just on the balmer lines
                spec_nl=fitting_scripts.da_line_normalize(spectra,l_crop,mod=False)
                
                #this calls the scripts in fitting_scipts and does the actual fitting
                new_best= lmfit.minimize(fitting_scripts.line_func_rv,fit_params,args=(spec_nl,l_crop,emu,wref))
                #problematic nodes results can sometimes be fixed by excluding certain lines
                prob_list=[7.01,9.49,7.5,8.2,8.]
                if round(new_best.params['logg'].value/100,4) in prob_list:
                    if first_T>=16000 and first_T<=40000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop.dat'),max_rows=6)
                    elif first_T>=8000 and first_T<16000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_cool.dat'),max_rows=6)
                    elif first_T<8000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_vcool.dat'),max_rows=5)
                    elif first_T>40000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_hot.dat'),max_rows=6)
                    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
                    new_best= lmfit.minimize(fitting_scripts.line_func_rv,fit_params,args=(spec_nl,l_crop,emu,wref))


                best_T=new_best.params['teff'].value
                best_Te=new_best.params['teff'].stderr
                best_g=new_best.params['logg'].value
                best_ge=new_best.params['logg'].stderr
                shift=new_best.params['rv'].value
                chi2=new_best.redchi #can easily get a chi2 

                if initial ==1:
                    esult_kwds.update(
                        teff=best_T,
                        e_teff=best_Te,
                        logg=best_g/100,
                        e_logg=best_ge/100)
                    
                elif initial ==0: #if initial guess not from photometric result need to repeat for hot/cold solution
                    fit_params = lmfit.Parameters()
                    fit_params['logg'] = lmfit.Parameter(name="logg",value=800,min=701,max=949) #stick with logg 800
                    fit_params['rv'] = lmfit.Parameter(name="rv",value=0.2, min=-80, max=80)
        
                    if first_T <=13000.:
                        tmp_Tg,tmp_chi= grid_param[grid_param[:,0]>13000.], grid_chi[grid_param[:,0]>13000.]
                        second_T= tmp_Tg[tmp_chi==np.min(tmp_chi)][0][0]
                        fit_params['teff'] = lmfit.Parameter(name="teff",value=second_T,min=12000,max=80000)

                    elif first_T >13000.:
                        tmp_Tg,tmp_chi= grid_param[grid_param[:,0]<13000.], grid_chi[grid_param[:,0]<13000.]
                        second_T= tmp_Tg[tmp_chi==np.min(tmp_chi)][0][0]
                        fit_params['teff'] = lmfit.Parameter(name="teff",value=second_T,min=3000,max=14000)

                    if second_T>=16000 and second_T<=40000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop.dat'),skiprows=1)
                    elif second_T>=8000 and second_T<16000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_cool.dat'),skiprows=1)
                    elif second_T<8000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_vcool.dat'),skiprows=1)
                    elif second_T>40000:
                        line_crop = np.loadtxt(os.path.join(PIPELINE_DATA_DIR, 'line_crop_hot.dat'),skiprows=1)
                    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
                
                #====================find second solution ==============================================
                    second_best= lmfit.minimize(fitting_scripts.line_func_rv,fit_params,args=(spec_nl,l_crop,emu,wref),method="leastsq")
                    best_T2=second_best.params['teff'].value
                    best_Te2=second_best.params['teff'].stderr
                    best_g2=second_best.params['logg'].value
                    best_ge2=second_best.params['logg'].stderr
                    shift2=second_best.params['rv'].value
                    chi2_2=second_best.redchi #can easily get a chi2 

               
                #========================use gaia G mag and parallax to solve for hot vs cold solution
                
                    T_true=fitting_scripts.hot_vs_cold(best_T,best_g/100,best_T2,best_g2/100,spectrum.source.plx or np.nan,spectrum.source.g_mag or np.nan,emu,wref)
                    if T_true==best_T:
                        result_kwds.update(
                            teff=best_T,
                            e_teff=best_Te,
                            logg=best_g/100,
                            e_logg=best_ge/100,
                            #v_rel=best_rv  # rv should not be an output of snow_white now
                        )
                    elif T_true==best_T2:
                        result_kwds.update(
                            teff=best_T2,
                            e_teff=best_Te2,
                            logg=best_g2/100,
                            e_logg=best_ge2/100,
                            #v_rel=best_rv2
                        )
                if spectrum.snr <= 8:
                    result_kwds["flag_low_snr"] = True

                result = SnowWhite(**result_kwds)

#=========================================================still use old fit_func to generateretrieve model for plot==================================================

                # Get and save the 2 best lines from the spec and model, and the full models
                lines_s,lines_m,mod_n=fitting_scripts.fit_func(
                    (best_T,best_g,shift),
                    spec_n,l_crop,emu,wref,mode=1
                )

                full_spec=np.stack(data_args,axis=-1)
                full_spec = full_spec[(np.isnan(full_spec[:,1])==False) & (full_spec[:,0]>3500)& (full_spec[:,0]<7900)]
                

                # Adjust the flux of models to match the spectrum
                check_f_spec=full_spec[:,1][(full_spec[:,0]>4500.) & (full_spec[:,0]<4550.)]
                check_f_model=mod_n[:,1][(mod_n[:,0]>4500.) & (mod_n[:,0]<4550.)]
                adjust=np.average(check_f_model)/np.average(check_f_spec)

                model_wavelength, model_flux = (mod_n[:,0]+shift, (mod_n[:,1]/adjust))
                # resample
                resampled_model_flux = interpolate.interp1d(model_wavelength, model_flux, kind='linear', bounds_error=False)(spectrum.wavelength)

                output_path = expand_path(result.intermediate_output_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fits.HDUList([
                    fits.PrimaryHDU(), 
                    fits.ImageHDU(resampled_model_flux)
                    ]
                ).writeto(output_path, overwrite=True)
                    
                if plot:
                    if initial==0:
                        lines_s_o,lines_m_o,mod_n_o=fitting_scripts.fit_func((best_T2,best_g2,shift2),
                                                                    spec_n,l_crop,emu,wref,mode=1)
                    fig=plt.figure(figsize=(8,5))
                    ax1 = plt.subplot2grid((1,4), (0, 3),rowspan=3)
                    step = 0
                    for i in range(0,len(lines_s)): # plots Halpha (i=0) to H6 (i=5)
                        min_p   = lines_s[i][:,0][lines_s[i][:,1]==np.min(lines_s[i][:,1])][0]
                        ax1.plot(lines_s[i][:,0]-min_p,lines_s[i][:,1]+step,color='k')
                        ax1.plot(lines_s[i][:,0]-min_p,lines_m[i]+step,color='r')
                        if initial ==0:
                            min_p_o = lines_s_o[i][:,0][lines_s_o[i][:,1]==np.min(lines_s_o[i][:,1])][0]                        
                            ax1.plot(lines_s_o[i][:,0]-min_p_o,lines_m_o[i]+step,color='g')
                        step+=0.5
                    xticks = ax1.xaxis.get_major_ticks()
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])

                    ax2 = plt.subplot2grid((3,4), (0, 0),colspan=3,rowspan=2)
                    ax2.plot(full_spec[:,0],full_spec[:,1],color='k')
                    ax2.plot(mod_n[:,0]+shift,(mod_n[:,1]/adjust),color='r')
                    if initial ==0:
                        check_f_model_o=mod_n_o[:,1][(mod_n_o[:,0]>4500.) & (mod_n_o[:,0]<4550.)]
                        adjust_o=np.average(check_f_model_o)/np.average(check_f_spec)
                        ax2.plot(mod_n_o[:,0]+shift2,mod_n_o[:,1]/adjust_o,color='g')

                    ax2.set_ylabel(r'F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1} \AA^{-1}$]',fontsize=12)
                    ax2.set_xlabel(r'Wavelength $(\AA)$',fontsize=12)
                    ax2.set_xlim([3400,5600])
                    ax2.set_ylim(0, 2 * np.nanmax(mod_n[:,1]/adjust))
                    ax3 = plt.subplot2grid((3,4), (2, 0),colspan=3,rowspan=1,sharex=ax2)

                    flux_i = interpolate.interp1d(mod_n[:,0]+shift,mod_n[:,1]/adjust,kind='linear')(full_spec[:,0])
                    wave3=full_spec[:,0]
                    flux3=full_spec[:,1]/flux_i
                    binsize=1
                    xdata3=[]
                    ydata3=[]
                    for i in range(0,(np.size(wave3)-binsize),binsize):
                        xdata3.append(np.average(wave3[i:i+binsize]))
                        ydata3.append(np.average(flux3[i:i+binsize]))
                    plt.plot(xdata3,ydata3)

                    plt.hlines(1.02, 3400,5600,colors="r")
                    plt.hlines(1.01, 3400,5600,colors="0.5",ls="--")
                    plt.hlines(0.98, 3400,5600,colors="r")
                    plt.hlines(0.99, 3400,5600,colors="0.5",ls="--")
                    ax3.set_xlim([3400,5600])
                    ax3.set_ylim([0.95,1.04])
                    
                    figure_path = output_path[:-4] + ".png"
                    fig.savefig(figure_path)
                    plt.close("all")

            # No chi2, statistics, or flagging information..
            yield result

        except:
            log.exception(f"Exception on spectrum={spectrum}")
            if debug:
                raise
