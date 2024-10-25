from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import splrep,splev
import scipy.interpolate
from scipy.optimize import curve_fit
from scipy.integrate import quad
import os

from astra.utils import expand_path

PIPELINE_DATA_DIR = expand_path(f"$MWM_ASTRA/pipelines/snow_white")

h=6.626e-34
c=2.997e18
k=1.38e-23


def ex_d(x, a,b):
    return a * np.exp(b * (x**(-1/4)))

def bb(l,T,scale):
    f = 2*5.955e10 / l#**5
    f /= l
    f /= l
    f /= l
    f /= l
    f /= (np.exp( 1.438e8 / (l*T) ) -1)*scale
    #f=(f/np.sum(f))*np.size(f)
    return f 

def line_info(wave,flux,err):
    #bin spectrum for anchor points
    somma=np.sum(flux)
    numero=np.size(flux)

    #flux_temp=(flux/somma)*numero
    #err_rel=err/flux
    #err=err_rel*flux_temp
    #flux=flux_temp

    scaling = (numero / somma)
    err = err * scaling
    flux = flux * scaling

    binsize=2
    xdata=[]
    ydata=[]
    edata=[]
    for i in range(0,(np.size(wave)-binsize),binsize):
        xdata.append(np.median(wave[i:i+binsize]))
        try:
            ydata.append(np.average(flux[i:i+binsize],weights=1/((err[i:i+binsize])**2)))
        except:
            ydata.append(np.mean(flux[i:i+binsize]))

        assert np.isfinite(ydata[-1])
        edata.append(np.median(err[i:i+binsize]))
    wave_a=np.array(xdata)
    flux_a=np.array(ydata)
    err_a=np.array(edata)
    err_a[~np.isfinite(err_a)] = 1e3

    #define fauter lists
    feature_list=['DA.features', 'DB.features', 'DQ.features', 'DZ.features', 'WDMS.features','Pec.features','hDQ.features']

   #define anchor points for spline
    if np.max(wave_a)<8910:
        start_n=np.array([3850.,4230.,4600., 5300., 5700., 7350.,8100.,(np.max(wave_a)-10.)])
        start_pec=np.array([6000,6250.,6900., 7350.,8100])

    else:
        start_n=np.array([3850.,4230.,4600., 5300., 5700., 7350.,8100.,8900.])#,(np.max(wave_a)-10.)])
        start_pec=np.array([6000,6250.,6900., 7350.,8900])
    #define anchor points for WDMS line
    start_wdms=np.array([4600.,5100,5700, 6250.])#4230.

    # interpolate on predefined wavelength interval and range
    standard_wave=np.arange(3850,8300,1)#8300
    w_m=scipy.interpolate.interp1d(wave_a,flux_a)
    w_err=scipy.interpolate.interp1d(wave_a,err_a)
    flux=w_m(standard_wave)
    err=w_err(standard_wave)
    wave=standard_wave

    fluxes=[]
    errs=[]
    for xxx in start_n:
        interv=flux_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]#5
        int_err=err_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        try:
            fluxes.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes.append(np.average(interv))
        errs.append(np.mean(int_err))
    fluxes=np.array(fluxes)
    start_n=np.array(start_n)
    errs=np.array(errs)

    t0=[10000,1e-8]
    

    s = scipy.interpolate.InterpolatedUnivariateSpline(start_n,fluxes,w=1/errs)#,s=1)#InterpolatedUnivariateSpline
    y=s(wave)


    fluxes_wdms=[]
    errs_wdms=[]
    for xxx in start_wdms:
        interv=flux_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        int_err=err_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        try:
            fluxes_wdms.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes_wdms.append(np.average(interv))
        errs_wdms.append(np.mean(int_err))
    fluxes_wdms=np.array(fluxes_wdms)
    start_wdms=np.array(start_wdms)
    errs_wdms=np.array(errs_wdms)
    

    
    #p0 = (1,1)# start with values near those we expect
    #params, cv = scipy.optimize.curve_fit(ex_d,start_wdms,fluxes_wdms, p0)
    #a,b = params
    #bbwdms=ex_d(wave,a,b)
    try:
        T_wdms,bla=scipy.optimize.curve_fit(bb,start_wdms,fluxes_wdms,[12000,1e-10],sigma=errs_wdms)
    except:
        T_wdms,bla=scipy.optimize.curve_fit(bb,start_wdms,fluxes_wdms,[6000,1e-10],sigma=errs_wdms)
    bbwdms=bb(wave,T_wdms[0],T_wdms[1])


    
    fluxes_pec=[]
    errs_pec=[]
    for xxx in start_pec:

        interv=flux_a[(wave_a>=xxx-15) & (wave_a<=xxx+15)]
        int_err=err_a[(wave_a>=xxx-15) & (wave_a<=xxx+15)]
        try:
            fluxes_pec.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes_pec.append(np.average(interv))
        errs_pec.append(np.mean(int_err))
    fluxes_pec=np.array(fluxes_pec)
    start_pec=np.array(start_pec)
    errs_pec=np.array(errs_pec)*10

    #p0 = (1,1)

    #params, cv = scipy.optimize.curve_fit(ex_d,start_pec,fluxes_pec, p0)
    #a,b = params
    #bbpec=ex_d(wave,a, b)
    try:
        T_pec1,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[12000,1e-10],sigma=errs_pec)
    except:
        T_pec1,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[6000,1e-10],sigma=errs_pec)
    residuals = fluxes_pec- bb(start_pec, T_pec1[0],T_pec1[1])
    ss_res1 = np.sum(residuals**2)
    #T_pec2,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[6000,1e-13])#,sigma=errs_pec)    
    #residuals = fluxes_pec- bb(start_pec, T_pec2[0],T_pec2[1])
    #ss_res2 = np.sum(residuals**2)
    #if ss_res2<ss_res1:
    #    T_pec=T_pec2
    #else:
    T_pec=T_pec1
    bbpec=bb(wave,T_pec[0],T_pec[1])
    bbpec[bbpec<1e-10]=1e-1
#===================Halpha core====================================
    #f_s=flux[np.logical_and(wave>6540,wave<6580)]
    #f_s=flux[(np.logical_and(wave>6540,wave<6555))|(np.logical_and(wave>6570,wave<6585))]
    #f_m=y[(np.logical_and(wave>6540,wave<6555))|(np.logical_and(wave>6570,wave<6585))]
    #ratio_line=f_s/f_m
    #minimum=np.min(ratio_line)
    minimum=np.mean(flux[(wave>6545)&(wave<6590)])
    core=np.max(flux[(wave>6555)&(wave<6570)])#np.mean(flux[np.logical_and(wave>6555,wave<6570)]/y[np.logical_and(wave>6555,wave<6570)])
    emission=core/minimum
    #print(emission,"BOOOM")
#==================================================================== 
#=====================================================
    #plt.plot(wave_a,flux_a,c="0.7")
    #plt.plot(wave,y,c="b")
    #plt.plot(wave,bbwdms,c="m")
    #plt.plot(wave,bbpec,c="g")

    #plt.scatter(start_wdms,fluxes_wdms,c="m",zorder=3)
    #plt.scatter(start_pec,fluxes_pec,c="g",zorder=3)
    #plt.scatter(start_n,fluxes,c="r",zorder=3)

    #plt.scatter(np.mean(wave[(wave>6555)&(wave<6570)]),np.max(flux[(wave>6555)&(wave<6570)]),c="g")
    #plt.scatter(np.mean(wave[(wave>6550)&(wave<6580)]),np.mean(flux[(wave>6550)&(wave<6580)]),c="k")
    #plt.show()
    #print(BOOOOM)
#===================================================
    result={}
    for elem in feature_list:
        type=elem.rstrip('.features')
        features=[]
        start,end=np.loadtxt(f"{PIPELINE_DATA_DIR}/{elem}",skiprows=1,delimiter='-', usecols=(0,1),unpack=True)
        for xxx in range(np.size(start)):
            if type=="WDMS":
                f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                f_m=bbwdms[np.logical_and(wave>start[xxx],wave<end[xxx])]
                ratio_line=np.average(f_s)/np.average(f_m)
                features.append(ratio_line)
            elif type=="Pec":
                f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                f_m=bbpec[np.logical_and(wave>start[xxx],wave<end[xxx])]
                ratio_line=np.average(f_s)/np.average(f_m)
                features.append(ratio_line)
            elif type=="hDQ":
                f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                f_m=bbpec[np.logical_and(wave>start[xxx],wave<end[xxx])]
                ratio_line=f_s/f_m#np.average(f_s)/np.average(f_m)
                features.extend(ratio_line)
            else:
                try:
                    f_s=flux[np.logical_and(wave>start[xxx],wave<end[xxx])]
                    f_m=y[np.logical_and(wave>start[xxx],wave<end[xxx])]
                    f_e=err[np.logical_and(wave>start[xxx],wave<end[xxx])]
                except:
                    f_s=flux[np.logical_and(wave>start,wave<end)]
                    f_m=y[np.logical_and(wave>start,wave<end)]
                    f_e=err[np.logical_and(wave>start,wave<end)]
                w=1/(f_e**2)
                w=w/(np.max(w))
                corr=(f_s-f_m)*abs(w-1)
                f_s=f_s-corr
                ratio_line=(f_s/f_m)
                #ratio_line[ratio_line<1]=ratio_line[ratio_line<1]/w[ratio_line<1]
                #ratio_line[ratio_line>=1]=ratio_line[ratio_line>=1]*w[ratio_line>=1]
                #print(ratio_line)
                features.extend(ratio_line)
        features=np.array(features)
        
#------------------------------------------------------------------------------------------------------------------
        result[type]=features
    #print(emission,"HERE")
    emission2=np.mean(flux[np.logical_and(wave>6500,wave<6630)])/np.mean(y[np.logical_and(wave>6500,wave<6630)])
    all_lines=np.concatenate((result['DA'],result['DB'],result['DQ'],result['DZ'],result['WDMS'],result['Pec'],result['hDQ'],emission,emission2),axis=None)#,emission2
    #print(all_lines)
    # all_lines=np.array([result['DA'][0],result['DA'][1],result['DA'][2],result['DA'][3],result['DA'][4],result['DA'][5],result['DB'][0],result['DB'][1],result['DB'][2],result['DB'][3],result['DB'][4],result['DB'][5],result['DB'][6],result['DB'][7],result['DB'][8],result['DB'][9],result['DB'][10],result['DB'][11],result['DB'][12],result['DQ'][0],result['DQ'][1],result['DZ'][0],result['DZ'][1],result['WDMS'][0],result['WDMS'][1],result['WDMS'][2],result['WDMS'][3]])
    
    return all_lines
 

