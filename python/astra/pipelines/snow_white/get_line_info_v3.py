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
    #flux=(flux/np.sum(flux))*np.size(flux)
    #bin spectrum for anchor points 
    binsize=5
    xdata=[]
    ydata=[]
    edata=[]
    for i in range(0,(np.size(wave)-binsize),binsize):
        xdata.append(np.median(wave[i:i+binsize]))
        try:
            ydata.append(np.average(flux[i:i+binsize],weights=1/((err[i:i+binsize])**2)))
        except:
            ydata.append(np.mean(flux[i:i+binsize]))
        edata.append(np.median(err[i:i+binsize]))
    wave_a=np.array(xdata)
    flux_a=np.array(ydata)
    err_a=np.array(edata)

    #define fauter lists
    feature_list=['DA.features', 'DB.features', 'DQ.features', 'DZ.features', 'WDMS.features','Pec.features','hDQ.features']

    #define anchor points for spline
    if np.max(wave_a)<8910:
        #start_n=np.array([3850,3925.,4230.,4600.,5300, 6300.,6900., 7350.,8100.,(np.max(wa_original)-10.)])
        #start_n=np.array([3850.,4230.,4600.,4975., 5300., 5700.,6200.,6900., 7350.,8100.,(np.max(wave_a)-10.)])
        #start_n=np.array([3850.,4230.,4600., 5300., 5700.,6200.,6900., 7350.,8100.,(np.max(wave_a)-10.)])
        start_n=np.array([3850.,4230.,4600., 5300., 5700., 7350.,8100.,(np.max(wave_a)-10.)])
        start_pec=np.array([6300.,6900., 7350.,8100])
        #start_n=np.array([3850,3925.,4230.,4600.,5300, 6300.,6900., 7350.,8100.,(np.max(wa_original)-10.)])

    else:
        #start_n=np.array([3850,3925.,4230.,4600.,5300, 6300.,6900., 7350.,8100.,8900.,(np.max(wa_original)-10.)])
        #start_n=np.array([3850.,4230.,4600.,4975., 5300., 5700.,6200.,6900., 7350.,8100.,8900.,(np.max(wave_a)-10.)])
        #start_n=np.array([3850.,4230.,4600., 5300., 5700.,6200.,6900., 7350.,8100.,8900.,(np.max(wave_a)-10.)])
        start_n=np.array([3850.,4230.,4600., 5300., 5700., 7350.,8100.,8900.])#,(np.max(wave_a)-10.)])
        start_pec=np.array([6300.,6900., 7350.,8900])
    #define anchor points for BB
    start_wdms=np.array([4230.,4600.,5300, 6300.])#., 7350.])
    #(np.max(wave_a)-10.)])#4750

    # interpolate on predefined wavelength interval and range
    #print(np.min(wave),"BOOM")
    standard_wave=np.arange(3850,8300,1)
    w_m=scipy.interpolate.interp1d(wave_a,flux_a)
    w_err=scipy.interpolate.interp1d(wave_a,err_a)
    flux=w_m(standard_wave)
    err=w_err(standard_wave)
    wave=standard_wave

    fluxes=[]
    errs=[]
    for xxx in start_n:
        interv=flux_a[(wave_a>=xxx-5) & (wave_a<=xxx+5)]
        int_err=err_a[(wave_a>=xxx-5) & (wave_a<=xxx+5)]
        try:
            fluxes.append(np.average(interv,weights=1/(int_err**2)))
        except:
            fluxes.append(np.average(interv))
        errs.append(np.median(int_err))
    fluxes=np.array(fluxes)
    start_n=np.array(start_n)
    errs=np.array(errs)

    t0=[10000,1e-8]
    #tout = least_squares(bb_fit, t0, args = ([fluxes]),method='dogbox')
    #T,bla=scipy.optimize.curve_fit(bb,start_n,fluxes,[10000,1e-8],sigma=errs)    
    #print(T)
    #print(fluxes)
    #print(start_n)
    #fluxes = fluxes[~np.isnan(fluxes)]
    #start_n=start_n[~np.isnan(fluxes)]

    #func_poly = np.polyfit(start_n,fluxes,7)
    #p = np.poly1d(func_poly)
    #y=p(wave)
    spline = splrep(start_n,fluxes,w=1/errs,k=3)
    y = splev(wave,spline)

    #z = np.polyfit(start_n, np.log(fluxes), deg=1, w=np.sqrt(fluxes))
    #p = np.poly1d(z)
    #y= np.exp(p(wave))



    
    #plt.plot(wa_original,y,c="g")
    #bbdw=bb(wave,T[0],T[1])
   
    #bbwd=bb(wa_original,T[0],T[1])
    #plt.plot(wa_original,bbwd)

    fluxes_wdms=[]
    errs_wdms=[]
    for xxx in start_wdms:
        interv=flux_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        int_err=err_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        fluxes_wdms.append(np.mean(interv))
        errs_wdms.append(np.median(int_err))
    fluxes_wdms=np.array(fluxes_wdms)
    start_wdms=np.array(start_wdms)
    errs_wdms=np.array(errs_wdms)

    T_wdms1,bla=scipy.optimize.curve_fit(bb,start_wdms,fluxes_wdms,[10000,1e-8],sigma=errs_wdms)
    residuals = fluxes_wdms- bb(start_wdms, T_wdms1[0],T_wdms1[1])
    ss_res1 = np.sum(residuals**2)

    T_wdms2,bla=scipy.optimize.curve_fit(bb,start_wdms,fluxes_wdms,[6000,1e-8],sigma=errs_wdms)
    bbwdms=bb(wave,T_wdms2[0],T_wdms2[1])
    residuals = fluxes_wdms- bb(start_wdms, T_wdms2[0],T_wdms2[1])
    ss_res2 = np.sum(residuals**2)
    if ss_res2<ss_res1:
        T_wdms=T_wdms2
    else:
        T_wdms=T_wdms1
    bbwdms=bb(wave,T_wdms[0],T_wdms[1])

    fluxes_pec=[]
    errs_pec=[]
    for xxx in start_pec:
        interv=flux_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        int_err=err_a[(wave_a>=xxx-10) & (wave_a<=xxx+10)]
        fluxes_pec.append(np.mean(interv))
        errs_pec.append(np.median(int_err))
    fluxes_pec=np.array(fluxes_pec)
    start_pec=np.array(start_pec)
    errs_pec=np.array(errs_pec)
    print(start_pec,fluxes_pec)
    T_pec1,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[10000,1e-8],sigma=errs_pec)
    residuals = fluxes_pec- bb(start_pec, T_pec1[0],T_pec1[1])
    ss_res1 = np.sum(residuals**2)
    T_pec2,bla=scipy.optimize.curve_fit(bb,start_pec,fluxes_pec,[6000,1e-8],sigma=errs_pec)    
    residuals = fluxes_pec- bb(start_pec, T_pec2[0],T_pec2[1])
    ss_res2 = np.sum(residuals**2)
    if ss_res2<ss_res1:
        T_pec=T_pec2
    else:
        T_pec=T_pec2
    bbpec=bb(wave,T_pec[0],T_pec[1])

    


    #wa=np.hstack((s_1,s_2,s_4,s_5,s_6))#,s_7))
    #flux=np.hstack((f_1,f_2,f_4,f_5,f_6))#,f_7))
    #sigma=np.hstack((e_1,e_2,e_4,e_5,e_6))#,e_7))

    #func_poly = np.polyfit(wa, flux,3)
    #p = np.poly1d(func_poly)
    #ybest=p(wa_original)
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

    #plt.scatter(start_n,fluxes,c="r",zorder=3)
    #plt.scatter(np.mean(wave[(wave>6555)&(wave<6570)]),np.max(flux[(wave>6555)&(wave<6570)]),c="g")
    #plt.scatter(np.mean(wave[(wave>6550)&(wave<6580)]),np.mean(flux[(wave>6550)&(wave<6580)]),c="k")
    #plt.show()
#===================================================
    result={}
    for elem in feature_list:
        type=elem.rstrip('.features')
        features=[]
        start,end=np.loadtxt(os.path.join(PIPELINE_DATA_DIR, elem),skiprows=1,delimiter='-', usecols=(0,1),unpack=True)
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
   
    all_lines=np.concatenate((result['DA'],result['DB'],result['DQ'],result['DZ'],result['WDMS'],result['Pec'],result['hDQ'],emission),axis=None)
    
    # all_lines=np.array([result['DA'][0],result['DA'][1],result['DA'][2],result['DA'][3],result['DA'][4],result['DA'][5],result['DB'][0],result['DB'][1],result['DB'][2],result['DB'][3],result['DB'][4],result['DB'][5],result['DB'][6],result['DB'][7],result['DB'][8],result['DB'][9],result['DB'][10],result['DB'][11],result['DB'][12],result['DQ'][0],result['DQ'][1],result['DZ'][0],result['DZ'][1],result['WDMS'][0],result['WDMS'][1],result['WDMS'][2],result['WDMS'][3]])
    
    return all_lines
        
    #print(result['DA'])
    #outfile.write(('%s,%f,%f,%f,%f,%f,%f,')%(nome,result['DA'][0],result['DA'][1],result['DA'][2],result['DA'][3],result['DA'][4],result['DA'][5]))
    #outfile.write(('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,')%(result['DB'][0],result['DB'][1],result['DB'][2],result['DB'][3],result['DB'][4],result['DB'][5],result['DB'][6],result['DB'][7],result['DB'][8],result['DB'][9],result['DB'][10],result['DB'][11],result['DB'][12]))
    #outfile.write(('%f,%f,')%(result['DQ'][0],result['DQ'][1]))
    #outfile.write(('%f,%f\n')%(result['DZ'][0],result['DZ'][1]))

