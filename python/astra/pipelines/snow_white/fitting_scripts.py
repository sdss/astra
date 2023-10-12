#import model_processing
#import emulator_DA
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
import os

from astra.utils import expand_path

PIPELINE_DATA_DIR = expand_path(f"$MWM_ASTRA/pipelines/snow_white")


def norm_spectra(spectra,model=True,add_infinity=False):
    """
    Normalised spectra by DA  continuum regions 
    spectra of form array([wave,flux,error]) (err not necessary so works on models)
    only works on SDSS spectra region
    Optional:
        EDIT n_range_s to change whether region[j] is fitted for a peak or mean'd
        add_infinity=False : add a spline point at [inf,0]
    returns spectra, cont_flux
    """
    
    #start_n=np.array([3630,3675.,3770.,3796.,3835.,3895.,3995.,4180,4490.,4620.,5070.,5200.,
     #                     5600.,6000.,7000.,7400.,7700.])#,8400.])
    #end_n=np.array([3660,3725.,3795.,3830.,3885.,3960.,4075.,4240,4570.,4670.,5100.,5300.,
     #                   5800.,6200.,7150.,7500.,7800.])#,8750.])
    #n_range_s=np.array(['M','M','M','P','P','P','P','P','M','M','M','M','M','M','M','M','M'])#,'M'])
    #if model==False:
     #   start_n=np.array([3805,3835.,3895.,3995.,4180,4490.,4620.,5070.,5200.,
      #                    5600.,6000.,7000.,7400.,7700.])
       # end_n=np.array([3830,3885.,3960.,4075.,4240,4570.,4670.,5100.,5300.,
        #                5800.,6100.,7150.,7500.,7800.])
        #n_range_s=np.array(['P','P','P','P','P','P','M','M','M','M','M','M','M','M','M','M','M'])
        #n_range_s=np.array(['M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M'])
    #else:
    start_n=np.array([3805,3835.,3895.,3995.,4180,4490.,4620.,5070.,5200.,
                      5600.,6000.,7000.,7400.,7700.])
    end_n=np.array([3830,3885.,3960.,4075.,4240,4570.,4670.,5100.,5300.,
                    5800.,6100.,7150.,7500.,7800.])
    n_range_s=np.array(['P','P','P','P','P','M','M','M','M','M','M','M','M','M','M','M','M'])
    if len(spectra[0])>2:
        snr = np.zeros([len(start_n),3])
        spectra[:,2][spectra[:,2]==0.] = spectra[:,2].max()
    else: 
        snr = np.zeros([len(start_n),2])
    wav = spectra[:,0]
    for j in range(len(start_n)):
        if (start_n[j] < wav.max()) & (end_n[j] > wav.min()):
            _s = spectra[(wav>=start_n[j])&(wav<=end_n[j])]
            _w = _s[:,0]
            #Avoids gappy spectra
            k=3 # Check if there are more points than 3
            if len(_s)>k:
                #interpolate onto 10* resolution
                l = np.linspace(_w.min(),_w.max(),(len(_s)-1)*10+1)
                if len(spectra[0])>2:
                    tck = interpolate.splrep(_w,_s[:,1],w=1/_s[:,2], s=1000)
                    #median errors for max/mid point
                    snr[j,2] = np.median(_s[:,2]) / np.sqrt(len(_w))
                else: tck = interpolate.splrep(_w,_s[:,1],s=0.0)
                f = interpolate.splev(l,tck)
                #find maxima and save
                if n_range_s[j]=='P':
                    if np.size(l[f==np.max(f)])>1:
                        if model==False:
                            snr[j,0]= np.mean(l)
                        else:
                            snr[j,0]= l[f==np.max(f)][0]
                    else:
                        if model==False:
                            snr[j,0]= np.mean(l)
                        else:
                            snr[j,0]= l[f==np.max(f)]
                    n=int(np.size(_s[:,1])/3.)
                    f_sort=np.sort(_s[:,1])
                    #errs=[np.where(f==i) for i in f_sort]
                    top5=f_sort[-n:]
                    #res = np.flatnonzero(np.isin(f, top5))  # NumPy v1.13+
                    snr[j,1]= np.average(top5)#,weights=[res])
                #find mean and save
                elif n_range_s[j]=='M':
                    snr[j,0:2] = np.mean(l), np.mean(f)
                else: print('Unknown n_range_s, ignoring')
    snr = snr[ snr[:,0] != 0 ]
    #t parameter chosen by eye. Position of knots.
    if snr[:,0].max() < 6460: knots = [4900,4100,4340,4500,4860,int(snr[:,0].max()-5)]
    else: knots = [3885,4340,4900,6460,7500]
    if snr[:,0].min() > 3885:
        print('Warning: knots used for spline norm unsuitable for high order fitting')
        knots=knots[1:]
    if (snr[:,0].min() > 4340) or (snr[:,0].max() < 4901): 
        knots=None # 'Warning: knots used probably bad'
   
                   
    if add_infinity: # Adds points at inf & 0 for spline to fit to err = mean(spec err)
        if snr.shape[1] > 2:
            mean_snr = np.mean(snr[:,2])
            snr = np.vstack([ snr, np.array([90000. ,0., mean_snr ]) ])
            snr = np.vstack([ snr, np.array([100000.,0., mean_snr ]) ])
        else:
            snr = np.vstack([ snr, np.array([90000.,0.]) ])
            snr = np.vstack([ snr, np.array([100000.,0.]) ])
    try: #weight by errors
        if len(spectra[0])>2:
            tck = interpolate.splrep(snr[:,0],snr[:,1], w=1/snr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
    except ValueError:
        knots=None
        if len(spectra[0])>2: 
            tck = interpolate.splrep(snr[:,0],snr[:,1], w=1/snr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
    spline = interpolate.splrep(snr[:,0],snr[:,1],k=3)
    cont_flux = interpolate.splev(wav,spline).reshape(wav.size, 1)   
    #cont_flux = interpolate.splev(wav,tck).reshape(wav.size, 1)
    spectra_ret = np.copy(spectra)
    spectra_ret[:,1:] = spectra_ret[:,1:]/cont_flux
    #spectra_ret=spectra[:,1]/cont_flux
#=======================plot for diagnostic============================
    #import matplotlib.pyplot as plt
    #print(spectra_ret)
    #plt.plot(spectra[:,0],spectra[:,1],zorder=1)
    #plt.plot(spectra[:,0],cont_flux,zorder=2)
    #plt.scatter(snr[:,0],snr[:,1],c="r",zorder=3)
    #plt.plot(spectra_ret[:,0],spectra_ret[:,1])
    #plt.show()
#======================================================================
    return spectra_ret, cont_flux

def fit_grid(specn,l_crop):
    #load normalised models and linearly interp models onto spectrum wave
    specn = specn[(specn[:,0]>3500)& (specn[:,0]<7500)]
    m_wave=np.arange(3000,8000,0.5)
    m_flux_n=np.load(os.path.join(PIPELINE_DATA_DIR, "da_flux_cube.npy"))
    m_param=np.load(os.path.join(PIPELINE_DATA_DIR, "da_param_cube.npy"))
    sn_w = specn[:,0]
    m_flux_n_i = interpolate.interp1d(m_wave,m_flux_n,kind='linear')(sn_w)
    #Crops models and spectra in a line region, renorms models, calculates chi2
    tmp_lines_m, lines_s, l_chi2 = [],[],[]
    for i in range(len(l_crop)):
        l_c0,l_c1 = l_crop[i,0],l_crop[i,1]
        l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
        l_s = specn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m,axis=1).reshape([len(l_m),1])
        l_chi2.append( np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2,axis=1) )
        tmp_lines_m.append(l_m)
        lines_s.append(l_s)
    #mean chi2 over lines and stores best model lines for output
    lines_chi2, lines_m = np.sum(np.array(l_chi2),axis=0), []
    is_best = lines_chi2==lines_chi2.min()
    for i in range(len(l_crop)): lines_m.append(tmp_lines_m[i][is_best][0])
    best_TL = m_param[is_best][0]
    return  lines_s,lines_m,best_TL,m_param,lines_chi2

def fit_func(x,specn,lcrop,emu,wref,mode=0):
    """Requires: x - initial guess of T, g, and rv
       specn/lcrop - normalised spectrum / list of cropped lines to fit
       mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model """
    
    tmp = tmp_func_rv(x[0], x[1],x[2], specn, lcrop, emu,wref,mode)
    if mode==0:
        return tmp[3] #this is the quantity that gets minimized   
    elif mode==1: return tmp[0], tmp[1], tmp[2]
    elif mode==2: return tmp[4]
    
def tmp_func_rv(_T, _g,_rv,_sn, _l, emu,wref,mode):
    c = 299792.458 # Speed of light in km/s=
    recovered=generate_modelDA(_T,(_g),emu)
    model=np.stack((wref,recovered),axis=-1)
    #if mode!=2:
     #   model1=convolve_gaussian_R(model,400)#5.487*2)
      #  model=np.vstack((m1,m2))
    norm_model, m_cont_flux=norm_spectra(model)
    m_wave_n, m_flux_n, sn_w = norm_model[:,0], norm_model[:,1], _sn[:,0]
   
    lines_m, lines_s, sum_l_chi2 = [],[],0
    flux_s,err_s=[],[]
    chi_line=[]
    for i in range(len(_l)):
        vv=_rv
        if mode!=2:
            _l_c=_l*(vv+c)/c
            m_wave_n_c=m_wave_n*(vv+c)/c
        else:
            _l_c=_l
            m_wave_n_c=m_wave_n
        m_flux_n_i_c = interpolate.interp1d(m_wave_n_c,m_flux_n,kind='linear')(sn_w)
     
#==========================================================================================        
        #    import matplotlib.pyplot as plt
         #   print(_T,_g,_rv)
          #  plt.plot(wref,recovered)
           # #plt.plot(sn_w,_sn[:,1],zorder=1)
            ##plt.plot(m_wave_n,m_flux_n,zorder=2)
            #plt.show()
#=========================================================================================
        m_flux_n_i=m_flux_n_i_c#*np.sum(_sn[:,1])/np.sum(m_flux_n_i_c)
        # Crop model and spec to line
        l_c0,l_c1 = _l_c[i,0],_l_c[i,1]
        #l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
        l_m= m_flux_n_i[(sn_w>=l_c0)&(sn_w<=l_c1)]
        l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        #print(l_c0,l_c1,np.size(l_m))
        lines_m.append(l_m)
        lines_s.append(l_s)
        flux_s.append(l_s[:,1])
        err_s.append(l_s[:,2])
        chi_line.append(np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)/np.size(l_m))

    all_lines_m=np.concatenate((lines_m),axis=0)
    all_lines_s=np.concatenate((flux_s),axis=0)
    all_err_s=np.concatenate((err_s),axis=0)

    sum_l_chi2=np.sum(((all_lines_s-all_lines_m)/all_err_s)**2)/np.size(all_lines_s)
    #chi_line=np.array(chi_line)
    #sum_l_chi2=np.mean(chi_line)
    chi=np.array((all_lines_s-all_lines_m)/all_err_s)
    #print(_T,_g,sum_l_chi2)
    return lines_s, lines_m, model, sum_l_chi2,chi



def generate_modelDA(teff,logg,emu):
    recovered = emu([np.log10(teff), logg])
    return(recovered)



def convolve_gaussian(spec, FWHM):
  """
  Convolve spectrum with a Gaussian with specifed FWHM.
  Causes wrap-around at the end of the spectrum.
  """
  sigma = FWHM/2.355
  x=spec[:,0]
  y=spec[:,1]
  def next_pow_2(N_in):
    N_out = 1
    while N_out < N_in:
      N_out *= 2
    return N_out

  #oversample data by at least factor 10 (up to 20).
  xi = np.linspace(x[0], x[-1], next_pow_2(10*len(x)))
  yi = interpolate.interp1d(x, y)(xi)

  yg = np.exp(-0.5*((xi-x[0])/sigma)**2) #half gaussian
  yg += yg[::-1]
  yg /= np.sum(yg) #Norm kernel

  yiF = np.fft.fft(yi)
  ygF = np.fft.fft(yg)
  yic = np.fft.ifft(yiF * ygF).real
  new_spec=np.stack((x,interpolate.interp1d(xi, yic)(x)),axis=-1)
  return new_spec

#

def convolve_gaussian_R(spec, R):
  """
  Similar to convolve_gaussian, but convolves to a specified 
  resolution R  rather than a specfied FWHM.
  """
  x=spec[:,0]
  y=spec[:,1]
  in_spec=np.stack((np.log(x),y),axis=-1)
  new_tmp= convolve_gaussian(in_spec, 1./R)
  new_spec=np.stack((x,new_tmp[:,1]),axis=-1)
  return new_spec


def hot_vs_cold(T1,g1,T2,g2,parallax,GaiaG,emu,wref):
    M_bol_sun, Teff_sun, Rsun_cm, R_sun_pc = 4.75, 5780., 69.5508e9, 2.2539619954370203e-08
    R1=R_from_Teff_logg(T1, g1)
    R2=R_from_Teff_logg(T2, g2)
    mod1=generate_modelDA(T1,g1*100,emu)
    flux1=(mod1/1e8)/((1000/parallax)*3.086e18)
    flux1=flux1/((1000/parallax)*3.086e18)
    flux1=flux1*(np.pi*(R1*Rsun_cm)**2)
    wave1=wref
    mod2=generate_modelDA(T2,g2*100,emu)
    #flux2=(mod2/1e8)/(((1000/parallax)*3.086e18)**2)*(np.pi*(R2*Rsun_cm)**2)
    flux2=(mod2/1e8)/((1000/parallax)*3.086e18)
    flux2=flux2/((1000/parallax)*3.086e18)
    flux2=flux2*(np.pi*(R2*Rsun_cm)**2)
    
    wave2=wref
    flux_G1,mag_G1=synthG(wave1,flux1)
    flux_G2,mag_G2=synthG(wave2,flux2)
    #print(mag_G1-GaiaG, mag_G2-GaiaG)
    if abs(mag_G1-GaiaG)<=abs(mag_G2-GaiaG):
        return(T1)
    else:
        return(T2)

def synthG(spectrum_w,spectrum_f):
    #spec=np.stack((spectrum_w, spectrum_f),axis=-1)
    fmin=3320.
    fmax=10828.
    filter_w,filter_r=np.loadtxt(os.path.join(PIPELINE_DATA_DIR, "GAIA_GAIA3.G.dat"),usecols=(0,1),unpack=True)    
    ifT = np.interp(spectrum_w, filter_w,filter_r, left=0., right=0.)
    nonzero = np.where(ifT > 0)[0]
    nonzero_start = max(0, min(nonzero) - 5)
    nonzero_end = min(len(ifT), max(nonzero) + 5)
    ind = np.zeros(len(ifT), dtype=bool)
    ind[nonzero_start:nonzero_end] = True
    #try:
     #   spec_flux = np.atleast_2d(spectrum_f)[..., ind]
    #except:
    spec_flux = spectrum_f[ind]
    a = np.trapz( ifT[ind] * spec_flux*spectrum_w[ind], spectrum_w[ind], axis=-1)
    b = np.trapz( ifT[ind]*spectrum_w[ind], spectrum_w[ind])
    if (np.isinf(a).any() | np.isinf(b).any()):
        print("Warn for inf value")
    nf=a/b
    ew=5836.
    c=2.99792e10
    zp=2.50386e-9* ew**2 *1.e15 / c
    fluxval=nf*(ew**2 * 1.e-8 / c)
    new_mag = -2.5 * np.log10(fluxval / (zp * 1.e-23))
    return  nf,new_mag

def R_from_Teff_logg(Teff, logg,atm="thick"):
    from scipy import interpolate
    if atm=="thick":
        #MGRID=pd.read_csv("CO_thickH_processed.csv")
        MGRID=pd.read_csv(os.path.join(PIPELINE_DATA_DIR, "new_MR_H.csv"))
    elif atm=="thin":
        MGRID=pd.read_csv(os.path.join(PIPELINE_DATA_DIR, "CO_thinH_processed.csv"))
    logT = np.log10(Teff)
    #logR=np.log10(R)
    #logR= interpolate.griddata((MGRID['logT'], MGRID['logg']), MGRID['logR'],(logT, logg))
    #R=10**(logR)
    R= interpolate.griddata((MGRID['logT'], MGRID['logg']), MGRID['R'],(logT, logg))
    return R
