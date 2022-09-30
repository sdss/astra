import math
import numpy as np
#from Rydberg import Rydberg

from astra.contrib.zetapayne.Rydberg import Rydberg


class RV_corrector:

    def __init__(self, min_wave, max_wave):
        paschen = Rydberg(3, air=True).get_series(7, 16)
        balmer = Rydberg(2, air=True).get_series(3, 16)
        braket = Rydberg(4, air=True).get_series(11, 25)
        full_list = paschen + balmer + braket
        filtered = [w for w in full_list if w>min_wave and w<max_wave]

        self.lines = np.log10(np.array(filtered)) # list of lines to do CCF
        RV_lim = 1000.0
        self.shifts = np.linspace(-RV_lim, RV_lim, 101) # velocities in km/s

    def apply_Doppler_shift(wave, v_km_s, reverse=False):
        c = 299792.458 # speed of light in km/s
        vc = v_km_s/c
        q = 1.0
        if reverse: q = -1.0
        return wave + np.log10(1 + q*vc)

    def get_CCF(ww, flux, lines):
        flux_values = np.interp(lines, ww, flux)
        return np.sum(flux_values**2)
        
    def get_RV(self, wave, flux):
        wave = np.log10(wave)
        CCFs=[]
        for shift in self.shifts:
            lines_shifted = RV_corrector.apply_Doppler_shift(self.lines, shift, reverse=True)
            ccf = RV_corrector.get_CCF(wave, flux, lines_shifted)
            CCFs.append(ccf)
        RV = self.shifts[np.argmin(CCFs)]
        self.CCF = CCFs
        return CCFs
        
    def correct(self, wave, flux, RV=None):
        if RV==None:
            RV = self.get_RV(wave, flux)
        wave_corr = RV_corrector.apply_Doppler_shift(wave, RV)
        return wave_corr


def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * math.sqrt(2*math.pi))

def get_moments(xx, yy):
    """
    Calculates 1st and 2nd moments of a function
    """
    ff = xx * yy
    mean = np.trapz(ff, xx)
    ff2 = (xx - mean)**2 * yy
    std = np.sqrt(np.trapz(ff2, xx))
    return mean, std


def get_RV_CCF_H_lines(wave, flux):
    """
    NOTE: balmer and paschen lines only
    """
    corr = RV_corrector(min(wave), max(wave))
    CCF = corr.get_RV(wave, flux)
    CCF = max(CCF) - CCF
    I = np.trapz(CCF, corr.shifts)
    CCF /= I

    mean, std = get_moments(corr.shifts, CCF)

    gg = gaussian(corr.shifts, mean, std)
    resid = CCF - gg
    sigma = np.std(resid)

    means = np.linspace(-1000, 1000, 1000)

    P = []
    for mean in means:
        gg = gaussian(corr.shifts, mean, std)
        resid = CCF - gg
        chi = np.sum((resid/sigma)**2)
        P.append(math.exp(-0.5*chi))
    P = np.array(P)
    I = np.trapz(P, means)
    P /= I

    RV, RV_1sigma = get_moments(means, P)
    return RV, RV_1sigma, (means, P)





