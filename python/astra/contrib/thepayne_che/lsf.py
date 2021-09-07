import math
import numpy as np
from bisect import bisect
from scipy.ndimage import gaussian_filter1d


def gaussian(xx, mu, sigma):
    return np.exp(-0.5*((xx-mu)/sigma)**2)/(sigma * math.sqrt(2*math.pi))

class GaussKernel:

    def __init__(self, wave, w0, sigma):
        Q = 5.0
        w_start = w0 - Q*sigma
        w_end = w0 + Q*sigma
        i_start = bisect(wave, w_start)
        i_end = bisect(wave, w_end)

        wave_slice = wave[i_start:i_end]
        self.i_start = i_start
        self.i_end = i_end
        self.wave = wave_slice
        self.kernel = gaussian(wave_slice, w0, sigma)

    def integrate(self, flux):
        return np.trapz(self.kernel * flux[self.i_start:self.i_end], self.wave)


class LSFBase:
    def __init__(self):
        self.init_FWHM()

    def init_FWHM(self):
        self.FWHM_factor = 2 * math.sqrt(2* math.log(2))

    def apply(self):
        raise Exception('Not implemented')


class LSF_wave_R(LSFBase):
    def __init__(self, wave, R, spec_wave, NN_wave):
        super().__init__()
        delta_lambda = wave / R
        delta_lambda = np.interp(spec_wave, wave, delta_lambda)
        self.spec_wave = spec_wave
        self.G = [ GaussKernel(NN_wave, w, delta_lambda[i] / self.FWHM_factor) for i,w in enumerate(spec_wave)]

    def apply(self, flux):
        res = [self.G[i].integrate(flux) for i,w in enumerate(self.spec_wave)]
        res = np.array(res)
        return res


class LSF_Fixed_R(LSF_wave_R):
    def __init__(self, R, spec_wave, NN_wave):
        self.init_FWHM()
        delta_lambda = spec_wave / R
        self.spec_wave = spec_wave
        self.G = [ GaussKernel(NN_wave, w, delta_lambda[i] / self.FWHM_factor) for i,w in enumerate(spec_wave)]


class LSF_APOGEE(LSFBase):

    def __init__(self, wave, LSF):
        self.wave = wave
        self.LSF = LSF


    def apply(self, wave, flux):
        L = self.LSF.shape[1]
        L2 = L//2
        res = []
        for i,w in enumerate(self.wave):
            start = i-L2
            end = i+L2
            if start<0: start = 0
            if end >= len(self.wave): end = len(self.wave)-1
            LSF_wave = self.wave[start:end]
            start = bisect(wave, LSF_wave[0])-1
            end = bisect(wave, LSF_wave[-1])+1
            wave_slice = wave[start:end]
            flux_slice = flux[start:end]
            LSFi = np.interp(wave_slice, LSF_wave, self.LSF[i,:])

            I = np.trapz(flux_slice*LSFi, wave_slice)
            res.append(I)


if __name__=='__main__':
    import matplotlib.pyplot as plt

    wave = np.linspace(15000, 17000, 10000)
    R = 1000*np.ones(len(wave))
    flux = np.ones(len(wave))
    flux[5000] = 0.0
    
    lsf = LSF_Fixed_R(1000, wave, wave)

    conv = lsf.apply(flux)

    plt.plot(wave, conv)
    plt.show()






