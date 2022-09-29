from math import *
import numpy as np

def DER_SNR(flux):
    """
    Estimates SNR for a given spectrum
    ----------------------------------
    Stoehr et al, 2008. DER_SNR: A Simple & General Spectroscopic
            Signal-to-Noise Measurement Algorithm
    """
    flux = [f for f in flux if not isnan(f)]
    signal = np.median(flux)
    Q = []
    for i in range(2, len(flux)-2):
        q = abs(2*flux[i] - flux[i-2] - flux[i+2])
        Q.append(q)
    noise = 1.482602 * np.median(Q) / sqrt(6.0)
    return signal/noise
