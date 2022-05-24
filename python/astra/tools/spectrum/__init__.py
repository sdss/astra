from astra.tools.spectrum.loaders import Spectrum1D, SpectrumList
from numpy import median

def calculate_snr(spectrum, aggregate=median):
    use = (
        (spectrum.uncertainty.array > 0)
    &   (spectrum.flux.value > 0)
    )
    flux = spectrum.flux.value[use]
    sigma = spectrum.uncertainty.array[use]**0.5
    return aggregate(flux/sigma)
