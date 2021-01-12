import numpy as np
from scipy import interpolate

def pseudo_continuum_normalize_da_wd(spectrum):

    regions = [
        (3600, 3650, 'M'),
        (3770, 3795, 'P'),
        (3796, 3830, 'P'),
        (3835, 3885, 'P'),
        (3895, 3960, 'P'),
        (3995, 4075, 'P'),
        (4160, 4210, 'M'),
        (4490, 4570, 'M'),
        (4620, 4670, 'M'),
        (5070, 5100, 'M'),
        (5200, 5300, 'M'),
        (6000, 6100, 'M'),
        (7000, 7050, 'M'),
        (7550, 7600, 'M'),
        (8400, 8450, 'M'),
    ]

    R = len(regions)

    N, P = spectrum.flux.shape

    continuum = np.ones((N, P))

    for j in range(N):

        points = np.zeros((R, 3))
        for i, (start, end, kind) in enumerate(regions):
            region_mask = (end >= spectrum.wavelength.value) \
                        * (spectrum.wavelength.value >= start)

            P = np.sum(region_mask)
            if P < 3:
                continue

            wave = spectrum.wavelength.value[region_mask]
            flux = spectrum.flux.value[j, region_mask]
            ivar = spectrum.uncertainty.array[j, region_mask]

            # Do as what was given,...
            tck = interpolate.splrep(wave, flux, w=ivar, s=1000)

            l = np.linspace(wave.min(), wave.max(), 10*(wave.size - 1) + 1)
            f = interpolate.splev(l, tck)

            w = np.median(ivar) / np.sqrt(P)

            if kind == "P":
                points[i, :] = [l[np.argmax(f)], f.max(), w]
            elif kind == "M":
                points[i, :] = [np.mean(l), np.mean(f), w]
            else:
                raise ValueError("unknown region kind")
            
        # Only keep points that had more than three pixels per region.
        points = points[points.T[0] > 0]

        min_wl, max_wl = (points.T[0].min(), points.T[0].max())
        if max_wl < 6460:
            knots = [3000, 4900, 4100, 4340, 4500, 4860, int(max_wl - 5)]
        else:
            knots = [3885, 4340, 4900, 6460]

        if (min_wl > 4340) or (max_wl < 4901):
            knots = None
        
        
        tck = interpolate.splrep(
            points[:, 0], points[:, 1],
            w=points[:, 2],
            t=knots,
            k=2
        )

        continuum[j] = interpolate.splev(spectrum.wavelength.value, tck)
    
    return continuum


if __name__ == "__main__":


    from astra.tools.spectrum import Spectrum1D
    from astra.tasks.io.sdss4 import SpecFile

    kwds = {
        "fiberid": 586,
        "mjd": 55863,
        "plateid": 4540,
        "release": "DR16",
        "run2d": "v5_13_0",
    }

    spec = SpecFile(**kwds)
    assert spec.complete()

    spectrum = Spectrum1D.read(spec.local_path)

    continuum = pseudo_continuum_normalize_da_wd(spectrum)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(spectrum.wavelength, spectrum.flux[0], c="k")
    ax.plot(spectrum.wavelength, continuum[0], c="tab:red", zorder=10)
    ax.set_ylim(0, 25)
    fig.savefig("tmp3.png")
