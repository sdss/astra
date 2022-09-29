import numpy as np
import matplotlib.pyplot as plt


class Rydberg:
    """
    Rydberg equation for hydrogen line series
    """
    def __init__(self, n1, air=False):
        """
        n1: lower energy level
        air: True for air, False for vacuum
        """
        self.n1 = n1
        self.air = air
        self.R = 10967758.340280352 # m^-1

    def get_series(self, n2_min, n2_max):
        ww = []
        for n2 in range(n2_min, n2_max):
            w = 1/(self.R*(1/self.n1**2 - 1/n2**2))
            w_A = w*1.e10 # m -> AA
            if self.air:
                n = Rydberg.refraction_index_V2A(w_A)
                w_A = w_A / n
            ww.append(w_A) 
        return ww

    def refraction_index_V2A(lambda_vacuum):
        """
        Returns refraction index of air for vacuum-air conversion
        Donald Morton (2000, ApJ. Suppl., 130, 403)
        see http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
        lambda_vacuum: wavelength in angstrom
        """
        s = 1.e4/lambda_vacuum
        n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
        return n

    def refraction_index_A2V(lambda_air):
        """
        Returns refraction index of air for air-vacuum conversion
        see http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
        lambda_vacuum: wavelength in angstrom
        """
        s = 1.e4/lambda_air
        n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
        return n

if __name__=='__main__':
    R = Rydberg(2, air=True)
    ww = np.linspace(3500, 10000, 1000)
    nn = [Rydberg.refraction_index_V2A(w) for w in ww]
    plt.plot(ww, nn)
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Refraction index')
    plt.show()








