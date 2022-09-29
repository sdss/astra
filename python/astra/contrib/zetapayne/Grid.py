import os
import sys
from os.path import join
import numpy as np

class StellarParams:
    
    def __init__(self, Teff, logg, vsini, MH, vmicro):
        self.Teff = Teff
        self.logg = logg
        self.vsini = vsini
        self.MH = MH
        self.vmicro = vmicro
        
    def as_tuple(self):
        return (self.Teff, self.logg, self.vsini, self.vmicro, self.MH)
        
    def __str__(self):
        return 'Teff=%.0f, log(g)=%.2f, v*sin(i)=%.0f, v_micro=%.1f, [M/H]=%.1f,'%self.as_tuple()
        
    def __repr__(self):
        return self.__str__()


class Model:
    def __init__(self, name, flux):
        assert(type(flux)==np.ndarray)
        self.flux = flux
        self.name = name
        
    def __str__(self):
        return self.name
        
    def __repr__(self):
        return self.__str__()
        
    def stellar_params(self):
        a = self.name[2:-4].split('_')
        MH = 0.1*float(a[0])
        if self.name[1]=='m': MH = -MH
        Teff = float(a[1])
        logg = 0.01*float(a[2])
        vmicro = 0.1*float(a[3])
        vsini = float(a[6])
        sp = StellarParams(Teff, logg, vsini, MH, vmicro)
        return sp



class Grid:
    def __init__(self, grid_folder):
        self.folder = grid_folder
        self.models = []
        self.models_dict = {}

    def load_flux(fn):
        flux = []
        with open(fn) as f:
            for line in f:
                arr = line.split()
                flux.append(float(arr[1]))
        return np.array(flux)

    def load(self):
        if True:
            L, names = [],[]
            grid_files = [f for f in os.listdir(self.folder) if f.endswith(".rgs")]
            for grid_fn in grid_files:
                fn = join(self.folder, grid_fn)
                model_flux = Grid.load_flux(fn)
                M = Model(grid_fn, model_flux)
                self.models.append(M)
                self.models_dict[str(M.stellar_params())] = M
                
    def __repr__(self):
        r = 'Grid with '+str(len(self.models))+' models'
        return r
        








