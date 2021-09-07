import math
import numpy as np
#from multiprocessing import Pool

from astra.contrib.thepayne_che.common import *

chi2_func = None

def _run_3_work_unit(opt):
    (c, pp) = opt
    for i,v in enumerate(c): pp[i] = v
    c.append(chi2_func(pp))
    return c

class UncertFit:
    """
    Wrapper around Fit that implements uncertainty estimation
    on top of the actual fit routine
    """
    def __init__(self, fit, spectral_resolution):
        self.fit = fit
        self.grid = fit.network.grid
        self.resol = spectral_resolution
        self.N_range_points = 1
        self.RV_step = 0.01

    def run(self, wave, flux, flux_err):
        return self._run_3(wave, flux, flux_err)

    def _get_RV_uncert(self, i, popt, CHI2_C, chi2_func):
        step = self.RV_step
        xx = [popt[i]-step, popt[i], popt[i]+step]
        yy = []
        for x in xx:
            pp = np.copy(popt)
            pp[i] = x
            yy.append(chi2_func(pp))
        poly_coef = np.polyfit(xx, yy, 2)
        poly_coef[-1] -= CHI2_C * yy[1]
        roots = np.roots(poly_coef)
        sigma = 0.5*abs(roots[0] - roots[1])
        return sigma
    
    def _run_3(self, wave, flux, flux_err, p0 = None):

        #Returns list of indices that have val as value
        def get_indices(val,param):
            indices = []
            for i in range(0,len(param)):
                if param[i] == val:
                    indices.append(i)
            return indices

        #Removes all elements at the given indices
        def remove_elements(indices,lis):
            for i in list(reversed(indices)):
                del lis[i]

        def extract_smallest(param,chi2):
            params = param.tolist()
            chi2s = chi2.tolist()
            #plt.plot(params,chi2s,'k.') 

            param_smallest = []
            chi2_smallest = []
            while len(params) != 0:
                value = params[0]
                indices = get_indices(value,params)
                min_chi2 = np.inf
                for j in indices:
                    if chi2s[j] < min_chi2:
                        min_chi2 = chi2s[j]
                remove_elements(indices,params)
                remove_elements(indices,chi2s)
                param_smallest.append(value)
                chi2_smallest.append(min_chi2)
            return param_smallest, chi2_smallest

        def make_fit(x,y,degree,weights = None):
            P = np.polyfit(x,y,degree,w=weights)
            grid_step = abs(x[0]-x[1])
            abscis = np.linspace(min(x)-1*grid_step,max(x)+1*grid_step,1000)
            fit = [0]*len(abscis)
            for i in range(len(P)):
                fit = [f+P[i]*x**(len(P)-1-i) for x,f in zip(abscis,fit)]
            return abscis, fit, P

        def get_param(param):
            data = Chi2_table
            chi2 = data[:,-1] 
            i = grid_params.index(param)

            #plt.figure()
            par, ch = extract_smallest(data[:,i], chi2-CHI2_C*min(chi2))

            abscis,fit, P = make_fit(par,ch,2)
            c_err = [0]*len(abscis)
            roots = np.roots(P)
            sigma = 0.5*abs(roots[0] - roots[1])

            #plt.plot(par,ch,'.k')
            #plt.xlabel(param, fontsize=18)
            #plt.ylabel(r'$\chi^{2}$',fontsize = 18)
            #plt.plot(abscis,c_err,'k')
            #plt.plot(abscis,fit,'k')
            #plt.show()

            return sigma


        def list_tensor_product(list_of_lists, just_list):
            return [ll+[v] for ll in list_of_lists for v in just_list]


        global chi2_func
        wave_start = min(wave)
        wave_end = max(wave)
        ndegree = 4 * self.resol * (wave_end - wave_start)/(wave_end + wave_start)
        CHI2_C = 1.0 + math.sqrt(2.0 / ndegree)

        res = self.fit.run(wave, flux, flux_err, p0=p0)
        popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func

        grid_params = []
        for pn in param_names:
            if self.grid[pn][0]!=self.grid[pn][1]: # excluding collapsed dimensions
                grid_params.append(pn)

        assert len(grid_params)>0

        steps = {}
        for pn in grid_params:
            if len(GSSP_steps[pn]) == 1:
                step = GSSP_steps[pn][0]
            else:
                step = GSSP_steps[pn][1]
            steps[pn] = step

        N_pts = self.N_range_points # number of points in each range N = N_pts*2 + 1
        ranges = []
        for i,pn in enumerate(grid_params):
            rr = []
            K = -N_pts
            while(K<=N_pts):
                rr.append(popt[i] + K*steps[pn])
                K += 1
            ranges.append(rr)

        new_params = [[x] for x in ranges[0]]
        for i,pn in enumerate(grid_params):
            if i==0: continue
            new_params = list_tensor_product(new_params, ranges[i])

        work = [(c, np.copy(popt)) for c in new_params]

        #with Pool() as pool:
        #    Chi2 = pool.map(_run_3_work_unit, work)
        Chi2 = list(map(_run_3_work_unit, work))

        Chi2_table = np.array(Chi2)
        uncert = [get_param(pn) for pn in grid_params]
            
        res.uncert = uncert

        i = len(grid_params)
        res.RV_uncert = self._get_RV_uncert(i, popt, CHI2_C, chi2_func)

        return res

    
    def _run_2(self, wave, flux, flux_err):
        wave_start = min(wave)
        wave_end = max(wave)
        ndegree = 4 * self.resol * (wave_end - wave_start)/(wave_end + wave_start)
        CHI2_C = 1.0 + math.sqrt(2.0 / ndegree)
        
        res = self.fit.run(wave, flux, flux_err)
        popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func
        
        uncert = []
        i=0
        for pn in param_names:
            if self.grid[pn][0]!=self.grid[pn][1]:
                if len(GSSP_steps[pn]) == 1:
                    step = GSSP_steps[pn][0]
                else:
                    step = GSSP_steps[pn][1]
                xx = [popt[i]-step, popt[i], popt[i]+step]
                yy = []
                for x in xx:
                    pp = np.copy(popt)
                    pp[i] = x
                    yy.append(chi2_func(pp))
                poly_coef = np.polyfit(xx, yy, 2)
                poly_coef[-1] -= CHI2_C * yy[1]
                roots = np.roots(poly_coef)
                sigma = 0.5*abs(roots[0] - roots[1])
                uncert.append(sigma)
                i+=1
        res.uncert = uncert

        res.RV_uncert = self._get_RV_uncert(i, popt, CHI2_C, chi2_func)
        
        return res
        
    
    def _run_1(self, wave, flux, flux_err):
        res = self.fit.run(wave, flux, flux_err)
        popt, pcov, model_spec, chi2_func = res.popt, res.pcov, res.model, res.chi2_func
        
        uncert = []
        i=0
        for pn in param_names:
            if self.grid[pn][0]!=self.grid[pn][1]:
                step = self.grid[pn][2]
                xx = [popt[i]-step, popt[i], popt[i]+step]
                yy = []
                for x in xx:
                    pp = np.copy(popt)
                    pp[i] = x
                    yy.append(chi2_func(pp))
                poly_coef = np.polyfit(xx, yy, 2)
                if poly_coef[0]>0:
                    sigma = math.sqrt(2.0/poly_coef[0])
                else:
                    sigma = 0.0
                uncert.append(sigma)
                i+=1
        res.uncert = uncert
        return res
  
  
  