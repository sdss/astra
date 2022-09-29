import numpy as np

class DimInfo:
    def __init__(self, length):
        self.length = length
        self.adjusted = False

    def set_length(self, new_length):
        self.length = new_length
        self.adjusted = True

class Partition:
    def __init__(self, params, model_limit):
        self.params = params
        self.model_limit = model_limit

        n_dim = len(params)
        cube_size = model_limit**(1.0/n_dim)
        self.dims_info = {}
        for p in params:
            self.dims_info[p] = DimInfo(cube_size)

    def adjust_dim_length(self, param_name, new_length):
        self.dims_info[param_name].set_length(new_length)
        n_default = 0
        adj_volume = 1.0
        for p in self.dims_info:
            if self.dims_info[p].adjusted:
                adj_volume *= self.dims_info[p].length
            else:
                n_default += 1
        new_default_length = (self.model_limit/adj_volume)**(1.0/n_default)
        for p in self.dims_info:
            if not self.dims_info[p].adjusted:
                self.dims_info[p].length = new_default_length
    
    def optimize_partition(self):
        param_len = []
        print('Grid dimensions:')
        pp = self.params
        self.total_volume = 1
        for p in pp:
            length = (pp[p][1] - pp[p][0])/pp[p][2]
            self.total_volume *= length
            param_len.append( (int(length), p) )
            print(' '*4, p, ':', length)

        param_len.sort()
        for pl in param_len:
            length = pl[0]
            p = pl[1]
            if self.dims_info[p].length > length:
                self.adjust_dim_length(p, length)

    def check(self):
        print('Subgrid dimensions:')
        V = 1
        for p in self.dims_info:
            n = int(self.dims_info[p].length)
            V *= n
            print(' '*4, p, ':', n)
        self.subgrid_volume = V
        print('Full grid volume:', int(self.total_volume))
        print('Target subgrid volume:', int(self.model_limit))
        print('Actual subgrid volume:', V)

if __name__=='__main__':
    model_limit = 10000

    param = {}
    param['T_eff'] = (6000, 10000, 100)
    param['log(g)'] = (3, 5, 0.1)
    param['[M/H]'] = (-0.8, 0.8, 0.1)
    param['v*sin(i)'] = (0, 200, 10)
    param['vmicro'] = (0, 5, 0.5)

    P = Partition(param, model_limit)
    P.optimize_partition()
    print('-'*25)

    for p in P.dims_info:
        print(p, P.dims_info[p].length)

    print('-'*25)
    P.check()

























