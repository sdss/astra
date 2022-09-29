
class Subgrid:

    def __init__(self, grid, partition, param_list):
        self.last_block = {}
        subgrid = {}
        for p in param_list:
            n = int(partition.dims_info[p].length)
            delta = n * grid[p][2]
            subgrid[p] = [grid[p][0], grid[p][0] + delta, grid[p][2]]
            remaining = grid[p][1] - subgrid[p][1]
            self.last_block[p] = False
            if remaining < delta:
                subgrid[p][1] = grid[p][1]
                self.last_block[p] = True
            elif abs(remaining) < 1.e-6:
                self.last_block[p] = True
            
        self.grid = grid
        self.partition = partition
        self.param_list = param_list
        self.subgrid = subgrid
        self.init = True

    def next_subgrid(self):
        if self.init:
            self.init = False
            return True
        return self.increase_param(0)

    def increase_param(self, param_index):
        p = self.param_list[param_index]
        n = int(self.partition.dims_info[p].length)
        
        if self.last_block[p]:
            next_index = param_index + 1
            if next_index == len(self.param_list): return False
            self.subgrid[p] = [self.grid[p][0], self.grid[p][0] + n * self.grid[p][2], self.grid[p][2]]
            self.last_block[p] = False
            return self.increase_param(next_index)
        
        delta = n * self.grid[p][2]
        self.subgrid[p][0] += delta
        self.subgrid[p][1] += delta
        remaining = self.grid[p][1] - self.subgrid[p][1]
        if remaining < delta:
            self.subgrid[p][1] = self.grid[p][1]
            self.last_block[p] = True
        if abs(remaining) < 1.e-6:
            self.last_block[p] = True
 
        return True
 
    def volume(self):
        V = 1.0
        sub = self.subgrid
        for p in sub:
            V *= (sub[p][1] - sub[p][0]) / sub[p][2]
        return int(round(V))
 
 
 
 
 
 
 
 
 
 