import numpy as np
import time
import json

def float_to_3_bytes(x:float):
    assert x >= 0
    i = int(round(1.e5 * x))
    b0 = i//65536
    q = b0*65536
    b1 = (i - q)//256
    b2 = i - q - b1*256
    return [b0, b1, b2]

def bytes_to_float(b, i:int):
    return 1.e-5 * (b[i]*65536 + b[i+1]*256 + b[i+2])

class FluxArrayFile:
    """
    Class to store flux arrays with 3 bytes per value,
    because GSSP outputs flux with 5 digits after the 
    decimal point, so using 8 bytes as numpy.savez(...) 
    does is a waste of disk space.
    """
    def __init__(self, path):
        self.path = path
        
    def save(self, arr, meta):
        """
        arr: flux array, iterable of floats
        meta: dictionary with metadata, usually stellar parameters
        """
        if meta==None: meta = {}
                
        meta_s = json.dumps(meta)
        meta_b = bytearray(meta_s, 'UTF-8')
    
        bb = []
        for x in arr:
            bb.extend(float_to_3_bytes(x))
        data_b = bytearray(bb)
        
        with open(self.path, 'wb') as f:
            f.write(meta_b)
            f.write(bytearray([0]))
            f.write(data_b)

    def load(self):
        """
        Returns flux array and metadata
        """
        with open(self.path, 'rb') as f:
            bb = f.read()
            
        zero_pos = -1
        for i,v in enumerate(bb):
            if v==0:
                zero_pos = i
                break
                
        meta_b = bb[:zero_pos]
        data_b = bb[zero_pos+1:]
        
        meta_s = meta_b.decode('UTF-8')
        meta = json.loads(meta_s)
            
        assert len(data_b)%3 == 0
        ff = []
        for i in range(len(data_b)//3):
            ff.append(bytes_to_float(data_b, i*3))
            
        return ff, meta


if __name__=='__main__':
    fn = 'FA_test.flux'
    meta = {}
    meta['~!@#$%^&*()_+'] = ':"|{}<>?[]`,.;'
    meta['T_eff']=1234.5
    rnd = np.random.rand(1000)
    FA = FluxArrayFile(fn)
    FA.save(rnd, meta)
    arr, meta2 = FA.load()
    
    assert meta == meta2
    for z in zip(rnd, arr):
        s1 = '%.5f'%z[0]
        s2 = '%.5f'%z[1]
        assert s1==s2
        
    print('test passed')
    
    N = 333333
    rnd = np.random.rand(N)
    FA = FluxArrayFile(fn)
    t1 = time.time()
    FA.save(rnd, meta)
    t2 = time.time()
    arr, meta2 = FA.load()
    t3 = time.time()
    
    print('benchmark with '+str(N)+' numbers')
    print('saved in ', t2-t1, 'sec')
    print('loaded in ', t3-t2, 'sec')
    









