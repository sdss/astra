import numpy as np

RAND = np.random.randint(0, 1000)
from socket import gethostname

HOSTNAME = gethostname()

def debugger(*foo):
    with open(f"/scratch/general/nfs1/u6020307/pbs/{HOSTNAME}-{RAND}.log", "a") as fp:
        fp.write(" ".join(map(str, foo)) + "\n")