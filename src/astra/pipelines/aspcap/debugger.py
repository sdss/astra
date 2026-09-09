import os
import numpy as np

RAND = np.random.randint(0, 1000)
from socket import gethostname
from astra.utils import expand_path
HOSTNAME = gethostname()

def debugger(*foo):
    # `debugger` is called from inside exception handlers (e.g. `_safe_pre_process_ferre`),
    # so it must never raise: a failure here would mask the error being reported. $PBS is
    # unset outside of a cluster environment (CI, a laptop), in which case `expand_path`
    # leaves the literal "$PBS" and the open() below would fail with FileNotFoundError.
    path = expand_path(f"$PBS/{HOSTNAME}-{RAND}.log")
    if "$" in path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fp:
            fp.write(" ".join(map(str, foo)) + "\n")
    except OSError:
        return
