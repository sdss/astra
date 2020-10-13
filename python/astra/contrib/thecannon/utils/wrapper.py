
__all__ = ["Wrapper"]

import os
import sys
from time import time
from astra.utils import log
from multiprocessing import (Lock, Value)



# Initialize global counter for incrementing between threads.
_counter = Value('i', 0)
_counter_lock = Lock()

def _init_pool(args):
    global _counter
    _counter = args

class Wrapper(object):
    """
    A generic wrapper with a progressbar, which can be used either in serial or
    in parallel.

    :param f:
        The function to apply.

    :param args:
        Additional arguments to supply to the function `f`.

    :param kwds:
        Keyword arguments to supply to the function `f`.

    :param N:
        The number of items that will be iterated over.

    :param message: [optional]
        An information message to log before showing the progressbar.

    :param size: [optional]
        The width of the progressbar in characters.

    :returns:
        A generator. 
    """

    def __init__(self, f, args, kwds, N, message=None, size=100):
        self.f = f
        self.args = list(args if args is not None else [])
        self.kwds = kwds if kwds is not None else {}
        self._init_progressbar(N, message)
        

    def _init_progressbar(self, N, message=None):
        """
        Initialise a progressbar.

        :param N:
            The number of items that will be iterated over.
        
        :param message: [optional]
            An information message to log before showing the progressbar.
        """

        self.N = int(N)

        if self.N < 0:
            return
        
        try:
            rows, columns = os.popen('stty size', 'r').read().split()

        except:
            log.debug("Couldn't get screen size. Progressbar may look odd.")
            self.W = 100

        else:
            self.W = min(100, int(columns) - (12 + 21 + 2 * len(str(self.N))))

        self.t_init = time()
        self.message = message
        if 0 >= self.N:
            return None

        if message is not None:
            log.info(message.rstrip())
        
        sys.stdout.flush()
        with _counter_lock:
            _counter.value = 0
            

    def _update_progressbar(self):
        """
        Increment the progressbar by one iteration.
        """
        
        if 0 >= self.N:
            return None

        global _counter, _counter_lock
        with _counter_lock:
            _counter.value += 1

        index = _counter.value
        
        increment = max(1, int(self.N/float(self.W)))
        
        eta_minutes = ((time() - self.t_init) / index) * (self.N - index) / 60.0
        
        if index >= self.N:
            status = "({0:.0f}s)                         ".format(time() - self.t_init)

        elif float(index)/self.N >= 0.05 \
        and eta_minutes > 1: # MAGIC fraction for when we can predict ETA
            status = "({0}/{1}; ~{2:.0f}m until finished)".format(
                        index, self.N, eta_minutes)

        else:
            status = "({0}/{1})                          ".format(index, self.N)

        sys.stdout.write(
            ("\r[{done: <" + str(self.W) + "}] {percent:3.0f}% {status}").format(
            done="=" * int(index/increment),
            percent=100. * index/self.N,
            status=status))
        sys.stdout.flush()

        if index >= self.N:
            sys.stdout.write("\r\n")
            sys.stdout.flush()


    def __call__(self, x):
        try:
            result = self.f(*(list(x) + self.args), **self.kwds)
        except:
            log.exception("Exception within wrapped function")
            raise

        self._update_progressbar()
        return result

