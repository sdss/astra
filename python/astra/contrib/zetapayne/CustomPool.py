from multiprocessing import Process
from multiprocessing.pool import Pool


#----------------------------------------------
# Avoiding nested Pools with daemonic processes
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# Thank you Chris Arndt
class NonDaemonProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={},
                 *, daemon=None):
        super().__init__(None, target, name, args, kwargs, daemon=daemon)

    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class CustomPool(Pool):
    Process = NonDaemonProcess
#----------------------------------------------
