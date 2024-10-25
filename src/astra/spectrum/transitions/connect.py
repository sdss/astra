from astropy.io import registry
from astropy.table.info import serialize_method_as

class TransitionsRead(registry.UnifiedReadWrite):

    def __init__(self, instance, cls):
        super().__init__(instance, cls, "read")

    def __call__(self, *args, **kwargs):
        cls = self._cls
        out = registry.read(cls, *args, **kwargs)
        return out


class TransitionsWrite(registry.UnifiedReadWrite):


    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'write')

    def __call__(self, *args, serialize_method=None, **kwargs):
        instance = self._instance
        with serialize_method_as(instance, serialize_method):
            registry.write(instance, *args, **kwargs)    
