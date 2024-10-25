

from astropy.io import registry
from astropy.table.info import serialize_method_as

class PhotosphereRead(registry.UnifiedReadWrite):

    def __init__(self, instance, cls):
        super().__init__(instance, cls, "read")

    def __call__(self, *args, **kwargs):
        cls = self._cls
        units = kwargs.pop('units', None)
        descriptions = kwargs.pop('descriptions', None)

        out = registry.read(cls, *args, **kwargs)

        out._set_column_attribute('unit', units)
        out._set_column_attribute('description', descriptions)

        # Set the read format.
        if "read_format" not in out.meta:
            out.meta["read_format"] = kwargs.get("format", None)
        return out


class PhotosphereWrite(registry.UnifiedReadWrite):

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'write')

    def __call__(self, *args, serialize_method=None, **kwargs):
        instance = self._instance
        with serialize_method_as(instance, serialize_method):
            registry.write(instance, *args, **kwargs)    
