
import numpy as np
from collections import OrderedDict

class BitFlagNameMap(object):

    def get_names(self, value):
        names = []
        for k, entry in self.__class__.__dict__.items():
            if k.startswith("_") or k != k.upper(): continue
            
            if isinstance(entry, int):
                v = entry
            else:
                v, comment = entry

            if value & (2**v):
                names.append(k)
        
        return tuple(names)

        

    def get_value(self, *names):
        value = np.int64(0)

        for name in names:
            try:
                entry = getattr(self, name)
            except KeyError:
                raise ValueError(f"no bit flag found named '{name}'")
            
            if isinstance(entry, int):
                entry = (entry, "")
            
            v, comment = entry
            value |= np.int64(2**v)
        
        return value


    def get_level_value(self, level):
        try:
            names = self.levels[level]
        except KeyError:
            raise ValueError(f"No level name '{level}' found (available: {' '.join(list(self.levels.keys()))})")

        return self.get_value(*names)
        