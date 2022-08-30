import numpy as np
from functools import cached_property


class BitMask(object):

    """Base class for bitmasks."""

    def get_name(self, val, level=0, strip=True):
        """
        Given input value, returns names of all set bits, optionally of a given level
        """
        strflag = ""
        for ibit, name in enumerate(self.name):
            if (np.uint64(val) & np.uint64(2**ibit)) > 0 and (
                level == 0 or self.level == level
            ):
                strflag = strflag + name + ","
        if strip:
            return strflag.strip(",")
        else:
            return strflag

    def get_value(self, name):
        """
        Get the numerical bit value of a given character name(s)
        """
        if type(name) is str:
            name = [name]
        bitval = np.uint64(0)
        for n in name:
            try:
                j = self.name.index(n.strip())
                bitval |= np.uint64(2**j)
            except:
                print("WARNING: undefined name: ", n)
        return bitval

    @cached_property
    def bad_value(self):
        """
        Return bitmask value of all bits that indicate BAD in input bitmask
        """
        val = np.uint64(0)
        for i, level in enumerate(self.level):
            if level == 1:
                val = val | np.uint64(2**i)
        return val

    @cached_property
    def warn_value(self):
        """
        Return bitmask value of all bits that indicate BAD in input bitmask
        """
        val = np.uint64(0)
        for i, level in enumerate(self.level):
            if level == 2:
                val = val | np.uint64(2**i)
        return val
