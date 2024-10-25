from grok.utils import periodic_table

# Only used to order molecules the right way.
# https://en.wikipedia.org/wiki/Electronegativity
_electronegativity = [
    2.300,                                                                                                        4.160, 
    0.912, 1.576,                                                              2.051, 2.544, 3.066, 3.610, 4.193, 4.787,
    0.869, 1.293,                                                              1.613, 1.916, 2.253, 2.589, 2.869, 3.242,
    0.734, 1.034, 1.19, 1.38, 1.53, 1.65, 1.75, 1.80, 1.84, 1.88, 1.85, 1.588, 1.756, 1.994, 2.211, 2.424, 2.685, 2.966,
    0.706, 0.963, 1.12, 1.32, 1.41, 1.47, 1.51, 1.54, 1.56, 1.58, 1.87, 1.521, 1.656, 1.824, 1.984, 2.158, 2.359, 2.582,
    0.659, 0.881, 1.09, 1.16, 1.34, 1.47, 1.60, 1.65, 1.68, 1.72, 1.92, 1.765, 1.789, 1.854, 2.010, 2.190, 2.390, 2.600,
    0.670, 0.890
]

class Formula(object):
    
    _Nz = 3 # support up to tri-atomic molecules

    def __init__(
            self, 
            representation
        ):

        # We assume that if a tuple is given, you (mostly) know what you're doing.
        if isinstance(representation, (tuple, list)):
            Zs = list(representation)
        else:
            # Try various things.
            if isinstance(representation, int):
                if 1 <= representation < len(periodic_table):
                    Zs = [representation]
                else:
                    raise ValueError("Must be some molecule, but you should use string representation.")
    
            elif isinstance(representation, str):
                if representation.isdigit():
                    # Numeric codes like 0801 -> OH
                    L = len(representation)
                    if L <= 2:
                        Zs = [int(representation)]
                    elif L <= 4:
                        code = f"{representation:0>4s}"
                        Z1 = int(code[0:2])
                        Z2 = int(code[2:4])
                        Zs = [Z1, Z2]
                    else:
                        raise ValueError(f"Numeric codes for molecules with more than 4 chars like "
                                         f"'{representation}' are not supported.")
                else:
                    # Should be something like "OH", "FeH", "Li", "C2" (strict)
                    # But sometimes people put HE for He which is really irritating (not strict)
                    for strict in (True, False):
                        try:
                            Zs = parse_formula_string(representation, strict=strict)
                        except ValueError:
                            continue
                        else:
                            break
                    else:
                        raise

            else:
                raise TypeError(f"Invalid type representation ({type(representation)}) for '{representation}'")

        # Prepend with zeros.
        Zs = (([0] * self._Nz) + Zs)[-self._Nz:]

        # We don't sort here because we don't know the isotope information, and we would have to order that too.
        self.Zs = tuple(Zs)
        return None



    def __repr__(self):
        return f"<{self}>"

    def __str__(self):
        # For string representations we should order by electropositivity.

        return "".join(self.atoms)

    @property
    def atoms(self):
        return tuple([periodic_table[Z-1] for Z in self.Zs if Z > 0])


def parse_formula_string(representation, strict=True):
    if strict:
        indices = [i for i, char in enumerate(representation) if char.isdigit() or char.isupper()]
        indices.append(len(representation))
    else:
        try:
            Zs = [periodic_table.index(representation.title())]
        except ValueError:
            # Something like "FEH"
            indices = [0]
            Zs = []
            ws = (2, 1)
            while True:
                for w in ws:
                    si = indices[-1]
                    chars = representation[si:si+w].title()
                    try:
                        Zs.append(periodic_table.index(chars.title()))
                    except ValueError:
                        continue
                    else:
                        indices.append(si + w)                        
                        break
                else:
                    raise ValueError(f"Couldn't find '{representation[si:si+max(ws)]}' in the periodic table!")

                if len(representation[si+w:]) == 0:
                    break
        finally:
            return Zs

    codes = []
    for j, index in enumerate(indices[:-1]):
        codes.append(representation[index:indices[j+1]])

    Zs = []
    for sub_code in codes:
        if sub_code.isnumeric():
            previous_Z = Zs[-1]
            for j in range(int(sub_code) - 1):
                Zs.append(previous_Z)
        else:
            Zs.append(periodic_table.index(sub_code) + 1)
    
    return Zs
