import re
from grok.transitions.formula import Formula

roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

class Species(object):

    def __init__(self, representation):

        representation = representation.strip(" 0")
        tokens = [token for token in re.split("[\s\._]", representation) if len(token) > 0]
        if len(tokens) > 2:
            raise ValueError(f"Invalid species code '{representation}' ({type(representation)})")
        
        self.formula = Formula(tokens[0])

        N = len(self.formula.Zs)
        isotopes = tuple([0] * N)
        if len(tokens) == 1 or len(tokens[1]) == 0:
            charge = 0
        else:
            if tokens[1] in roman_numerals:
                charge = roman_numerals.index(tokens[1])
            elif tokens[1].isdigit():
                # Turbospectrum encodes isotopic information with remaining digits,
                # but MOOG encodes both charge and isotopic information.
                # TODO:
                # Take the first digit only.
                charge = int(tokens[1][0])
                if len(tokens[1]) > 1:
                    s, w = (0, 3) # start, width
                    isotopes = []
                    for i in range(len(self.formula.atoms)):
                        isotopes.append(int(tokens[1][s+i*w:s+(i + 1)*w]))

                    isotopes = tuple((([0] * N) + isotopes)[-N:])
            else:
                raise ValueError(f"Invalid charge '{tokens[1]}' in '{representation}'")

        self.charge = charge
        self.isotopes = isotopes

        return None

    def __repr__(self):
        return f"<{self}>"

    def __str__(self):
        return f"{''.join(self.atoms)} {roman_numerals[self.charge]}"

    @property
    def atoms(self):
        return self.formula.atoms

    @property
    def Zs(self):
        return self.formula.Zs

    
    @property
    def compact(self):
        """
        A compact representation of the species.
        """

        compact_formula = "".join([f"{Z:0>2.0f}" for Z in self.Zs if Z > 0])
        compact_isotope = "".join([f"{I:0>3.0f}" for Z, I in zip(self.Zs, self.isotopes) if Z > 0])
        return f"{compact_formula}.{self.charge:.0f}{compact_isotope}"


