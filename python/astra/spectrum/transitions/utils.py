


def vacuum_to_air(lambdas):
    """
    Convert vacuum wavelengths to air.

    The formula from Donald Morton (2000, ApJ. Suppl., 130, 403) is used for the 
    refraction index, which is also the IAU standard.

    As per https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    :param lambdas:
        The vacuum wavelengths.
    """
    try:
        unit = lambdas.unit
    except:
        raise ValueError("Missing units on lambdas.")
    
    values = lambdas.to("Angstrom").value
    s = 10**4 / values
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return (values / n) * unit


def air_to_vacuum(lambdas):
    """
    Convert air wavelengths to vacuum.

    As per: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    
    :param lambdas:
        The air wavelengths.
    """
    try:
        unit = lambdas.unit
    except:
        raise ValueError("Missing units on lambdas.")
    
    values = lambdas.to("Angstrom").value
    s = 10**4 / values
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return values * n * unit
    

'''
common_molecule_name2Z = {
    'Mg-H': 12,'H-Mg': 12,
    'C-C':  6,
    'C-N':  7, 'N-C':  7, 
    'C-H':  6, 'H-C':  6,
    'O-H':  8, 'H-O':  8,
    'Fe-H': 26,'H-Fe': 26,
    'N-H':  7, 'H-N':  7,
    'Si-H': 14,'H-Si': 14,
    'Ti-O': 22,'O-Ti': 22,
    'V-O':  23,'O-V':  23,
    'Zr-O': 40,'O-Zr': 40
    }
common_molecule_name2species = {
    'Mg-H': 112,'H-Mg': 112,
    'C-C':  606,
    'C-N':  607,'N-C':  607,
    'C-H':  106,'H-C':  106,
    'O-H':  108,'H-O':  108,
    'Fe-H': 126,'H-Fe': 126,
    'N-H':  107,'H-N':  107,
    'Si-H': 114,'H-Si': 114,
    'Ti-O': 822,'O-Ti': 822,
    'V-O':  823,'O-V':  823,
    'Zr-O': 840,'O-Zr': 840
    }
common_molecule_species2elems = {
    112: ["Mg", "H"],
    606: ["C", "C"],
    607: ["C", "N"],
    106: ["C", "H"],
    108: ["O", "H"],
    126: ["Fe", "H"],
    107: ["N", "H"],
    114: ["Si", "H"],
    822: ["Ti", "O"],
    823: ["V", "O"],
    840: ["Zr", "O"]
}

periodic_table = """H                                                  He
                    Li Be                               B  C  N  O  F  Ne
                    Na Mg                               Al Si P  S  Cl Ar
                    K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                    Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                    Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                    Fr Ra Lr Rf"""

lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"

periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
                               .replace(" Ra ", " Ra " + actinoids + " ")\
                               .split()

def species_as_float(element_repr):
    """
    Converts a string representation of an element and its ionization state
    to a floating point 
    """
            
    if element_repr.count(" ") > 0:
        element, ionization = element_repr.split()[:2]
    else:
        element, ionization = element_repr, "I"
    
    if element not in periodic_table:
        try:
            return common_molecule_name2species[element]
        except KeyError:
            # Don't know what this element is
            return float(element_repr)
    
    ionization = max([0, ionization.upper().count("I") - 1]) /10.
    return periodic_table.index(element) + 1 + ionization
    

def species(species_as_float):
    # TODO: Does not handle molecules yet
    i = int(species_as_float)
    element = periodic_table[i - 1]
    ionization = int(10 * (species_as_float % i))
    return f"{element} {ionization}"

def as_element(species_as_float):
    return species(species_as_float).split(" ")[0]
'''