
import re
import gzip
from typing import ForwardRef
import numpy as np
from collections import OrderedDict
from astropy.io import registry

from .photosphere import Photosphere
from grok.utils import periodic_table, safe_open


def parse_meta(contents):
    """
    
    The `abundance_scale` describes the overall metallicity, in that
    
        [M/H] = log10(abundance_scale)
    """

    # N\(He\)\/Ntot=(?P<n_he_over_n_total>[\d\.]+) 
    pattern = (
        "TEFF\s+(?P<teff>[\d\.]+)\s+GRAVITY\s(?P<logg>[\d\.]+)\s(?P<lte_flag>\w+).*\n"
        "TITLE.+VTURB=(?P<microturbulence>[\d\.]+).+\n"#\s+L/H=(?P<mixing_length>[\d\.]+) (?P<odf_desc>\w+)\s+\n" # TODO: what is [-0.5a]
        " OPACITY IFOP (?P<absorption_by_H>[01]) (?P<absorption_by_H2_plus>[01]) (?P<absorption_by_H_minus>[01]) (?P<rayleigh_scattering_by_H>[01]) (?P<absorption_by_He>[01]) (?P<absorption_by_He_plus>[01]) (?P<absorption_by_He_minus>[01]) (?P<rayleigh_scattering_by_He>[01]) (?P<absorption_by_metals_for_cool_stars_Si_Mg_Al_C_Fe>[01]) (?P<absorption_by_metals_for_intermediate_stars_N_O_Si_plus_Mg_plus_Ca_plus>[01]) (?P<absorption_by_metals_for_hot_stars_C_N_O_Ne_Mg_Si_S_Fe>[01]) (?P<thomson_scattering_by_electrons>[01]) (?P<rayleigh_scattering_by_h2>[01]) (?P<other_opacity_switches>[01\s]+)\n" # TODO
        " CONVECTION (?P<convection>\w+)\s+(?P<convection_mixing_length>[\d\.]+) TURBULENCE (?P<turbulence>\w+)\s+(?P<turbulence_var_1>[\d\.E\-\+]+)\s+(?P<turbulence_var_2>[\d\.E\-\+]+)\s+(?P<turbulence_var_3>[\d\.E\-\+]+)\s+(?P<turbulence_var_4>[\d\.E\-\+]+)\s?\n"
        "ABUNDANCE SCALE\s+(?P<abundance_scale>[\d\.]+) ABUNDANCE CHANGE"
     )
    #normalized abundances for the first 99 elements in the periodic table: N_atom/N_all where N are number densities. The sum of all abundances is 1. Positive numbers are fractions while negative numbers are log10 of fractions. The last number in line 13 is the number of depth layers in the model.
    for i, element in enumerate(periodic_table[:99], start=1):
        pattern += f"\s+{i:.0f}\s+(?P<log10_normalized_abundance_{element}>[\d\-\.]+)"
        if (i - 2) % 6 == 0:
            pattern += "\n ABUNDANCE CHANGE"

    pattern += "\nREAD DECK6 (?P<n_depth>\d+)"

    meta = OrderedDict(re.match(pattern, contents).groupdict())

    # Assign dtypes.
    on_off = lambda v: True if v == "ON" else False
    dtype_patterns = [
        ("teff$", float),
        ("logg", float),
        ("microturbulence$", float),
        ("n_he_over_n_total", float),
        (".*mixing_length$", float),
        ("turbulence_var_", float),
        ("^convection$", on_off),
        ("^turbulence$", on_off),
        ("absorption_by.+", lambda v: bool(int(v))),
        (".+scattering_by.+", lambda v: bool(int(v))),
        ("abundance_scale", float),
        ("log10_normalized_abundance", float),
        ("n_depth", int),
    ]

    for key in meta.keys():
        for pattern, dtype in dtype_patterns:
            if re.match(pattern, key):
                meta[key] = dtype(meta[key])
                break

    for element in periodic_table[:99]:
        key = f"log10_normalized_abundance_{element}"
        if meta[key] > 0:
            meta[key] = np.log10(meta[key])

    pradk, = re.findall("PRADK (?P<pradk>[\dE\-\.\+]+)", contents)
    meta["pradk"] = float(pradk)
    meta["m_h"] = np.round(np.log10(meta["abundance_scale"]), 2)
    # TODO: Actually calculate the abundance by subtracting Solar, instead of this hack.
    meta["alpha_m"] = meta["log10_normalized_abundance_O"] + 3.21
    return meta



def read_atlas9(fp_or_path, structure_start=23):

    filename, contents, content_stream = safe_open(fp_or_path)

    meta = parse_meta(contents)
    meta["filename"] = filename

    S, N = (structure_start, meta["n_depth"])
    column_locations = [
        [S, ("RHOX", "T", "P", "XNE", "ABROSS", "ACCRAD", "VTURB", "FLXCNV", "VCONV", "VELSND")]
    ]

    data = OrderedDict([])
    for skiprows, column_names in column_locations:
        values = np.loadtxt(content_stream, skiprows=skiprows, max_rows=N)
        data.update(dict(zip(column_names, values.T)))

    # According to https://sme.readthedocs.io/en/latest/content/modules.html
    column_descriptions = {
        "RHOX": ("Mass column density", "g/cm^2"),
        "T": ("Temperature", "K"),
        "P": ("Total gas pressure", "dyn/cm^2"),
        "XNE": ("Electron number fraction", "1/cm^3"),
        "ABROSS": ("Rosseland mean mass extinction", "cm^2/g"),
        "ACCRAD": ("Radiative acceleration", "cm/s^2"),
        "VTURB": ("Microturbulence velocity", "cm/s"),
        "FLXCONV": ("Convective flux", "-"),
        "VCONV": ("Velocity of convective cells", "cm/s"),
        "VELSND": ("Local sound speed", "cm/s")
    }

    descriptions = { k: desc for k, (desc, unit) in column_descriptions.items() if k in data }
    units = { k: unit for k, (desc, unit) in column_descriptions.items() if k in data }
    
    # The grid dimensions for these kinds of photospheres is usually:
    # teff, logg, metallicity, alpha, microturbulence
    meta["grid_keywords"] = ("teff", "logg", "m_h", "alpha_m", "microturbulence")

    return Photosphere(
        data=data,
        units=units,
        descriptions=descriptions,
        meta=meta
    )

def identify_atlas9(origin, *args, **kwargs):
    return (isinstance(args[0], str) and \
            args[0].lower().endswith((".dat", ".dat.gz")))

registry.register_reader("atlas9", Photosphere, read_atlas9)
registry.register_identifier("atlas9", Photosphere, identify_atlas9)

