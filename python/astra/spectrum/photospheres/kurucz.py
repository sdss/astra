
import re
import gzip
import numpy as np
from collections import OrderedDict
from astropy.io import registry
from textwrap import dedent

from .photosphere import Photosphere
from grok.utils import periodic_table, safe_open


def parse_meta(contents):
    # Documented at https://marcs.astro.uu.se/krz_format.html

    # Check if spherical or not. Could also do this by checking for model type:
    # MODEL TYPE = 3 (spherical)
    # MODEL TYPE = 0 (plane parallel)
    spherical_pattern = ".+SPHERICAL, RADIUS=\s(?P<radius>[\d\.E\+]+)\s+(?P<radius_unit>.+)"
    is_spherical = any(re.findall(spherical_pattern, contents))
    
    pattern = (
        "(?P<header>^.+)\n"# V[turb|TURB]\s+(?P<microturbulence>[\d\.]+) km/s.*\n"
        "T EFF=\s?(?P<teff>[\d\.]+)\s+GRAV=\s?(?P<logg>[\-\d\.]+)\s+MODEL TYPE= (?P<model_type>\d) WLSTD= (?P<standard_wavelength>\d+)"
    )
    if is_spherical:
        pattern += spherical_pattern + ".+\n"
    else:
        pattern += ".+\n"

    pattern += "\s*(?P<absorption_by_H>[01]) (?P<absorption_by_H2_plus>[01]) (?P<absorption_by_H_minus>[01]) (?P<rayleigh_scattering_by_H>[01]) (?P<absorption_by_He>[01]) (?P<absorption_by_He_plus>[01]) (?P<absorption_by_He_minus>[01]) (?P<rayleigh_scattering_by_He>[01]) (?P<absorption_by_metals_for_cool_stars_Si_Mg_Al_C_Fe>[01]) (?P<absorption_by_metals_for_intermediate_stars_N_O_Si_plus_Mg_plus_Ca_plus>[01]) (?P<absorption_by_metals_for_hot_stars_C_N_O_Ne_Mg_Si_S_Fe>[01]) (?P<thomson_scattering_by_electrons>[01]) (?P<rayleigh_scattering_by_h2>[01]).+OPACITY SWITCHES\n"

    #normalized abundances for the first 99 elements in the periodic table: N_atom/N_all where N are number densities. The sum of all abundances is 1. Positive numbers are fractions while negative numbers are log10 of fractions. The last number in line 13 is the number of depth layers in the model.
    for i, element in enumerate(periodic_table[:99], start=1):
        pattern += f"\s*(?P<log10_normalized_abundance_{element}>[\-\d\.]+) "
        if i % 10 == 0:
            pattern = pattern.rstrip() + "\n"

    pattern += "\s*(?P<n_depth>\d+)"
    
    meta = OrderedDict(re.match(pattern, contents).groupdict())
    if not is_spherical:
        meta.update(radius=0, radius_unit="cm")
    
    # Assign dtypes.
    dtype_patterns = [
        ("teff$", float),
        ("flux$", float),
        ("logg$", float),
        ("radius$", float),
        ("standard_wavelength", float),
        ("log10_normalized_abundance_\w+", float),
        ("n_depth", int),
        ("absorption_by", lambda v: bool(int(v))),
        ("scattering_by", lambda v: bool(int(v))),
    ]
    for key in meta.keys():
        for pattern, dtype in dtype_patterns:
            if re.findall(pattern, key):
                meta[key] = dtype(meta[key])
                break    

    # Correct the normalized abundance columns so they are all log10 base.
    # Numbers that are positive are not yet log10
    for element in periodic_table[:99]:
        key = f"log10_normalized_abundance_{element}"
        if meta[key] > 0:
            meta[key] = np.log10(meta[key])

    # Add overall metallicity keyword, taking Fe as representative of metallicity.
    # Remember that the normalized_abundance keywords are number densities, so to convert to the "12 scale" we need:
    #   log_10(X) = log_10(N_X/N_H) + 12
    # and for [X/H] we need:
    #   [X/H] = log_10(X) - log_10(X_Solar)

    # TODO: Assuming 7.45 Fe solar composition. Check this.
    meta["m_h"] = meta["log10_normalized_abundance_Fe"] - meta["log10_normalized_abundance_H"] + 12 - 7.45
    
    # TODO: Should we just calculate abundances for ease?
    meta["grid_keywords"] = ("teff", "logg", "m_h")
    return meta



def read_kurucz(fp_or_path, structure_start=13, __include_extra_columns=True):

    filename, contents, content_stream = safe_open(fp_or_path)

    meta = parse_meta(contents)
    model_type = int(meta["model_type"])
    if model_type in (0, 1):
        # Plane-parallel.
        usecols = (0, 1, 2, 3, 4)
    elif model_type in (3, 4):
        # Spherical geometry.
        usecols = (0, 1, 2, 3, 4, 5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if model_type in (0, 3):
        # depth scale - column mass in g/cm^2
        column_locations = [
            (structure_start, ("RHOX", "T", "XNE", "numdens_other", "Density", "Depth"))
        ]
    elif model_type in (1, 4):
        # depth scale - tau at std wavelength
        column_locations = [
            (structure_start, ("tau", "T", "XNE", "numdens_other", "Density", "Depth"))
        ]      
    
    data = {}
    for skiprows, column_names in column_locations:
        values = np.loadtxt(
            content_stream, 
            skiprows=skiprows, 
            max_rows=meta["n_depth"],
            delimiter=",",
            usecols=usecols
        )
        assert values.shape[0] == meta["n_depth"]
        data.update(dict(zip(column_names, values.T)))

    column_descriptions = {
        "RHOX": ("Mass column density", "g/cm^2"),
        "tau": ("Optical depth at standard wavelength", "-"),
        "T": ("Temperature", "K"),
        "XNE": ("Number density of free electrons", "1/cm^3"),
        "numdens_other": ("Number density of all other particles (except electrons)", "1/cm^3"),
        "Density": ("Density", "g/cm^3"),
        "Depth": ("Height relative to the reference radius given in the metadata", "cm")
    }
    descriptions = { k: desc for k, (desc, unit) in column_descriptions.items() if k in data }
    units = { k: unit for k, (desc, unit) in column_descriptions.items() if k in data  }
    
    photosphere = Photosphere(
        data=data,
        units=units,
        descriptions=descriptions,
        meta=meta
    )

    # See marcs.py comment on this. 
    # MOOG needs more things than the krz format can offer.
    
    # TODO: This is an awful hack that relies on you having both MARCS formats.
    #       It's only necessary for MOOG. God it's all a hack.
    if __include_extra_columns:
        alt_filename = filename.replace("marcs_krz", "marcs_mod").replace(".krz", ".mod")
        alt_photosphere = Photosphere.read(alt_filename, __include_extra_columns=False)
        for key in ("KappaRoss", "P"):
            photosphere[key] = alt_photosphere[key]

    return photosphere

def write_kurucz(photosphere, path, **kwargs):
    """
    Write a photosphere in Kurucz format, documented at:
    https://marcs.astro.uu.se/krz_format.html.

    :param photosphere:
        The photosphere.
    
    :path:
        The local path.
    """
    meta = photosphere.meta
    # Always use either model_type of 0 or 3, which outputs column mass as first column
    first_column_name = "RHOX"
    if photosphere.is_spherical_geometry:
        model_type = 3 
        geometry_desc = f"SPHERICAL,  RADIUS= {meta['radius']:.3e} cm"
    else:
        model_type = 0
        geometry_desc = "PLANE-PARALLEL."

    opacity_keys = (
        "absorption_by_H",
        "absorption_by_H2_plus",
        "absorption_by_H_minus",
        "rayleigh_scattering_by_H",
        "absorption_by_He",
        "absorption_by_He_plus",
        "absorption_by_He_minus",
        "rayleigh_scattering_by_He",
        "absorption_by_metals_for_cool_stars_Si_Mg_Al_C_Fe",
        "absorption_by_metals_for_intermediate_stars_N_O_Si_plus_Mg_plus_Ca_plus",
        "absorption_by_metals_for_hot_stars_C_N_O_Ne_Mg_Si_S_Fe",
        "thomson_scattering_by_electrons",
        "rayleigh_scattering_by_h2"
    )
    opacity_switches = [2 if meta.get(k, None) is None else int(meta.get(k, None)) for k in opacity_keys]

    contents = dedent(
    f"""
    {meta.get('header', 'NO HEADER')}
    T EFF={meta['teff']:.0f}. GRAV= {meta['logg']:.2f} MODEL TYPE= {model_type} WLSTD= {meta.get('standard_wavelength', 5000):.0f} {geometry_desc}
    {' '.join(map(str, opacity_switches))} - OPACITY SWITCHES
    """).lstrip()

    N_elements = 99
    normalized_abundances = []
    for i, element in enumerate(periodic_table[:N_elements]):
        value = meta.get(f'log10_normalized_abundance_{element}', -20)
        if 10**value > 0.01:
            value = 10**value
        if element in ("H", "He"):
            normalized_abundances.append(f"{value:1.3f}")
        else:
            normalized_abundances.append(f"{value:1.2f}")

    n_per_line = 10
    for i in range(int(np.ceil(N_elements / n_per_line))):
        si, ei = (i * n_per_line, (i + 1) * n_per_line)
        contents += " ".join(normalized_abundances[si:ei]) + "\n"

    contents = contents.rstrip() + f" {len(photosphere):.0f}\n"
    
    for row in photosphere:
        
        contents += f"{row[first_column_name]:1.9e},{row['T']: >7.1f},{row['XNE']: >12.5e},{row['numdens_other']: >12.5e},{row['Density']: >12.5e},"
        if photosphere.is_spherical_geometry:
            contents += f"{row['Depth']: >12.5e},"
        contents += "\n"

    with open(path, "w") as fp:
        fp.write(contents)
    
    return None


def identify_kurucz(origin, *args, **kwargs):
    return (isinstance(args[0], str) and \
            args[0].lower().endswith((".krz", ".krz.gz")))

registry.register_reader("kurucz", Photosphere, read_kurucz)
registry.register_writer("kurucz", Photosphere, write_kurucz)
registry.register_identifier("kurucz", Photosphere, identify_kurucz)