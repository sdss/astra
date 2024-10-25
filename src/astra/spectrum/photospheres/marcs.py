
import gzip
from os import replace
import re
import numpy as np
from datetime import datetime
from astropy.io import registry
from collections import OrderedDict

from .photosphere import Photosphere
from grok.utils import periodic_table, safe_open

def parse_meta(contents):

    patterns = [
        "(?P<file_id>^.+)\n",
        "\s+(?P<teff>[\d\.]+)\s+Teff \[(?P<teff_unit>.+)\].+yyyymmdd\=(?P<updated>\d+)\n",
        "\s+(?P<flux>[\d\.E\-\+]+)\s+Flux \[(?P<flux_unit>.+)\]\n",
        "\s+(?P<logg>[\d\.E\-\+]+)\s+Surface gravity \[(?P<surface_gravity_unit>.+)\]\n",
        "\s+(?P<microturbulence>[\d\.]+)\s+Microturbulence parameter \[(?P<microturbulence_unit>.+)\]\n",
        "\s+(?P<mass>[\d\.]+)\s+(No m|m|M)ass.+\n", #\s+Mass \[(?P<mass_unit>.+)\]\n"
        "\s+(?P<m_h>[\d\.\+\-]+)\s(?P<alpha_m>[\d\-\.\+]+) Metallicity.+\n",
        "\s+(?P<radius>[\d\.E\-\+]+).+(1 cm r|R)adius.+\n", #Radius \[(?P<radius_unit>.+)\] at Tau\(Rosseland\)=(?P<tau_rosseland_at_radius>[\d\.]+)\n"    
        "\s+(?P<luminosity>[\d\.E\-\+]+) Luminosity \[(?P<luminosity_unit>.+)\].*\n",
        "\s+(?P<convection_alpha>[\d\.E\-\+]+)\s+(?P<convection_nu>[\d\.E\-\+]+)\s+(?P<convection_y>[\d\.E\+]+)\s+(?P<convection_beta>[\d\.E\+]+).+\n",
        "\s+(?P<X>[\d\.E\-\+]+)\s+(?P<Y>[\d\.E\-\+]+)\s+(?P<Z>[\d\.E\-\+]+)\s+are X, Y and Z, +12C\/13C=(?P<isotope_ratio_12C_13C>\d+).*\n",
    ]
    meta = OrderedDict([])
    for pattern in patterns:
        meta.update(next(re.finditer(pattern, contents)).groupdict())
        
    pattern = "Logarithmic chemical number abundances, H always 12.00\n"
    for i, element in enumerate(periodic_table[:92], start=1):
        pattern += f"\s+(?P<log_abundance_{element}>[\d\.\-]+)"
        if (i % 10) == 0:
            pattern += "\n"

    pattern += (
        "\n"
        "\s+(?P<n_depth>\d+) Number of depth points"
    )
    if isinstance(contents, bytes):
        contents = contents.decode("utf-8")

    meta.update(next(re.finditer(pattern, contents)).groupdict())

    # Assign dtypes.
    dtype_patterns = [
        ("teff$", float),
        ("flux$", float),
        ("logg", float),
        ("microturbulence$", float),
        ("mass$", float),
        ("m_h", float),
        ("alpha_m", float),
        ("radius$", float),
        ("tau_rosseland_at_radius", float),
        ("luminosity$", float),
        ("convection_alpha", float),
        ("convection_nu", float),
        ("convection_y", float),
        ("convection_beta", float),
        ("isotope_ratio_12C_13C", float),
        ("X", float),
        ("Y", float),
        ("Z", float),
        ("log_abundance_\w+", float),
        ("n_depth", int),
        ("updated", lambda date: datetime.strptime(date, "%Y%m%d")),
    ]

    for key in meta.keys():
        for pattern, dtype in dtype_patterns:
            if re.match(pattern, key):
                meta[key] = dtype(meta[key])
                break
            
    # We want to keep the position of the surface gravity in the dict, but what we have is not
    # actually log10(surface gravity), it's just surface gravity.
    meta["logg"] = np.log10(meta["logg"])

    # TODO: probably more than that
    meta["grid_keywords"] = ("teff", "logg", "m_h",  "alpha_m", "microturbulence", "mass")
    meta["read_format"] = "marcs"
    return meta


def loadtxt(filename, skiprows, max_rows, replace_as_nan="******"):
    
    can_opener = gzip.open if filename.lower().endswith(".gz") else open
    with can_opener(filename, "r") as fp:
        lines = fp.readlines()

    data = []
    for line in lines[skiprows:skiprows + max_rows]:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if replace_as_nan is not None:
            line = line.replace(replace_as_nan, "NaN")
        data.append([each.group() for each in re.finditer("(-?\d{1,6}(\.\d{1,6})?(E[\-\+]\d{1,2})?|(NaN))", line)])
    return np.array(data, dtype=float)


def write_marcs(photosphere, path):
    
    if photosphere.is_spherical_geometry:
        radius_desc = "Radius [cm] at Tau(Rosseland)=1.0"
        luminosity_desc = ""
        mass_desc = "Mass [Msun]"
    else:
        radius_desc = "1 cm radius for plane-parallel models"
        luminosity_desc = "FOR A RADIUS OF 1 cm!"
        mass_desc = "No mass for plane-parallel models"

    meta = photosphere.meta
    contents = f"""
  {meta['teff']:.0f}.      Teff [K].
  {meta['flux']:1.4E} Flux [erg/cm2/s]
  {10**meta['logg']:1.4E} Surface gravity [cm/s2]
  {meta['microturbulence']:3.1f}        Microturbulence parameter [km/s]
  {meta['mass']:3.1f}        {mass_desc}
 {meta['m_h']:+.2f} {meta['alpha_m']:+.2f} Metallicity [Fe/H] and [alpha/Fe]
  {meta['radius']:1.4E} {radius_desc}
  {meta['luminosity']:1.4E} Luminosity [Lsun] {luminosity_desc}
  {meta['convection_alpha']:.2f} {meta['convection_nu']:.2f} {meta['convection_y']:.3f} {meta['convection_beta']:.2f} are the convection parameters: alpha, nu, y, beta
  {meta['X']:.5f} {meta['Y']:.5f} {meta['Z']:.2E} are X, Y and Z, +12C/13C={meta['isotope_ratio_12C_13C']:.0f}
Logarithmic chemical number abundances, H always 12.00
"""
    for element in periodic_table[:92]:
        contents += f" {meta['log_abundance_' + element]: >6.2f}"
        if (periodic_table.index(element) + 1) % 10 == 0:
            contents += "\n"
    
    contents += f"\n  {len(photosphere)} Number of depth points\n"
    contents += "Model structure\n"
    contents += " k lgTauR  lgTau5    Depth     T        Pe         Pg        Prad       Pturb\n"
    depth_sign = +1 if photosphere["Depth"][0] < 0 else -1
    for k, row in enumerate(photosphere, start=1):
        contents += f"{k: >3.0f} {row['lgTauR']:+1.2f} {row['lgTau5']:+1.4f} {depth_sign * row['Depth']:+1.3E} {row['T']: >7.1f} {row['Pe']: >10.3E} {row['Pg']: >10.3E} {row['Prad']: >10.3E} {row['Pturb']: >10.3E}\n"

    contents += " k lgTauR  KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX\n"
    for k, row in enumerate(photosphere, start=1):
        contents += f"{k: >3.0f} {row['lgTauR']:+1.2f} {row['KappaRoss']: >10.3E} {row['Density']: >10.3E} {row['Mu']:1.3f}  0.000E+00 0.00000 {row['RHOX']: >13.6E}\n"

    contents += "Assorted logarithmic partial pressures\n"
    contents += " k  lgPgas   H I    H-     H2     H2+    H2O    OH     CH     CO     CN     C2\n"
    for k, row in enumerate(photosphere, start=1):
        contents += f"{k: >3.0f} {row['lgPgas']: >6.3f} {row['H I']: >6.2f} {row['H-']: >6.2f} {row['H2']: >6.2f} {row['H2+']: >6.2f} {row['H2O']: >6.2f} {row['OH']: >6.2f} {row['CH']: >6.2f} {row['CO']: >6.2f} {row['CN']: >6.2f} {row['C2']: >6.2f}\n"

    contents += " k    N2     O2     NO     NH     TiO   C2H2    HCN    C2H    HS     SiH    C3H\n"
    for k, row in enumerate(photosphere, start=1):
        contents += f"{k: >3.0f} {row['N2']: >6.2f} {row['O2']: >6.2f} {row['NO']: >6.2f} {row['NH']: >6.2f} {row['TiO']: >6.2f} {row['2H2']: >6.2f} {row['HCN']: >6.2f} {row['C2H']: >6.2f} {row['HS']: >6.2f} {row['SiH']: >6.2f} {row['C3H']: >6.2f}\n"
    contents += " k    C3     CS     SiC   SiC2    NS     SiN    SiO    SO     S2     SiS   Other\n"
    for k, row in enumerate(photosphere, start=1):
        contents += f"{k: >3.0f} {row['C3']: >6.2f} {row['CS']: >6.2f} {row['SiC']: >6.2f} {row['SiC2']: >6.2f} {row['NS']: >6.2f} {row['SiN']: >6.2f} {row['SiO']: >6.2f} {row['SO']: >6.2f} {row['S2']: >6.2f} {row['SiS']: >6.2f} {row['Other']: >6.2f}\n"

    with open(path, "w") as fp:
        fp.write(contents)

def read_marcs(fp_or_path, structure_start=25, __include_extra_columns=True):
    
    filename, contents, content_stream = safe_open(fp_or_path)

    meta = parse_meta(contents)
    meta["filename"] = filename

    S, N = (structure_start, meta["n_depth"])
    column_locations = [
        [S, ("k", "lgTauR", "lgTau5", "Depth", "T", "Pe", "Pg", "Prad", "Pturb")],
        [S + N + 1, ("k", "lgTauR", "KappaRoss", "Density", "Mu", "Vconv", "Fconv/F", "RHOX")],
        [S + (N + 1)*2 + 1, ("k", "lgPgas", "H I", "H-", "H2", "H2+", "H2O", "OH", "CH", "CO", "CN", "C2")],
        [S + (N + 1)*3 + 1, ("k", "N2", "O2", "NO", "NH", "TiO", "2H2", "HCN", "C2H", "HS", "SiH", "C3H")],
        [S + (N + 1)*4 + 1, ("k", "C3", "CS", "SiC", "SiC2", "NS", "SiN", "SiO", "SO", "S2", "SiS", "Other")]
    ]

    data = OrderedDict([])
    for skiprows, column_names in column_locations:
        values = loadtxt(filename, skiprows=skiprows, max_rows=N)
        data.update(dict(zip(column_names, values.T)))

    column_descriptions = {
        # See https://marcs.astro.uu.se/documents/auxiliary/readmarcs.f
        "k": ("Depth point", "-"),
        "lgTauR": ("log(tau(Rosseland))", "-"),
        "lgTau5": ("log(tau(5000 Angstroms))", "-"),
        "Depth": ("Depth (depth=0 and tau(Rosseland)=1.0)", "cm"),
        "T": ("Temperature", "K"),
        "Pe": ("Electron pressure", "dyn/cm^2"),
        "Pg": ("Gas pressure", "dyn/cm^2"),
        "Prad": ("Radiation pressure", "dyn/cm^2"),
        "Pturb": ("Turbulence pressure", "dyn/cm^2"),
        "KappaRoss": ("Rosseland opacity", "cm^2/g"),
        "Density": ("Density", "g/cm^3"),
        "Mu": ("Mean molecular weight", "amu"),
        "Vconv": ("Convective velocity", "cm/s"),
        "Fconv/F": ("Fractional convective flux", "-"),
        "RHOX": ("Column mass above point k", "g/cm^2"),
    }
    # Add descriptions for partial pressure columns
    index = list(data.keys()).index("H I")
    for partial_pressure_column_name in list(data.keys())[index:]:
        # For Other, change the description.
        if partial_pressure_column_name == "Other":
            desc = "pressure of not listed gas constituents"
        else:
            desc = partial_pressure_column_name
        column_descriptions[partial_pressure_column_name] = (
            f"log({desc} pressure/[1 dyn/cm^2])",
            "-"
        )

    descriptions = { k: desc for k, (desc, unit) in column_descriptions.items() }
    units = { k: unit for k, (desc, unit) in column_descriptions.items() }

    data["k"] = data["k"].astype(int)

    photosphere = Photosphere(
        data=data,
        units=units,
        descriptions=descriptions,
        meta=meta
    )

    # Amazingly, despite the number of columns here, the MARCS models in ".mod"
    # format do not store electron density. That is stored in the Kurucz format.
    # We might be able to calculate it, but I'm not certain we'd be doing it
    # self-consistently. Instead we are going to load the Kurucz equivalent
    # photosphere and supplement this one with electron density.
    
    # TODO: This is an awful hack that relies on you having both MARCS formats.
    #       It's only necessary for MOOG. God it's all a hack.
    
    # TODO: What we should do is calculate the number densities from the gas pressure
    #       and the electron pressure.
    if __include_extra_columns:
        alt_filename = filename.replace("marcs_mod", "marcs_krz").replace(".mod", ".krz")
        alt_photosphere = Photosphere.read(alt_filename, __include_extra_columns=False)
        extra_columns = list(set(alt_photosphere.dtype.names).difference(photosphere.dtype.names))
        if alt_photosphere.is_spherical_geometry:
            # You won't believe this, but the "Depth" is recorded in the KRZ and MOD formats, but in
            # the MOD format the Depth goes from negative to positive, and then in KRZ format the 
            # Depth goes from positive to negative.
            
            # The KRZ format has more significant digits recorded, and it is setup in in the correct direction.
            # So we will use that instead.
            extra_columns.append("Depth")
        
            # Oh god it gets worse.
            # Turbospectrum *expects* the Depth to go from negative to positive.
            # Korg sensibly expects the Depth to go from positive to negative.

        for key in extra_columns:
            photosphere[key] = alt_photosphere[key]
        
    return photosphere

def identify_marcs(origin, *args, **kwargs):
    return (isinstance(args[0], str) and \
            args[0].lower().endswith((".mod", ".mod.gz")))

registry.register_reader("marcs", Photosphere, read_marcs)
registry.register_writer("marcs", Photosphere, write_marcs)
registry.register_identifier("marcs", Photosphere, identify_marcs)
