import numpy as np
from textwrap import dedent
from astropy.io import registry
import re
from grok.photospheres import Photosphere
from grok.solar import periodic_table 
from grok import DO_NOT_SCALE_ABUNDANCES

# Page 16 of WRITEMOOG.ps would make you think that we should include all of
# these species, but if we do that then it fucks everything up.
# default_molecules = (
#    1.1,     2.1,     6.1,     7.1,     8.1,    12.1,
#   13.1,    14.1,    15.1,    16.1,    17.1,    20.1,
#   22.1,    23.1,    24.1,    26.1,
#  101.0,   106.0,   107.0,   108.0,   109.0,   112.0,
#  113.0,   114.0,   115.0,   116.0,   117.0,   120.0,
#  124.0,   126.0, 10106.0, 10107.0, 10108.0, 10115.0,
# 10116.0, 10608.0, 10812.0, 10813.0, 10820.0,   606.0,
#  607.0,   608.0,   616.0, 60808.0,   707.0,   708.0,
#  714.0,   715.0,   716.0,   808.0,   812.0,   814.0,
#  815.0,   816.0,   822.0,   823.0,   826.0,  1416.0
# )
# This, however, works.
# TODO: revisit this when you've fixed interpolation.
default_molecules = (
    606.0,
    106.0,
    607.0,
    608.0,
    107.0,
    108.0,
    112.0,
    707.0,
    708.0,
    808.0,
    12.1,
    60808.0,
    10108.0,
    101.0,
    6.1,
    7.1,
    8.1,
    822.0,
    22.1,
)


def write_photosphere_for_moog(photosphere, path, include_molecules=default_molecules, solar_abundances=None):
    """
    Write a photosphere to disk in a format that is known to MOOG.
    
    :param photosphere:
        The photosphere, an instance of `photospheres.Photosphere`.
    
    :param path:
        The path to store the photosphere.
    
    """

    format = "KURUCZ"
    if photosphere.meta["read_format"] == "atlas9":
        line_format = " {line[RHOX]:.8e} {line[T]:10.3e}{lie[Pg]:10.3e}{line[XNE]:10.3e}{line[ABROSS]:10.3e}"
    else:
        line_format = " {line[RHOX]:.8e} {line[T]:10.3e}{line[Pg]:10.3e}{line[XNE]:10.3e}{line[KappaRoss]:10.3e}"

    """
    available_formats = {
        #"NEWMARCS": " {line[lgTau5]:.8e} {line[T]:10.3e}{line[Pe]:10.3e}{line[Pg]:10.3e}{rho}{line[KappaRoss]:10.3e}"
        # WEBMARCs wants Ne, but if we give Pe then MOOG will convert it.
        # Recall: Ne = (Pe [dyn/cm^2] / T [K]) / (k_Boltzmann [erg / K])
        #"WEBMARCS": " {i:>3.0f} {i:>3.0f} {line[lgTau5]:10.3e} {i:>3.0f} {line[T]:10.3e} {line[Pe]:10.3e} {line[Pg]:10.3e}",
        "KURUCZ": " {line[RHOX]:.8e} {line[T]:10.3e}{line[P]:10.3e}{line[XNE]:10.3e}{line[ABROSS]:10.3e}",
        #"NEXTGEN":
        #"BEGN": " {line[lgTauR]:.8e} {line[T]:10.3e}{line[Pe]:10.3e}{line[Pg]:10.3e}{line[Mu]:10.3e}{line[KappaRoss]:10.3e}"
        #"KURTYPE":
        #"KUR-PADOVA":
        #"GENERIC": 
    }
    format = str(format).strip().upper()
    try:
        line_format = available_formats[format]
    except KeyError:
        raise ValueError(f"Format '{format}' not one of the known formats: {', '.join(list(available_formats.keys()))}")
    """

    output = dedent(
        f"""
        {format}
            PHOTOSPHERE {photosphere.meta['teff']:.0f} / {photosphere.meta['logg']:.3f} / {photosphere.meta['m_h']:.3f} / {photosphere.meta.get('microturbulence', 0):.3f}
        NTAU       {photosphere.meta['n_depth']}
        """
    ).lstrip()

    # Add standard wavelength
    if format == "WEBMARCS":
        output += "5000.0\n"

    for i, line in enumerate(photosphere, start=1):
        output += line_format.format(i=i, line=line) + "\n"

    output += f"        {photosphere.meta['microturbulence']:.3f}\n"
    if solar_abundances is None:
        output += f"NATOMS        0     {photosphere.meta['m_h']:.3f}\n"
    else:
        output += f"NATOMS        {len(solar_abundances)}     {photosphere.meta['m_h']:.3f}\n"
        for element, abundance in solar_abundances.items():
            atomic_number = 1 + periodic_table.index(element)
            if element in DO_NOT_SCALE_ABUNDANCES:
                value = abundance
            else:
                value = abundance + photosphere.meta['m_h']
            output += f"  {atomic_number:.0f} {value:.3f}\n"
    output += f"NMOL        {len(include_molecules): >3}\n"
    for i, molecule in enumerate(include_molecules):
        output += f"{molecule: >10.1f}"
        if i > 0 and i % 8 == 0:
            output += "\n"
        
    # MOOG11 fails to read if you don't add an extra line
    output += "\n"

    with open(path, "w") as fp:
        fp.write(output)

    return None


def parse_single_spectrum(lines):
    """
    Parse the header, dispersion and depth information for a single spectrum 
    that has been output to a summary synthesis file.

    :param lines:
        A list of string lines partitioned from a MOOG summary output file.

    :returns:
        A three-length tuple containing the (1) header rows, (2) an array of
        dispersion values, and (3) an array of intensity values.
    """

    # Skip over the header information
    for i, line in enumerate(lines):
        if line.startswith("MODEL:"):
            break

    else:
        raise ValueError("could not find model information for spectrum")

    # Get the dispersion information.
    start, end, delta, _ = np.array(lines[i + 1].strip().split(), dtype=float)

    # If MOOG doesn't have opacity contributions at a given wavelength, it will
    # just spit out ****** entries.
    # _pre_format = lambda l: l.strip().replace("*******", " 0.0000").replace("-0.0000"," 0.0000").split()
    def _pre_format(l):
        l = l.replace("*******", " 0.0000").rstrip()
        assert len(l) % 7 == 0, len(l)
        return [l[7 * i : 7 * (i + 1)] for i in range(len(l) // 7)]

    depths = np.array(
        sum([_pre_format(line) for line in lines[i + 2 :]], []), dtype=float
    )

    dispersion = np.arange(start, end + delta, delta)[: depths.size]
    intensity = 1.0 - depths

    # Parse the headers into metadata
    meta = {"raw": lines[: i + 2]}
    return (dispersion, intensity, meta)


def parse_standard_synth_output(standard_out_path):
    """
    Parse the standard output from a MOOG synth operation.
    
    :param standard_out_path:
        The path of the standard output file produced by MOOG.
    """

    with open(standard_out_path, "r") as fp:
        stdout = fp.read()

    # Parse the continuum
    pattern = "AT WAVELENGTH/FREQUENCY = \s+(?P<wavelength>[0-9\.]+)\s+CONTINUUM FLUX/INTENSITY = (?P<continuum>[0-9]\.[0-9]{5}D[\+-][0-9]{1,2})"
    continuum_spectrum = []
    for each in re.finditer(pattern, stdout):
        continuum_spectrum.append(
            [
                float(each.groupdict()["wavelength"]),
                float(each.groupdict()["continuum"].replace("D", "E")),
            ]
        )
    continuum_spectrum = np.array(continuum_spectrum)
    return dict(
        continuum_lambda_air=continuum_spectrum.T[0],
        continuum_flux=continuum_spectrum.T[1],
    )


def parse_summary_synth_output(summary_out_path):
    """
    Parse the summary output from a MOOG synth operation.

    :param summary_out:
        The path of the summary output file produced by MOOG.
    """

    with open(summary_out_path, "r") as fp:
        summary = fp.readlines()

    # There could be multiple spectra in this output file, each separated by
    # headers. Scan to find the headers, then extract the information.
    partition_indices = [
        i
        for i, line in enumerate(summary)
        if line.lower().startswith("all abundances not listed below differ")
    ] + [None]

    spectra = []
    for i, start in enumerate(partition_indices[:-1]):
        end = partition_indices[i + 1]
        spectra.append(parse_single_spectrum(summary[start:end]))

    return spectra


def write_marcs_photosphere_for_moog(photosphere, path):
    return write_photosphere_for_moog(photosphere, path, format="WEBMARCS")


def write_kurucz_photosphere_for_moog(photosphere, path):
    return write_photosphere_for_moog(photosphere, path, format="KURUCZ")


registry.register_writer("moog", Photosphere, write_photosphere_for_moog, force=True)
registry.register_writer(
    "moog.marcs", Photosphere, write_marcs_photosphere_for_moog, force=True
)
registry.register_writer(
    "moog.kurucz", Photosphere, write_kurucz_photosphere_for_moog, force=True
)
