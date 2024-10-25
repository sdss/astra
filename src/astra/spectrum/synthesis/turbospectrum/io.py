import gzip
import numpy as np
from textwrap import dedent
from astropy.io import registry

from grok.photospheres import Photosphere


def write_photosphere_for_turbospectrum(photosphere, path):
    """
    Write a photosphere in an ASCII format that Turbospectrum will recognise.

    :param photosphere:
        The photosphere, an instance of `photospheres.Photosphere`.
    
    :param path:
        The path to store the photosphere.    
    """

    photosphere.write(path, format="marcs")
    return None
    """
    # TODO: A hack because we are testing things without interpolation here.
    if photosphere.meta.get("read_format", None) == "marcs":
        print(f"WARNING: Copying MARCS file directly. I hope you're not interpolating atmospheres!")

        original_path = photosphere.meta["filename"]
        can_opener = gzip.open if original_path.lower().endswith(".gz") else open
        with can_opener(original_path, "r") as fp:
            content = fp.read()
        with open(path, "w") as fp:
            fp.write(content.decode("utf-8"))

        photosphere.meta["read_format"] = None
        photosphere.write("tmp2", format="turbospectrum")
        raise
        return None 
    """

    # TODO: Turbospectrum describes this format as 'KURUCZ', but it looks like an ATLAS-style format to me.
    # NOTE: Turbospectrum does not read the abundance information from the photosphere. It reads it from the control file.
    output = f"'KURUCZ' {photosphere.meta['n_depth']} {photosphere.meta.get('standard_wavelength', 5000):.0f} {photosphere.meta['logg']:.2f} 0 0.\n"

    # See https://github.com/bertrandplez/Turbospectrum2019/blob/master/source-v19.1/babsma.f#L727
    ross_keys = ("ABROSS", "KappaRoss")
    for ross_key in ross_keys:
        if ross_key in photosphere.dtype.names:
            break
    else:
        raise ValueError("Need ABROSS or KappaRoss")

    line_format = " {rho_x:.8e} {T: >8.1f} {P:.3e} {XNE:.3e} {ABROSS:.3e} {ACCRAD:.3e} {VTURB:.3e} {FLXCNV:.3e}\n"

    for i, line in enumerate(photosphere, start=1):
        output += line_format.format(
            i=i,
            rho_x=line["RHOX"],
            T=line["T"],
            P=line["P"],
            XNE=line["XNE"],
            ABROSS=line[ross_key],
            ACCRAD=line["ACCRAD"] if "ACCRAD" in photosphere.dtype.names else 0,
            VTURB=line["VTURB"] if "VTURB" in photosphere.dtype.names else 0,
            FLXCNV=line["FLXCNV"] if "FLXCNV" in photosphere.dtype.names else 0,
        )

    with open(path, "w") as fp:
        fp.write(output)
    return None


registry.register_writer(
    "turbospectrum", Photosphere, write_photosphere_for_turbospectrum, force=True
)
