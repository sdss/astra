import os
import numpy as np
import subprocess
from collections import OrderedDict
from pkg_resources import resource_stream
from time import time

from astropy import units as u
from grok.transitions import Transitions
from grok.synthesis.utils import get_default_lambdas
from grok.utils import copy_or_write

from grok.transitions.utils import air_to_vacuum, vacuum_to_air
from grok.solar import periodic_table

is_hydrogen_line = lambda t: t.species.atoms == ("H",)

from grok import DO_NOT_SCALE_ABUNDANCES

def turbospectrum_bsyn(
    photosphere,
    transitions,
    lambdas=None,
    solar_abundances=None,
    isotopes=None,
    opacities=None,
    verbose=False,
    dir=None,
    input_transitions_format=None,
    hydrogen_lines=True,
    skip_irrelevant_transitions=True,
    update_missing_data=True,
    **kwargs,
):
    """
    Synthesize a stellar spectrum using Turbospectrum's `bsyn` routine.

    :param photosphere:
        The model photosphere.

    :param transitions:
        The atomic and molecular line transitions.

    :param lambdas: [optional]
        A three length tuple containing the start wavelength, end wavelength, and the
        wavelength step size. If `None` is given then this will default to nearly the
        range of the transition list, with a step size of 0.01 Angstroms.

    # TODO: Docs for abundances, isotopes.

    :param opacities: [optional]
        A path where the calculated opacities are stored. If None is given, then
        these will be calculated by Turbospectrum.

    :param verbose: [optional]
        Provide verbose outputs.

    :param dir: [optional]
        The directory to execute Turbospectrum from.
    """

    # These are vacuum wavelengths, but Turbospectrum works in air
    lambda_vacuum_min, lambda_vacuum_max, lambda_delta = lambdas
    lambda_min = vacuum_to_air(lambda_vacuum_min * u.Angstrom).value
    lambda_max = vacuum_to_air(lambda_vacuum_max * u.Angstrom).value

    _path = lambda basename: os.path.join(dir or "", basename)

    transition_basename_format = "transitions_{i}"
    modelinput_basename = "model"
    modelopac_basename = "opac"
    result_basename = "result"

    if opacities is not None:
        copy_or_write(opacities, _path(modelopac_basename))

    # Turbospectrum allows for transitions to be written into multiple different files.
    T = 1
    if isinstance(transitions[0], str):
        transition_paths_formatted = []
        for T, path in enumerate(transitions, start=1):
            if input_transitions_format == "turbospectrum":
                copy_or_write(
                    path, _path(transition_basename_format.format(i=T)),
                )
            else:
                _transitions = Transitions.read(path, format=input_transitions_format)
                _transitions = Transitions(
                    [t for t in _transitions if t.species.atoms != ("H",)]
                )
                _transitions.write(
                    _path(transition_basename_format.format(i=T)),
                    format="turbospectrum",
                    skip_irrelevant_transitions=skip_irrelevant_transitions,
                    update_missing_data=update_missing_data,
                )

            transition_paths_formatted.append(transition_basename_format.format(i=T))
    else:
        copy_or_write(
            transitions, _path(transition_basename_format.format(i=0)),
        )

        transition_paths_formatted = [
            transition_basename_format.format(i=i) for i in range(T)
        ]

    if hydrogen_lines:
        transition_paths_formatted.append("DATA/Hlinedata")
        T += 1

    # Write photosphere.
    copy_or_write(
        photosphere,
        _path(modelinput_basename),
        format=kwargs.get("photosphere_format", "turbospectrum"),
    )
        

    kwds = dict(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_delta=lambda_delta,
        marcs_file_flag=".true."
        if photosphere.meta["read_format"] == "marcs"
        else ".false.",
        metallicity=photosphere.meta["m_h"],
        alpha_fe=0,  # TODO: parse from abundances
        helium_abundance=0,  # TODO: parse from abundances
        r_process_abundance=0,  # TODO: parse from abundances
        s_process_abundance=0,  # TODO: parse from abundances
        modelinput_basename=modelinput_basename,
        modelopac_basename=modelopac_basename,
        result_basename=result_basename,
        microturbulence=photosphere.meta["microturbulence"],
        N_transition_paths=len(transition_paths_formatted),
        transition_paths_formatted="\n".join(transition_paths_formatted),
        spherical_flag="T" if photosphere.is_spherical_geometry else "F",
    )
    if solar_abundances is not None:
        kwds["num_individual_abundances"] = len(solar_abundances)
        individual_abundances_formatted = "\n" 
        for element, abundance in solar_abundances.items():
            atomic_number = 1 + periodic_table.index(element)
            if element in DO_NOT_SCALE_ABUNDANCES:
                value = abundance
            else:
                value = abundance + photosphere.meta['m_h']
            individual_abundances_formatted += f"  {atomic_number:.0f} {value:.3f}\n"
        kwds["individual_abundances_formatted"] = individual_abundances_formatted

    if isotopes is not None:
        raise NotImplementedError

    # Symbolically link the turbospectrum data folder.
    # TODO: UGH, this is a pain in the ass.
    os.symlink(
        "/uufs/chpc.utah.edu/common/home/u6020307/Turbospectrum2019/DATA", _path("DATA")
    )

    if opacities is None:
        # Execute Turbospectrum's babsma_lu

        # Write the babsma control file.
        with resource_stream(__name__, "babsma.template") as fp:
            babsma_template = fp.read()
            if isinstance(babsma_template, bytes):
                babsma_template = babsma_template.decode("utf-8")

        babsma_contents = babsma_template.format(**kwds)
        input_path = _path("babsma.par")
        with open(input_path, "w") as fp:
            fp.write(babsma_contents)

        # Execute Turbospectrum's babsma
        t_init = time()
        # $> babsma_lu < babsma.par 
        process = subprocess.run(
            ["babsma_lu"],
            cwd=dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=babsma_contents,
            encoding="ascii",
        )
        t_babsma_lu = time() - t_init
        if process.returncode != 0:
            raise RuntimeError(process.stderr)

    # Write the bsyn_lu control file.
    with resource_stream(__name__, "bsyn.template") as fp:
        bsyn_template = fp.read()
        if isinstance(bsyn_template, bytes):
            bsyn_template = bsyn_template.decode("utf-8")

    contents = bsyn_template.format(**kwds)
    input_path = _path("bsyn.par")
    with open(input_path, "w") as fp:
        fp.write(contents)

    # Execute Turbospectrum's bsyn_lu
    t_init = time()
    process = subprocess.run(
        ["bsyn_lu"],
        cwd=dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=contents,
        encoding="ascii",
    )
    t_bsyn_lu = time() - t_init
    if process.returncode != 0:
        raise RuntimeError(f"In {dir}: {process.stderr}")

    # Load the spectrum from the result file.
    data = np.loadtxt(_path(result_basename))

    wavelength, rectified_flux, flux = data.T
    # convert to vacuum wavelengths
    result = OrderedDict(
        [
            ("wavelength", air_to_vacuum(wavelength * u.Angstrom).value),
            # ("wavelength_unit", "Angstrom"),
            ("rectified_flux", rectified_flux),
            ("flux", flux),
            # ("continuum_wavelength", air_to_vacuum(wavelength * u.Angstrom).value),
            ("continuum", flux / rectified_flux),
            # ("flux_unit", "1e-17 erg / (Angstrom cm2 s)"),
        ]
    )

    meta = dict(
        dir=dir,
        # TODO: lots of stuff..
    )
    timing = dict(
        t_synthesis=t_bsyn_lu + t_babsma_lu,
        t_bsyn_lu=t_bsyn_lu,
        t_babsma_lu=t_babsma_lu,
    )
    return (result, timing, meta)
