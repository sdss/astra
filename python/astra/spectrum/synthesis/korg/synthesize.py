import os
import numpy as np
import subprocess
import re
from collections import OrderedDict
from pkg_resources import resource_stream
from time import time
from astropy import units as u

from grok.synthesis.utils import get_default_lambdas
from grok.transitions.utils import air_to_vacuum, vacuum_to_air
from grok.utils import copy_or_write


def synthesize(
    photosphere, transitions, lambdas=None, dir=None, solar_abundances=None, hydrogen_lines=True, **kwargs
):
    """
    Execute synthesis with Korg.
    """

    julia_script_basename = "grok.jl"
    photosphere_path_basename = "atmosphere"

    lambda_vacuum_min, lambda_vacuum_max, lambda_delta = lambdas

    _path = lambda basename: os.path.join(dir or "", basename)

    # Write the photosphere first.
    copy_or_write(
        photosphere,
        _path(photosphere_path_basename),
        format=kwargs.get("photosphere_format", "marcs"),
    )

    # lambda_vacuum_min = np.round(air_to_vacuum(lambda_air_min * u.Angstrom).to("Angstrom").value, 2) - 0.01
    # lambda_vacuum_max = np.round(air_to_vacuum(lambda_vacuum_max * u.Angstrom).to("Angstrom").value, 2) + 0.01
    
    kwds = dict(
        # Korg works in vacuum wavelengths.
        # TODO: Don't assume units for lambdas.
        lambda_vacuum_min=lambda_vacuum_min,
        lambda_vacuum_max=lambda_vacuum_max,
        lambda_vacuum_delta=lambda_delta,
        atmosphere_path=photosphere_path_basename,
        metallicity=photosphere.meta["m_h"],
        # I'm just giving a different metallicity for the initial run so that people don't
        # incorrectly think the result from the second run is actually being cached.
        fake_metallicity=photosphere.meta["m_h"] - 0.25,
        korg_read_transitions_format=kwargs.get("korg_read_transitions_format", "vald"),
        hydrogen_lines=str(hydrogen_lines).lower(),
        microturbulence=photosphere.meta["microturbulence"],
    )
    if solar_abundances is None:
        solar_abundances_formatted = 'Dict()'
        solar_relative = 'true'
    else:
        solar_relative = 'false'
        solar_abundances_formatted = 'Dict('
        f = []
        monh = photosphere.meta["m_h"]
        for element, v in solar_abundances.items():
            if element == "H":
                # Korg doesn't allow you to set this.
                continue
            elif element == "He":
                f.append(f'"{element}"=>{v}')
            else:
                f.append(f'"{element}"=>{v + monh}')
        solar_abundances_formatted += ', '.join(f)
        solar_abundances_formatted += ')'

    kwds.update(
        solar_relative=solar_relative,
        solar_abundances_formatted=solar_abundances_formatted,
    )

    # I wish I knew some Julia... eeek!
    if isinstance(transitions, (list, tuple)) and len(transitions) == 2:

        # Special case for TS 15000 - 15500
        if any(os.path.basename(path).startswith("turbospec.") for path in transitions):
            template_path = "template_turbospectrum.jl"
            assert transitions[-1].endswith(
                ".molec"
            ), "Put the molecule transition file last"

            for i, each in enumerate(transitions):
                basename = f"transitions_{i:.0f}"
                copy_or_write(
                    each,
                    _path(basename),
                    format=kwargs.get("transitions_format", "vald.stellar"),
                )
                kwds[f"linelist_path_{i}"] = basename
        else:
            template_path = "template_two_linelists.jl"

            for i, each in enumerate(transitions):
                basename = f"transitions_{i:.0f}"
                copy_or_write(
                    each,
                    _path(basename),
                    format=kwargs.get("transitions_format", "vald.stellar"),
                )
                kwds[f"linelist_path_{i}"] = basename

    else:
        if isinstance(transitions, (list, tuple)) and len(transitions) == 1:
            (transitions,) = transitions

        template_path = "template.jl"
        transitions_path_basename = "transitions"
        copy_or_write(
            transitions,
            _path(transitions_path_basename),
            format=kwargs.get("transitions_format", "vald.stellar"),
        )
        kwds["linelist_path"] = transitions_path_basename

    # Write the template.
    with resource_stream(__name__, template_path) as fp:
        template = fp.read()
        if isinstance(template, bytes):
            template = template.decode("utf-8")

    contents = template.format(**kwds)
    input_path = _path(julia_script_basename)
    with open(input_path, "w") as fp:
        fp.write(contents)

    # Execute it in a subprocess like we do for Turbospectrum and MOOG.
    # (I don't know how Julia works, but this could introduce some weird time penalty on Julia. Should check on that.)
    t_init = time()
    process = subprocess.run(
        ["julia", julia_script_basename],
        cwd=dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    t_subprocess = time() - t_init
    if process.returncode != 0:
        raise RuntimeError(process.stderr)

    # Write the stdout and stderr.
    with open(_path("stdout"), "w") as fp:
        fp.write(process.stdout)
    with open(_path("stderr"), "w") as fp:
        fp.write(process.stderr)

    # Parse for the synthesis time.
    compilation_stdout, runtime_stdout = process.stdout.split("Time 1 end")
    pattern = "Timing (?P<description>.+)\n\s*(?P<seconds>[\d\.]+) seconds"
    timing = dict(process=t_subprocess)
    for search in re.finditer(pattern, compilation_stdout):
        description, seconds = search.groups()
        description = description.replace(" ", "_")
        timing[f"t_{description}_compilation"] = float(seconds)

    for search in re.finditer(pattern, runtime_stdout):
        description, seconds = search.groups()
        description = description.replace(" ", "_")
        timing[f"t_{description}"] = float(seconds)

    wavelength, flux = np.loadtxt(_path("spectrum.out")).T

    continuum_wavelength, continuum = np.loadtxt(_path("continuum.out")).T

    result = OrderedDict(
        [
            ("wavelength", wavelength),
            # ("wavelength_unit", "Angstrom"),
            ("flux", flux),
            # ("flux_unit", "erg / (Angstrom cm2 s"),
            # ("continuum_wavelength", wavelength),
            ("continuum", continuum),
            ("rectified_flux", flux / continuum),
        ]
    )

    # Before:
    # - meta['wallclock_time'] was only showing synthesis time after compilation
    # Now:
    # - meta['timing']['t_synthesis'] is the time spent in synthesis

    meta = dict(
        dir=dir,
        stdout=process.stdout,
        stderr=process.stderr,
        lambda_vacuum_min=lambda_vacuum_min,
        lambda_vacuum_max=lambda_vacuum_max,
        lambda_vacuum_delta=lambda_delta,
    )
    # Parse the version used.
    match = re.search(
        "Korg v(?P<korg_version_major>\d+)\.(?P<korg_version_minor>\d+)\.(?P<korg_version_micro>\d+)",
        process.stdout,
    )
    meta.update(match.groupdict())

    return (result, timing, meta)
