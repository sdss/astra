import re
import numpy as np
from astropy.io import registry
from astropy import units as u
from typing import OrderedDict

from grok.transitions import (Species, Transition, Transitions)
from grok.transitions.vald import parse_levels
from grok.utils import safe_open

def read_ges(path):
    """
    Read transitions from the Gaia-ESO Survey that are stored on CDS at:

    https://cdsarc.u-strasbg.fr/viz-bin/cat/J/A+A/645/A106#/browse
    """

    path, content, content_stream = safe_open(path)
    lines = content.split("\n")

    header_rows = 10
    transitions = []
    for i, line in enumerate(lines[header_rows:]):
        if "|" not in line:
            continue

        element, ion, isotope, lambda_air, _ref_lambda_air, \
        log_gf, _e_log_gf, *_ref_log_gf, _gfflag, _synflag, lower_level_desc, \
        j_lower, E_lower, _ref_E_lower, upper_level_desc, j_upper, E_upper, _ref_E_upper, \
        rad_damp, _ref_rad_damp, stark_damp, _ref_stark_damp, vdw_damp, \
        _ref_vdw_damp = line.split("|")

        element = element.strip()
        charge = int(ion) - 1
        lambda_air = float(lambda_air)
        log_gf = float(log_gf)
        j_lower = float(j_lower)
        E_lower = float(E_lower) * u.eV
        j_upper = float(j_upper)
        E_upper = float(E_upper) * u.eV

        rad_damp = float(rad_damp) * (1/u.s)
        stark_damp = float(stark_damp) * (1/u.s)
        vdw_damp = float(vdw_damp)

        species = Species(f"{element} {charge}")
        # assumes atoms only!
        assert len([Z for Z in species.Zs if Z > 0]) == 1
        species.isotopes = (0, 0, int(isotope.strip()))
        lower_orbital_type, upper_orbital_type = parse_levels(lower_level_desc, upper_level_desc)

        transitions.append(
            Transition(
                species=species,
                lambda_vacuum=None,
                lambda_air=lambda_air * u.Angstrom,
                log_gf=log_gf,
                E_lower=E_lower,
                E_upper=E_upper,
                j_lower=j_lower,
                j_upper=j_upper,
                gamma_rad=rad_damp,
                gamma_stark=stark_damp,
                vdW=vdw_damp,
                lower_level_desc=lower_level_desc.strip(),
                upper_level_desc=upper_level_desc.strip(),
                lower_orbital_type=lower_orbital_type,
                upper_orbital_type=upper_orbital_type,
                comment=f"{species} {lower_level_desc} {upper_level_desc}"
            )
        )

    return Transitions(transitions)

registry.register_reader("cds.ges", Transitions, read_ges)