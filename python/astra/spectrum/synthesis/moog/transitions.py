import numpy as np
from astropy.io import registry
from collections import OrderedDict
from astropy import units as u

from grok.transitions import Transition, Transitions
from grok.utils import periodic_table


def read_moog(path):
    """
    Read a MOOG-formatted list of transitions.

    :param path:
        The path of the line list.
    """

    with open(path, "r") as fp:
        lines = fp.readlines()

    transitions = []

    for i, line in enumerate(lines, start=1):
        s = line.split()
        if len(s) < 1:
            continue

        lambda_air, species, E_lower, log_gf = s[:4]

        # MOOG expects Angstroms and eV.
        lambda_air *= u.Angstrom
        E_lower *= u.eV

        vdW, E_dissoc, comment = (None, None, "")
        if len(s) > 4:
            try:
                vdW, *_ = line[40:60].strip().split()
                vdW = float(vdW)
                # TODO: parse EW / E_dissoc
            except:
                None
            comment = line[50:].strip()

        transitions.append(
            Transition(
                lambda_air=lambda_air,
                species=species,
                E_lower=E_lower,
                log_gf=log_gf,
                vdW=vdW,
                E_dissoc=E_dissoc,
                comment=comment,
            )
        )

    return Transitions(lines)


_moog_amu = [
    1.008,
    4.003,
    6.941,
    9.012,
    10.81,
    12.01,
    14.01,
    16.00,
    19.00,
    20.18,
    22.99,
    24.31,
    26.98,
    28.08,
    30.97,
    32.06,
    35.45,
    39.95,
    39.10,
    40.08,
    44.96,
    47.90,
    50.94,
    52.00,
    54.94,
    55.85,
    58.93,
    58.71,
    63.55,
    65.37,
    69.72,
    72.59,
    74.92,
    78.96,
    79.90,
    83.80,
    85.47,
    87.62,
    88.91,
    91.22,
    92.91,
    95.94,
    98.91,
    101.1,
    102.9,
    106.4,
    107.9,
    112.4,
    114.8,
    118.7,
    121.8,
    127.6,
    126.9,
    131.3,
    132.9,
    137.3,
    138.9,
    140.1,
    140.9,
    144.2,
    145.0,
    150.4,
    152.0,
    157.3,
    158.9,
    162.5,
    164.9,
    167.3,
    168.9,
    173.0,
    175.0,
    178.5,
    181.0,
    183.9,
    186.2,
    190.2,
    192.2,
    195.1,
    197.0,
    200.6,
    204.4,
    207.2,
    209.0,
    210.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.0,
    231.0,
    238.0,
    237.0,
    244.0,
    243.0,
]


def write_moog(transitions, path, include_header=True):
    """
    Write a list of atomic and molecular transitions to disk in a format that
    is friendly to MOOG.
    
    :param transitions:
        The atomic and molecular transitions to write.
    
    :param path:
        The path to store the transitions.
    """

    fmt = "{:10.3f}{: >10}{:10.3f}{:10.3f}{}{}{}{}"
    space = " " * 10
    with open(path, "w") as f:
        if include_header:
            f.write("\n")

        for line in sorted(transitions, key=lambda l: l.lambda_air):
            C6 = "{:10.3f}".format(line.vdW) if np.isfinite(line.vdW) else space
            D0 = (
                f"{line.E_dissociation.to('eV').value:10.3f}"
                if line.E_dissociation is not None
                else space
            )

            # Sort the molecules the right way.

            species = "".join([f"{Z:0>2.0f}" for Z in sorted(line.species.Zs) if Z > 0])
            if len(species) == 2:
                species = species.lstrip("0")
            species += f".{line.species.charge:.0f}"

            if sum(line.species.isotopes) > 0:

                for Z, isotope in zip(line.species.Zs, line.species.isotopes):
                    if Z == 0:
                        continue
                    if isotope == 0:
                        # If we don't have isotopes for one thing, MOOG will die.
                        # So here we give it the default mass from MOOG.
                        isotope = int(np.round(_moog_amu[Z - 1]))
                    species += f"{isotope:0>2.0f}"

                    if isotope > 100:
                        raise ValueError(
                            f"From the MOOG documentation: MOOG cannot handle a molecular isotope "
                            f"in which one of the atomic constitutents has a mass greater than two digits! "
                            f"The line {line} breaks this fundamental limit"
                        )
            # left-pad and strip
            species = species.lstrip("0")
            species = f"{species: >10}"

            f.write(
                fmt.format(
                    line.lambda_air.to("Angstrom").value,
                    species,
                    line.E_lower.to("eV").value,
                    line.log_gf,
                    C6,
                    D0,
                    space,
                    line.comment,
                )
                + "\n"
            )

    return None


registry.register_reader("moog", Transitions, read_moog, force=True)
registry.register_writer("moog", Transitions, write_moog, force=True)
