import re
import numpy as np
from astropy.io import registry
from astropy import units as u

from grok.transitions import (Species, Transition, Transitions)
from grok.utils import safe_open

_header_pattern = "'\s*(?P<turbospectrum_species_as_float>[\d\.]+)\s*'\s*(?P<ionisation>\d+)\s+(?P<num>\d+)"
_line_pattern = "\s*(?P<lambda_air>[\d\.]+)\s*(?P<E_lower>[\-\d\.]+)\s*(?P<log_gf>[\-\d\.]+)\s*(?P<vdW>[\-\d\.]+)\s*(?P<g_upper>[\d\.]+)\s*(?P<gamma_rad>[\d\.Ee\+\-]+)\s'(?P<lower_orbital_type>\w)' '(?P<upper_orbital_type>\w)'\s+(?P<equivalent_width>[\d\.]+)\s+(?P<equivalent_width_error>[\d\.]+)\s'(?P<comment>.+)'"
_line_pattern_short = "\s*(?P<lambda_air>[\d\.]+)\s*(?P<E_lower>[\-\d\.]+)\s*(?P<log_gf>[\-\d\.]+)\s*(?P<vdW>[\-\d\.]+)\s*(?P<g_upper>[\d\.]+)\s*(?P<gamma_rad>[\d\.Ee\+\-]+)"

# Amazingly, there does not seem to be a python string formatting that does the following:
_format_log_gf = lambda log_gf: "{0:< #6.3f}".format(log_gf)[:6]
# You might think that "{0:< #6.4g}" would work, but try it for -0.002 :head_exploding:
_line_template = "{line.lambda_air.value:10.3f} {line.E_lower.value:6.3f} {formatted_log_gf:s} {line.vdW_compact:8.3f} {line.g_upper:6.1f} {line.gamma_rad.value:9.2E} '{line.lower_orbital_type:s}' '{line.upper_orbital_type:s}' {line.equivalent_width:5.1f} {line.equivalent_width_error:6.1f} '{line.comment}'"


# Ionization potentials for each element. Used to identify bound-free transitions.
_ionization_potential_p1 = [
    13.60, 24.59,  5.39,  9.32,  8.30, 11.26, 14.53, 13.62, 17.40,
    21.56,  5.14,  7.64,  5.99,  8.15, 10.48, 10.36, 12.97, 15.76,
     4.34,  6.11,  6.54,  6.82,  6.74,  6.77,  7.44,  7.87,  7.86,
     7.64,  7.73,  9.39,  6.00,  7.88,  9.81,  9.75, 11.81, 14.00,
     4.18,  5.69,  6.38,  6.84,  6.88,  7.10,  7.28,  7.36,  7.46,
     8.33,  7.57,  8.99,  5.79,  7.34,  8.64,  9.01, 10.45, 12.13,
     3.89,  5.21,  5.58,  5.65,  5.42,  5.49,  5.55,  5.63,  5.68,
     6.16,  5.85,  5.93,  6.02,  6.10,  6.18,  6.25,  5.43,  7.00,
     7.88,  7.98,  7.87,  8.50,  9.10,  9.00,  9.22, 10.44,  6.11,
     7.42,  7.29,  8.42,  9.30, 10.75,  4.00,  5.28,  6.90,  6.08,
     9.99,  6.00
]
_ionization_potential_p2 = [
      .00, 54.42, 75.64, 18.21, 25.15, 24.38, 29.60, 35.12, 35.00,
    40.96, 47.29, 15.03, 18.83, 16.35, 19.72, 23.33, 23.81, 27.62,
    31.63, 11.87, 12.80, 13.58, 14.65, 16.50, 15.64, 16.18, 17.06,
    18.17, 20.29, 17.96, 20.51, 15.93, 18.63, 21.19, 21.60, 24.36,
    27.50, 11.03, 12.24, 13.13, 14.32, 16.15, 15.26, 16.76, 18.07,
    19.42, 21.49, 16.90, 18.87, 14.63, 16.53, 18.60, 19.13, 21.21,
    25.10, 10.00, 11.06, 10.85, 10.55, 10.73, 10.90, 11.07, 11.25,
    12.10, 11.52, 11.67, 11.80, 11.93, 12.05, 12.17, 13.90, 14.90,
    16.20, 17.70, 16.60, 17.50, 20.00, 18.56, 20.50, 18.76, 20.43,
    15.03, 16.68, 19.00, 20.00, 21.00, 22.00, 10.14, 12.10, 11.50,
    99.99, 12.00
]


def should_keep(transition, consider_reasons=(0, ), return_reason=False):
    """
    Returns a boolean flag whether a transition should be excluded (False) or included (True)
    from Turbospectrum.

    The logic here follows that from the VALD export script by Bertrand Plez.
    """
    reason = None

    if transition.species.charge not in (0, 1) and 0 in consider_reasons:
        reason = "Not a neutral or singly ionised species."
    elif transition.E_lower >= (15 * u.eV) and 1 in consider_reasons:
        reason = "Lower excitation potential exceeds 15 eV."
    elif (len(transition.species.atoms) == 1 and transition.species.atoms[0] in ("H", "He")) and 2 in consider_reasons:
        reason = "Skipping H and He atomic lines."
    elif not transition.is_molecule and transition.species.charge == 1 \
        and transition.E_upper is not None and transition.E_upper.to("eV").value > _ionization_potential_p1[transition.species.Zs[-1] - 1] and 3 in consider_reasons:
        reason = f"Neutral atomic species is a bound-free transition ({transition.E_upper.value} > {_ionization_potential_p1[transition.species.Zs[-1] - 1]})."
    elif not transition.is_molecule and transition.species.charge == 2 \
        and transition.E_upper is not None and transition.E_upper.to("eV").value > _ionization_potential_p2[transition.species.Zs[-1] - 1] and 4 in consider_reasons:
        reason = f"Singly ionized atomic species is a bound free transition ({transition.E_upper.value} > {_ionization_potential_p2[transition.species.Zs[-1] - 1]})."

    return reason if return_reason else reason is None


def update_missing_transition_data(transition):
    """
    Validate (and update) transition data. This is done by the script that converts VALD line lists
    to Turbospectrum formats. Specifically it checks the van der Waals broadening constant and
    gamma radiative. If it doesn't like the values, it changes them.

    This returns a copy of the transition, with the updated data.
    """
    t = transition.copy()
    if np.isclose(t.vdW, 0, atol=1e-10):
        t.vdW = lookup_approximate_vdW(t)

    t.gamma_rad = parse_gamma_rad(t)
    t.equivalent_width = t.equivalent_width or 0
    t.equivalent_width_error = t.equivalent_width_error or 1.0
    t.lower_orbital_type = t.lower_orbital_type or "X"
    t.upper_orbital_type = t.upper_orbital_type or "X"
    return t


def parse_gamma_rad(transition):
    #https://github.com/bertrandplez/Turbospectrum2019/blob/master/Utilities/vald3line-BPz-freeformat.f#L420-428
    if transition.gamma_rad.value > 3:
        return 10**transition.gamma_rad.value * (1/u.s)
    return 1e5 * (1/u.s)


def lookup_approximate_vdW(transition):

    default_value = 2.5 # Mackle et al. 1975 A&A 38, 239

    neutral_damping = {
        "Na": 2.0, # Holweger 1971 A&A 10, 128
        "Si": 1.3, # Holweger 1973 A&A 26, 275
        "Ca": 1.8, # O'Neill & Smith 1980 A&A 81, 100
        "Fe": 1.4  # Simmons & Blackwell 1982 A&A 112, 209 and Magain & Zhao 1996 A&A 305, 245
    }
    ionized_damping = {
        "Ca": 1.4, # from fit of H&K in the HM model to the fts intensity spectrum
        "Sr": 1.8, # from fit of Sr II 4077.724 in the HM model to the fts intensity spectrum
        "Ba": 3.0 # Holweger & Muller 1974 Solar Physics 39, 19
    }
    is_molecule = (len([Z for Z in transition.species.Zs if Z > 0]) > 1)
    if not is_molecule:
        if transition.species.charge == 0:
            try:
                return neutral_damping[transition.species.atoms[0]]
            except KeyError:
                return default_value
        elif transition.species.charge == 1:
            try:
                return ionized_damping[transition.species.atoms[0]]
            except KeyError:
                return default_value
        else:
            # We shouldn't even be bothering with these highly ionized species!
            return default_value
    else:
        return default_value


def read_transitions(path):
    """
    Read transitions formatted for Turbospectrum.

    :param path:
        The path where the transitions are written.
    """

    path, content, content_stream = safe_open(path)
    lines = content.split("\n")

    keys_as_floats = (
        "lambda_air", "E_lower", "log_gf",
        "vdW", "g_upper", "gamma_rad",
        "equivalent_width", "equivalent_width_error"
    )
    i, transitions = (0, [])
    while i < len(lines):
        if len(lines[i].strip()) == 0:
            i += 1
            continue

        common = re.match(_header_pattern, lines[i])
        if common is None:
            raise ValueError(f"Cannot match header pattern '{_header_pattern}' from '{lines[i]}'")

        common = common.groupdict()
        num = int(common.pop("num"))

        # Note that we are interpreting the ionisation state from the species representation,
        # and ignoring the 'ionisation' in 'common'.
        common["species"] = Species(common["turbospectrum_species_as_float"])
        # Update the charge, using the representation like "Th II", since Turbospectrum does
        # not encode the ionisation in this species representation.
        common["species"].charge = Species(lines[i + 1][1:-1]).charge

        for j, line in enumerate(lines[i+2:i+2+num], start=i+2):

            match = re.match(_line_pattern, line)
            if match is None:
                match = re.match(_line_pattern_short, line)
            transition = match.groupdict()

            row = { **common, **transition }

            # Format things.
            for key in keys_as_floats:
                row.setdefault(key, 0) # in case we only have things from the short format
                row[key] = float(row[key])

            # Add units.
            row["lambda_air"] *= u.Angstrom
            row["E_lower"] *= u.eV
            row["gamma_rad"] *= (1/u.s)

            row.setdefault("gamma_stark", 0)
            row["gamma_stark"] *= (1/u.s)

            # https://github.com/bertrandplez/Turbospectrum2019/blob/master/Utilities/vald3line-BPz-freeformat.f#448
            # g_upper = j_upper * 2 + 1
            row["j_upper"] = 0.5 * (row["g_upper"] - 1)
            transitions.append(Transition(
                lambda_vacuum=None,
                **row
            ))
        i += 2 + num

    return Transitions(transitions)


def write_transitions(
        transitions,
        path,
        skip_irrelevant_transitions=False,
        update_missing_data=False,
    ):
    """
    Write transitions to disk in a format that Turbospectrum accepts.

    :param transitions:
        The atomic and molecular transitions.

    :param path:
        The path to store the transitions on disk.

    :param skip_irrelevant_transitions: [optional]
        Skip over transitions that Bertrand Plez considers irrelevant, based on Plez's script for
        translating VALD-formatted line lists to Turbospectrum format (default: False).

    :param update_missing_data: [optional]
        Update the transitions with missing data values from a lookup table using the
        `grok.synthesis.turbospectrum.update_missing_transition_data` function (default: False).
    """

    if skip_irrelevant_transitions:
        transitions = Transitions(filter(should_keep, transitions))

    # Sort the right way.
    group_names = list(map(str, sorted(set([float(line.species.compact) for line in transitions]))))

    lines = []
    for i, group_name in enumerate(group_names):

        group = sorted(
            filter(lambda t: str(float(t.species.compact)) == group_name, transitions),
            key=lambda t: t.lambda_vacuum
        )

        # Add header information.
        species = group[0].species

        # Sort so we represent TiO as 822, etc.
        indices = [index for index in np.argsort(species.Zs) if species.Zs[index] > 0]

        #formula = "".join([f"{Z:0>2.0f}" for Z in species.Zs if Z > 0])
        #isotope = "".join([f"{I:0>3.0f}" for Z, I in zip(species.Zs, species.isotopes) if Z > 0])
        formula = "".join([f"{species.Zs[index]:0>2.0f}" for index in indices])
        isotope = "".join([f"{species.isotopes[index]:0>3.0f}" for index in indices])

        # Left-pad and strip formula.
        formula = formula.lstrip("0")
        formula = f"{formula: >4}"

        compact = f"{formula}.{isotope}"

        lines.extend([
            f"'{compact: <20}' {species.charge + 1: >4.0f} {len(group): >9.0f}",
            f"'{str(species):7s}'"
        ])

        for line in group:

            # Turbospectrum makes some approximations and lookups if the input values are not what
            # they want.
            updated_line = update_missing_transition_data(line) if update_missing_data else line
            try:
                formatted_line = _line_template.format(
                    line=updated_line,
                    formatted_log_gf=_format_log_gf(line.log_gf)
                )
            except TypeError:
                if not update_missing_data:
                    raise TypeError("Missing transition data. Update it or set `update_missing_data` to True.")
            lines.append(formatted_line)


    with open(path, "w") as fp:
        fp.write("\n".join(lines))

    return None



def identify_turbospectrum(origin, *args, **kwargs):
    """
    Identify a line list being in Turbospectrum format.

    There's no accepted nomenclature for line list filenames, so we will check
    the header of the file.
    """
    if args[1] is None:
        return False

    first_line = args[1].readline()
    if isinstance(first_line, bytes):
        first_line = first_line.decode("utf-8")

    # Reset pointer.
    args[1].seek(0)

    return (re.match(_header_pattern, first_line) is not None)

registry.register_reader("turbospectrum", Transitions, read_transitions, force=True)
registry.register_writer("turbospectrum", Transitions, write_transitions, force=True)
registry.register_identifier("turbospectrum", Transitions, identify_turbospectrum, force=True)
