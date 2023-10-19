
import numpy as np
import os
from glob import glob
from astra.utils import expand_path
from astra.pipelines.ferre.utils import parse_header_path

# This is a DERIVATIVE product of ABUNDANCE_CONTROLS, but I put it here because it doesn't change
# often, and because it's hard to decipher all this TTIES crap. If you change ABUNDANCE_CONTROLS,
# you should check your ABUNDANCE_RELATIVE_TO_H
ABUNDANCE_RELATIVE_TO_H = {
    'C': False,
    'C_1': False,
    'N': False,
    'O': False,
    'Na': True,
    'Mg': False,
    'Al': True,
    'Si': False,
    'P': True,
    'S': False,
    'K': True,
    'Ca': False,
    'Ti': False,
    'Ti_2': False,
    'V': True,
    'Cr': True,
    'Mn': True,
    'Fe': True,
    'Co': True,
    'Ni': True,
    'Cu': True,
    'Ge': True,
    'Rb': True,
    'Ce': True,
    'Nd': True,
    'Yb': True,
    'C_12_13': False,
    # These are duplicates from when we had testing results with different species names.
    'C13': False,
    }

ABUNDANCE_CONTROLS = {
    "Al": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Ca": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "Ce": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],  
    },
    #CI
    "C_1": {
        "INDV_LABEL": ("C",),
    },
    "C": {
        "INDV_LABEL": ("C",),
    },
    "C_12_13": {
        "INDV_LABEL": ("C", ),
    }, 
    "CN": {
        "INDV_LABEL": (
            "C",
            "O Mg Si S Ca Ti",
        ),
    },
    "Co": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Cr": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Cu": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Fe": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Ge": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "K": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Mg": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "Mn": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Na": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Nd": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Ni": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "N": {
        "INDV_LABEL": ("N",),
    },
    "O": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "P": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Rb": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Si": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "S": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "Ti_2": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "Ti": {
        "INDV_LABEL": ("O Mg Si S Ca Ti",),
    },
    "V": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
    "Yb": {
        "INDV_LABEL": ("METALS",),
        "TIES": [("C", 0, -1), ("N", 0, -1), ("O Mg Si S Ca Ti", 0, -1)],
    },
}

def sanitise_parent_dir(parent_dir):
    return parent_dir.rstrip("/") + "/"

def get_input_nml_paths(parent_dir, stage):
    return glob(os.path.join(expand_path(parent_dir), stage, "*", "input.nml"))


def get_species_label_references():
    species_label_reference = {}
    for species, controls in ABUNDANCE_CONTROLS.items():
        if species == "CN": 
            continue

        label, = controls["INDV_LABEL"]
        # From https://www.sdss4.org/dr17/irspec/parameters/
        # metals is [M/H], alpha is [alpha/M], C is [C/M], N is [N/M]

        # TODO: allow for more complex ties?
        has_ties = controls.get("TIES", None) is not None

        if label in ("O Mg Si S Ca Ti", "C", "N"):
            is_x_m = True
        elif label in ("METALS", ):
            is_x_m = not has_ties
        else:
            raise ValueError("AAAAAHHHHH")

        species_label_reference[species] = (label, is_x_m)
    return species_label_reference



def group_header_paths(header_paths):
    """
    Sort a list of header paths into groups based on the telescope / LSF model adopted.
    This groups data together that can be finished without waiting on other observations.

    :param header_paths:
        A list of FERRE header paths.
    """

    all_lsfs = []
    grouped_header_paths = {}
    for header_path in header_paths:
        observatory, lsf, spectral_desc = task_id_parts(header_path)

        grouped_header_paths.setdefault(observatory, {})
        grouped_header_paths[observatory].setdefault(lsf, [])
        grouped_header_paths[observatory][lsf].append((spectral_desc, header_path))
        if lsf is not None:
            all_lsfs.append(lsf)
    all_lsfs = list(set(all_lsfs))

    return (grouped_header_paths, all_lsfs)


def task_id_parts(header_path):
    (
        *_,
        radiative_transfer_code,
        model_photospheres,
        isotopes,
        suffix,
        basename,
    ) = header_path.split("/")

    parts = basename.split("_")
    _ = 4
    gd, spectral_type = (parts[1][_], parts[1][_ + 1 :])

    if gd == "B" and spectral_type == "A":
        # Different string format for BA, since it is not telescope-dependent or fibre-dependent LSF.
        lsf_telescope_model, lsf = (None, None)
        desc = "BA"
    else:
        lsf_telescope_model = "LCO" if parts[3][4:] == "s" else "APO"
        gd = dict(g="giant", d="dwarf")[gd]
        lsf = parts[3][3]
        desc = f"{spectral_type}-{gd}"
    return [lsf_telescope_model, lsf, desc]


def yield_suitable_grids(
    all_headers,
    mean_fiber,
    teff,
    logg,
    m_h,
    telescope, 
    strict=True,
    **kwargs
):
    """
    Yield suitable FERRE grids given header information from an observation and a dictionary of grid limits.

    :param grid_info:
        A dictionary containing header paths as keys, and a three-length tuple as values: (1) metadata, (2) lower limits, (3) upper limits.
        This is the expected output from `parse_grid_information`.

    :param mean_fiber:
        The mean fiber number of observations.

    :param teff:
        An initial guess of the effective temperature.

    :param logg:
        An initial guess of the surface gravity.

    :param m_h:
        An initial guess of the metallicity.

    :returns:
        A generator that yields two-length tuples containing header path, and metadata..
    """

    # Figure out which grids are suitable.
    lsf_grid = get_lsf_grid_name(int(np.round(mean_fiber)))

    #point = np.array([m_h, logg, teff])
    point = np.array([logg, teff])
    P = point.size

    for header_path, headers in all_headers.items():

        meta = parse_header_path(header_path)
        lower_limits, upper_limits = (headers["LLIMITS"], headers["ULIMITS"])
        # print(meta["lsf"], lsf_grid, telescope, meta["lsf_telescope_model"], header_path)
        # Match star to LSF fiber number model (a, b, c, d) and telescope model (apo25m/lco25m).
        # TODO: This is a very APOGEE-specific thing and perhaps should be moved elsewhere.
        # Special case to deal with Kurucz photospheres.


        #if (
        #    meta["lsf"] != lsf_grid and not meta["lsf"].startswith("combo")
        #) or telescope != meta["lsf_telescope_model"]:
        #    continue

        if (
            # If it's the BA combo grid, don't worry about matching fiber and telescope LSF
            meta["lsf"].startswith("combo") \
            or (
                (meta["lsf"] == lsf_grid)
            and (
                (telescope == meta["lsf_telescope_model"])
                |   (
                        (telescope == "apo1m") 
                    &   (meta["lsf_telescope_model"] == "apo25m")
                    )
                )
            )
        ):                
            # We will take the RV parameters as the initial parameters.
            # Check to see if they are within bounds of the grid.
            if strict:
                if np.all(point >= lower_limits[-P:]) and np.all(point <= upper_limits[-P:]):
                    yield (header_path, meta, headers)
            else:
                # Only require temperature to be within the limits.
                if (upper_limits[-1] >= teff >= lower_limits[-1]):
                    yield (header_path, meta, headers)


def get_lsf_grid_name(fibre_number):
    """
    Return the appropriate LSF name (a, b, c, or d) to use, given a mean fiber number.

    :param fiber_number:
        The mean fiber number of observations.

    :returns:
        A one-length string describing which LSF grid to use ('a', 'b', 'c', or 'd').
    """
    if 50 >= fibre_number >= 1:
        return "d"
    if 145 >= fibre_number > 50:
        return "c"
    if 245 >= fibre_number > 145:
        return "b"
    if 300 >= fibre_number > 245:
        return "a"




def approximate_log10_microturbulence(log_g):
    """
    Approximate the log10(microturbulent velocity) given the surface gravity.

    :param log_g:
        The surface gravity.

    :returns:
        The log base-10 of the microturbulent velocity, vt: log_10(vt).
    """

    coeffs = np.array([0.372160, -0.090531, -0.000802, 0.001263, -0.027321])
    # I checked with Holtz on this microturbulence relation because last term is not used.
    DM = np.array([1, log_g, log_g**2, log_g**3, 0])
    return DM @ coeffs


def get_abundance_keywords(element, header_label_names):
    """
    Return a dictionary of task parameters given a chemical element. These are adopted from DR16.

    :param element:
        The chemical element to measure.

    :param header_label_names:
        The list of label names in the FERRE header file.
    """

    # These can be inferred from running the following command on the SAS:
    # cd /uufs/chpc.utah.edu/common/home/sdss50/dr16/apogee/spectro/aspcap/r12/l33/apo25m/cal_all_apo25m007/ferre
    # egrep 'INDV|TIE|FILTERFILE' */input.nml

    def get_header_index(label):
        # FERRE uses 1-indexing and Python uses 0-indexing.
        return 1 + header_label_names.index(label)

    try:
        c = ABUNDANCE_CONTROLS[element]
    except:
        raise ValueError(
            f"no abundance controls known for element '{element}' (available: {tuple(ABUNDANCE_CONTROLS.keys())}"
        )

    try:
        indv = [get_header_index(label) for label in c["INDV_LABEL"]]
    except ValueError:
        # Cannot tie to a dimension that does not exist.
        ferre_kwds = {}

    else:
        ties = c.get("TIES", [])

        ntie = 0
        ferre_kwds = {
            # We don't pass INDV here because this will be determined from the
            # 'frozen_<param>' arguments to the FerreGivenSDSSApStarFile tasks
            # "INDV": [get_header_index(label) for label in c["INDV_LABEL"]],
            "TYPETIE": 1,
            # We set NTIE below based on how many we could actually tie, since
            # the BA grid does not have as many dimensions.
            # "NTIE": len(ties),
        }
        for i, (tie_label, ttie0, ttie) in enumerate(ties, start=1):
            try:
                index = get_header_index(tie_label)
            except ValueError:
                # dimension not in this grid, so nothing to tie
                continue
            else:
                ferre_kwds.update(
                    {
                        f"INDTIE({i:.0f})": get_header_index(tie_label),
                        f"TTIE0({i:.0f})": ttie0,
                        # TODO: What if we don't want to tie it back to first INDV element?
                        f"TTIE({i:.0f},{indv[0]:.0f})": ttie,
                    }
                )
                ntie += 1

        ferre_kwds["NTIE"] = ntie

    # Freeze all other labels.
    frozen_parameters = {
        hln: (hln not in c["INDV_LABEL"]) for hln in header_label_names
    }

    return (frozen_parameters, ferre_kwds)
