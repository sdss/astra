import numpy as np
import os
from collections import Counter

def get_suffix(input_nml_path):
    try:
        _, suffix = input_nml_path.split(".nml.")
    except ValueError:
        return 0
    else:
        return int(suffix)

def get_new_path(existing_path, new_suffix):
    if new_suffix == 1:
        return f"{existing_path}.{new_suffix}"
    else:
        return ".".join(existing_path.split(".")[:-1]) + f".{new_suffix}"


def re_process_partial_ferre(existing_input_nml_path, pwd=None, exclude_indices=None):

    if pwd is None:
        pwd = os.path.dirname(existing_input_nml_path)
    
    existing_suffix = get_suffix(existing_input_nml_path)
    new_suffix = existing_suffix + 1

    new_input_nml_path = get_new_path(existing_input_nml_path, new_suffix)

    with open(existing_input_nml_path, "r") as f:
        lines = f.readlines()
    
    keys = ("PFILE", "OFFILE", "ERFILE", "OPFILE", "FFILE", "SFFILE")
    paths = {}
    for i, line in enumerate(lines):
        key = line.split("=")[0].strip()
        if key in keys:
            existing_relative_path = line.split("=")[1].strip("' \n")
            new_relative_path = get_new_path(existing_relative_path, new_suffix)
            lines[i] = line[:line.index("=")] + f"= '{new_relative_path}'"
            paths[key] = (existing_relative_path, new_relative_path)
    
    with open(new_input_nml_path, "w") as fp:
        fp.write("\n".join(lines))
    
    # Find the things that are already written in all three output files.
    output_path_keys = ["OFFILE", "OPFILE"]
    if os.path.exists(os.path.join(pwd, paths["SFFILE"][0])):
        output_path_keys.append("SFFILE")

    counts = []
    for key in output_path_keys:
        names = set(np.loadtxt(os.path.join(pwd, paths[key][0]), usecols=(0, ), dtype=str))
        counts.extend(names)

    completed_names = [k for k, v in Counter(counts).items() if v == len(output_path_keys)]
    input_names = np.loadtxt(os.path.join(pwd, paths["PFILE"][0]), usecols=(0, ), dtype=str)

    ignore_names = [] + completed_names
    if exclude_indices is not None:
        ignore_names.extend([input_names[int(idx)] for idx in exclude_indices])

    mask = [(name not in ignore_names) for name in names]
    if not any(mask):
        return (None, None)

    # Create new input files that ignore specific names.
    for key in ("PFILE", "ERFILE", "FFILE"):
        existing_path, new_path = paths[key]
        with open(os.path.join(pwd, existing_path), "r") as f:
            lines = f.readlines()
        
        with open(os.path.join(pwd, new_path), "w") as f:
            for line, m in zip(lines, mask):
                if m:
                    f.write(line)

    # Clean up the output files to only include things that are written in all three files.
    for key in output_path_keys:
        existing_path, new_path = paths[key]
        with open(os.path.join(pwd, existing_path), "r") as f:
            lines = f.readlines()
        
        lines = [line for line in lines if line.split()[0].strip() in completed_names]
        with open(os.path.join(pwd, existing_path) + ".cleaned", "w") as fp:
            fp.write("\n".join(lines))
        
    ignore_names = list(ignore_names)
    return (new_input_nml_path, ignore_names)
