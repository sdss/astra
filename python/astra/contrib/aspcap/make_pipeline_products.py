
from astra.contrib.aspcap.models import ASPCAPOutput, ASPCAPAbundances, ASPCAPStellarParameters
from astra.database.astradb import DataProduct
from astra.utils import expand_path

from sdss_access import SDSSPath

p = SDSSPath("sdss5")

import os
import subprocess
import numpy as np


from glob import glob
parent_folders = glob(expand_path("$MWM_ASTRA/0.3.0/v6_0_9-1.0/results/ferre/20??-??-??"))
from tqdm import tqdm

def get_ferre_flux(path, data_product_id, hdu):
    ps = subprocess.Popen(["awk", "{print $1}", path], stdout=subprocess.PIPE)
    output = subprocess.check_output(["grep", "-n", f"_{data_product_id}_{hdu}"], stdin=ps.stdout)
    ps.wait()
    line_number = int(output.decode("utf-8").split(":")[0])

    # Extract the model flux
    return np.loadtxt(
        path, 
        skiprows=line_number-1, 
        max_rows=1,
        usecols=range(1, 7515)
    )



for parent_folder in tqdm(parent_folders):

    element_fluxes = {}
    element_names = {}
    data_product_ids = []

    for grid_folder in tqdm(glob(f"{parent_folder}/params/*nlte*/")):

        output_parameter_names = np.atleast_1d(np.loadtxt(f"{grid_folder}/parameters.output", usecols=(0,), dtype=str))

        output_flux = np.atleast_2d(np.loadtxt(f"{grid_folder}/normalized_flux.output", usecols=range(1, 7515)))
        output_flux_names = np.atleast_1d(np.loadtxt(f"{grid_folder}/normalized_flux.output", usecols=(0,), dtype=str))

        assert np.all(output_parameter_names == output_flux_names)

        # Get the corresponding model fluxes from each element run.
        abundance_grid_folder = grid_folder.replace("/params/", "/abundances/")

        for element_folder in glob(f"{abundance_grid_folder}/*/"):
            element = element_folder.rstrip("/").split("/")[-1]
            element_names[element] = np.array(["_".join(ea.split("_")[2:]) for ea in np.atleast_1d(np.loadtxt(f"{element_folder}/normalized_flux.output", usecols=(0,), dtype=str))])
            element_fluxes[element] = np.atleast_2d(np.loadtxt(f"{element_folder}/normalized_flux.output", usecols=range(1, 7515), dtype=float))

        # Get the best flux per data product, HDU.
        model_flux_per_spec = {}
        for i, name in enumerate(output_flux_names):
            _, __, data_product_id, hdu = map(int, name.split("_"))

            key = f"{data_product_id}_{hdu}"

            model_flux_per_spec[key] = {
                "params": output_flux[i],
            }

            for element, names in element_names.items():
                try:
                    index, = np.where(names == key)[0]
                except:
                    model_flux_per_spec[key][element] = np.nan * np.ones(7514)
                else:                    
                    model_flux_per_spec[key][element] = element_fluxes[element][index]

    
        # Create data products.
        data_product_ids.extend([int(n.split("_")[2]) for n in output_parameter_names])

    data_product_ids = set(data_product_ids)

    q = (
        ASPCAPOutput
        .select()
        .where(ASPCAPOutput.data_product_id.in_(data_product_ids))            
    )

    raise a
    
        



'''
def prepare_pipeline_product(data_product_id):
    q = (
        ASPCAPOutput
        .select(
            ASPCAPOutput,
            ASPCAPStellarParameters.pwd,
        )        
        .join(ASPCAPStellarParameters, on=(
            (ASPCAPStellarParameters.data_product_id == ASPCAPOutput.data_product_id)
        &   (ASPCAPStellarParameters.hdu == ASPCAPOutput.hdu)
        ))
        #.where(ASPCAPOutput.data_product_id == data_product_id)
        .limit(1)
        .objects()
    )

    # For each pwd, need to get the model flux spectrum.

    for row in q:

        data_product = row.data_product
        prefix, suffix = row.pwd.split("0.3.0/v6_0_9-1.0/results/ferre/")
        date = suffix.split("/")[0]
        grid_name, = [ea for ea in row.pwd.split("/") if "nlte" in ea]

        parent_folder = expand_path(f"$MWM_ASTRA/0.3.0/v6_0_9-1.0/results/ferre/{date}")

        model_flux_path = f"{parent_folder}/params/{grid_name}/normalized_flux.output"

        ps = subprocess.Popen(["awk", "{print $1}", model_flux_path], stdout=subprocess.PIPE)
        output = subprocess.check_output(["grep", "-n", f"_{data_product.id}_{row.hdu}"], stdin=ps.stdout)
        ps.wait()
        line_number = int(output.decode("utf-8").split(":")[0])

        # Extract the model flux
        model_flux = np.loadtxt(
            model_flux_path, 
            skiprows=line_number-1, 
            max_rows=1,
            usecols=range(1, 7515)
        )

        # Do the same thing for all the elements

    '''
