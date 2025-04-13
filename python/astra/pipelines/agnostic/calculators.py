from collections import defaultdict, deque
from itertools import combinations
from datetime import date
from typing import List
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm
from util import Lookup
from gridsearch import find_params, model


def calculate_visit_frequency(
    data: pd.DataFrame,
    fn: str,
    datapath: str = "/home/riley/uni/rproj/data/",
    write: bool = True,
):
    """
    Find the total visit frequency.

    :param data:
        Main DataFrame containing ID's and Variable data
    :param fn:
        Filename (i.e. APOGEENet-0.4.0) for save organizing
    :param datapath:
        Datapath to save to. Defaults to Riley workspace.
    :param write:
        Whether to write to a file
    """
    lenStack = deque()
    for name, group in tqdm(data.groupby(["GAIA_DR3_SOURCE_ID"])):
        lenStack.append(len(group))
    vf_data = pd.DataFrame(data={"LENGTH": lenStack})

    # save list frequency to file if requested
    if write:
        vf_data.to_parquet(f"{datapath}visit_frequency_{tdy}_{fn}.parquet",
                           engine="pyarrow")

    return vf_data


def calculate_pairwise_distance(
    data: pd.DataFrame,
    vars: List[str],
    fn: str,
    datapath: str = "/home/riley/uni/rproj/data/",
    write: bool = True,
    adjust: bool = False,
    gridsearch: bool = False,
):
    """
    Calculate pairwise distance scores across a set of given variables.
    Also calculates for additional systematic uncertainty adjustment

    :param data:
        Main DataFrame containing ID's and Variable data
    :param vars:
        List of variable names as strings.
    :param fn:
        Filename (i.e. APOGEENet-0.4.0) for save organizing
    :param datapath:
        Datapath to save to. Defaults to Riley workspace.
    :param write:
        Whether to write to file. Defaults to True.
    :param adjust:
        Whether to calculate systematic uncertainty adjustments. Defaults to False.
    :param gridsearch:
        Perform a gridsearch on each variable.
    """
    # make dictionary of empty stacks
    vardict = defaultdict(deque)
    paramdict = defaultdict(list)

    start = timer()
    for var in vars:
        for name, group in tqdm(data.groupby(["GAIA_DR3_SOURCE_ID"]),
                                desc=var):
            if len(group) > 1:
                # find all possible combinations
                indices = np.arange(0, len(group), 1)
                combos = combinations(indices, 2)

                # loop for every combination and variable
                for combo in combos:
                    combo = list(combo)
                    # get variable data
                    A, B = group[var].values[combo]
                    uAB = (group["E_" + var].values[combo[0]]**2 +
                           group["E_" + var].values[combo[1]]**2)
                    vardict[var].append((A - B) / np.sqrt(uAB))

                    # attach gridsearch values if requested
                    if gridsearch:
                        # obtain uA uB
                        uA, uB = group["E_" + var].values[combo]

                        # obtain relevant SNR values
                        SNR_A, SNR_B = group["SNR"].values[combo]
                        TEFF_A, TEFF_B = group["TEFF"].values[combo]

                        # if all non NaN, append
                        if (~np.isnan([
                                A, B, uAB, uA, uB, SNR_A, SNR_B, TEFF_A, TEFF_B
                        ])).all():
                            # TODO: find a way better way of implementing this
                            # than the defaultdict(list) method
                            paramdict["A"].append(A)
                            paramdict["B"].append(B)
                            paramdict["uA"].append(uA)
                            paramdict["uB"].append(uB)
                            paramdict["uAB"].append(uAB)
                            paramdict["SNR_A"].append(SNR_A)
                            paramdict["SNR_B"].append(SNR_B)
                            paramdict["TEFF_A"].append(TEFF_A)
                            paramdict["TEFF_B"].append(TEFF_B)

                    # do added sys uncertainties if requested
                    if adjust:
                        for adjustment in Lookup.base_adjustments:
                            if var == "TEFF":
                                additive = adjustment * (10**3)
                            else:
                                additive = adjustment
                            uAB = uAB + 2 * additive**2
                            vardict[var +
                                    Lookup.convert_extension(additive)].append(
                                        (A - B) / np.sqrt(uAB))

        # perform gridsearch if requested
        if gridsearch:
            # normalize TEFF array
            paramdict["TEFF_A"] = (np.array(paramdict["TEFF_A"]) -
                                   3000) / (9000 - 3000)
            paramdict["TEFF_B"] = (np.array(paramdict["TEFF_B"]) -
                                   3000) / (9000 - 3000)

            # get best param values
            best = find_params(paramdict, var)

            # add gridsearched version
            A = np.array(paramdict["A"])
            B = np.array(paramdict["B"])
            uAB = np.array(paramdict["uAB"])
            uA = np.array(paramdict["uA"])
            uB = np.array(paramdict["uB"])
            SNR_A = np.array(paramdict["SNR_A"])
            TEFF_A = np.array(paramdict["TEFF_A"])
            SNR_B = np.array(paramdict["SNR_B"])
            TEFF_B = np.array(paramdict["TEFF_B"])
            vardict[var +
                    "_GS"] = (A - B) / np.sqrt(uAB +
                                               model(best, SNR_A, TEFF_A)**2 +
                                               model(best, SNR_B, TEFF_B)**2)
            vardict[var + "_E_GS"] = np.concatenate((
                np.sqrt(uA**2 + model(best, SNR_A, TEFF_A)**2),
                np.sqrt(uB**2 + model(best, SNR_B, TEFF_B)**2),
            ))

            vardict[var + "_SNR"] = np.concatenate((SNR_A, SNR_B))
            vardict[var + "_TEFF"] = np.concatenate((TEFF_A, TEFF_B))
            vardict[var + "_t0"] = best[0]
            vardict[var + "_t1"] = best[1]
            vardict[var + "_t2"] = best[2]
            print(f"{var}: {np.std(vardict[var+'_GS'])} || {best}")

            # delete dictionary
            paramdict = defaultdict(list)

    print(f"Time for calculation: {timer() - start}")

    # perform gridsearch
    if gridsearch:
        for var in vars:
            continue

    data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vardict.items()]))

    # save if desired
    if write:
        # save to file
        tdy = date.today()
        z_fn = f"{datapath}z_{tdy}_{fn}.parquet"
        data.to_parquet(z_fn, engine="pyarrow")

    return data


def calculate_coadd_delta(
    data_visits: pd.DataFrame,
    data_stars: pd.DataFrame,
    fn: str,
    datapath: str = "/home/riley/uni/rproj/data/",
    write: bool = False,
):
    """
    Calculate pairwise distance scores across a set of given variables.
    Also calculates for additional systematic uncertainty adjustment.

    :param data_visits:
        Main DataFrame containing visit data
    :param data_stars:
        Main DataFrame containing coadd data
    :param fn:
        Filename (i.e. APOGEENet-0.4.0).
    :param datapath:
        Datapath to save to. Defaults to Riley workspace.
    :param write:
        Whether to write to file. Defaults to False.
    """

    # 1. combine by concatenate into multi-index
    data_combined = pd.concat([data_stars, data_visits],
                              keys=["coadd", "visit"])

    # 2. loop through via groupby object
    coadd_dict = defaultdict(list)
    for name, group in tqdm(data_combined.groupby(["GAIA_DR3_SOURCE_ID"]),
                            desc="Calculating Coadd Delta"):
        # perform loop for each var
        if (len(group) > 1) & ("coadd" in group.index):
            visit_subgroup = group.loc["visit"]
            for var in Lookup.vars:
                # get coadd score (there should be one only)
                coadd_var = group.loc["coadd"][var].values

                # loop for all visits
                var_array = visit_subgroup[var].values
                coadd_scores = np.abs(var_array - coadd_var)

                # add coadd scores to defaultdict
                for i in range(len(coadd_scores)):
                    coadd_dict[var].append(coadd_scores[i])

            # append to defaultdict with additional data for each visit
            # TODO: find better cleaner method of handling pointer issue
            for i in range(len(visit_subgroup)):
                coadd_dict["SNR"].append(visit_subgroup["SNR"].values[i])
                coadd_dict["TELESCOPE"].append(
                    visit_subgroup["TELESCOPE"].values[i])
    data = pd.DataFrame(data=coadd_dict)

    # save data
    if write:
        tdy = date.today()
        data.to_parquet(datapath + f"coadd_{tdy}_{fn}.parquet")

    return data


if __name__ == "__main__":
    # testing
    from cleaners import clean_ASPCAP

    print("Loading & cleaning data...")
    start = timer()
    data_visits = Table.read(
        "/home/riley/uni/rproj/data/allASPCAPVisit-0.4.0.fits")
    data_stars = Table.read(
        "/home/riley/uni/rproj/data/allASPCAPStar-0.4.0.fits")
    data_visits = data_visits.to_pandas()[Lookup.all_columns + ["TELESCOPE"]]
    data_stars = data_stars.to_pandas()[Lookup.all_columns + ["TELESCOPE"]]

    data_visits, data_stars = clean_ASPCAP(data_visits, data_stars=data_stars)
    print("Processing finished! Time taken =", timer() - start)

    print("Calculating z-score...")
    tdy = date.today()
    z_calc = calculate_pairwise_distance(
        data_visits,
        Lookup.vars,
        "allASPCAPVisit-0.4.0",
        write=True,
        adjust=True,
        gridsearch=True,
    )

    from plotters import plot_pairwise

    print("Performing zscore plots!")
    plot_pairwise(z_calc, adjust="grid")
