import os
"""
import luigi
import types
import numpy as np

from luigi.util import requires

from sdss_access import SDSSPath

from astra.tasks.base import BaseTask
from astra.tasks.io import AllStarFile

from astropy.io import fits
"""

_profile_keywords = {
    "notchpeak": {
        "account": "sdss",
        "partition": "notchpeak",
        "ntasks": 16,
        "mem": "48G",
        # TODO: Ask Holtz why he has this constraint
        "constraint": "rom",
        "time": "48:00:00",
        "nodes": 1
    },
    "tacc": {
        "partition": "normal",
        "ntasks-per-node": 24
    },
    "np": {
        "account": "sdss-np",
        "partition": "np",
        "ntasks": 64,
        "time": "48:00:00",
        "nodes": 1
    },
    "kingspeak-fast": {
        "account": "sdss-kp-fast",
        "partition": "sdss-kp",
        "ntasks": 16,
        "nodes": 1
    },
    "kingspeak": {
        "account": "sdss-kp",
        "partition": "sdss-kp",
        "ntasks": 16,
        "nodes": 1
    }    
}


def prepare_slurm_file(
        commands,
        name,
        profile="kingspeak",
        cwd=None,
        query_host=None,
        query_port=None,       
        max_run=1,
        num_py_threads=None,
        **kwargs
    ):

    if name is None:
        name = commands[0].split()[0].strip()
    
    if cwd is not None:
        cwd = os.getcwd()
    
    try:
        kwds = _profile_keywords[profile]
    
    except KeyError:
        raise KeyError(f"unknown profile '{profile}'. Available: {','.join(list(_profile_keywords.keys()))}")

    kwds.update(kwargs)

    content = ["#!/bin/csh"]

    for key, value in kwds.items():
        content.append(f"#SBATCH --{key}={value}")
    
    # Error / output:
    content.extend([
        f"#SBATCH -o {name}.slurm.out",
        f"#SBATCH -e {name}.slurm.out"
    ])
    if num_py_threads is not None:
        keys = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS"
        )
        for key in keys:
            content.append(f"setenv {key} {num_py_threads}")

    content.append(f"cd {cwd}")
    content.extend(commands)

    content.extend([
        "",
        "wait",
        "echo FIN"
    ])

    return "\n".join(content)


if __name__ == "__main__":

    foo = prepare_slurm_file(["echo FOOBAR"], "echo_test", time="01:00:00")
    with open("test.slurm", "w") as fp:
        fp.write(foo)





"""
@requires(AllStarFile)
class GetAllStarResult(BaseTask):

    identifier = luigi.Parameter(
        description="The identifying string to find the correct row. This should be used "
                    "in conjunction with `identifier_column`. If you want to find 'VESTA'"
                    " by it's `APOGEE_ID` then you should set `identifier='VESTA'` and "
                    "`identifier_column='APOGEE_ID'`. However, you may get multiple sets "
                    "of results if you are not using `APSTAR_ID`, `TARGET_ID`, or `ASPCAP_ID`"
                    " as the identifier column. These are the only unique identifiers for "
                    " allStar files."
    )

    identifier_column = luigi.Parameter(
        default="APSTAR_ID",
        description="The column name to use to identify the ApStar."
    )

    hdu = luigi.IntParameter(
        default=1,
        description="HDU index to access in the allStar file",
        visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    def setup(self):
        self._allStar_path = AllStarFile(**self.get_common_param_kwargs(AllStarFile)).local_path
        self._allStar_image = fits.open(self._allStar_path)
        return self._allStar_image

    @property
    def data(self):
        return self._allStar_image[self.hdu]

    def run(self):

        image = self.setup()

        try:
            hdu = image[self.hdu]
        except:
            raise KeyError(f"Error loading HDU index {self.hdu} of {self._allStar_path}")

        try:
            mask = hdu.data[self.identifier_column] == self.identifier

        except KeyError:
            raise KeyError(f"Identifying column '{self.identifier_column}' not found "
                           f"in {self._allStar_path}. Available columns ")

        results = np.array(hdu.data[mask])
        self.teardown()

        # Don't output anything!
        return results


    def teardown(self):
        self._allStar_image.close()
        del self._allStar_image


if __name__ == "__main__":

    task = GetAllStarResult(
        release="dr16",
        aspcap="l33",
        apred="r12",
        identifier_column="APOGEE_ID",
        identifier="VESTA"
    )
    
    task.run()
"""