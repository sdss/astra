import os
import luigi
import types
import numpy as np

from luigi.util import requires

from sdss_access import SDSSPath

from astra.tasks.base import BaseTask
from astra.tasks.io import AllStarFile

from astropy.io import fits

@requires(AllStarFile)



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
