
import astra
from astra.tasks.io import AllVisitSum
from astropy.table import Table
from collections import defaultdict
from tqdm import tqdm

# Load the DR16 AllVisit file to get the parameters we need to locate ApVisit files.
all_visits = Table.read(AllVisit(release="dr16", apred="r12").local_path)

# Keywords that are needed to train the NIR classifier.
common_kwds = dict(
    training_spectra_path="data/nir/nir_clean_spectra_shfl.npy",
    training_labels_path="data/nir/nir_clean_labels_shfl.npy",
    validation_spectra_path="data/nir/nir_clean_spectra_valid_shfl.npy",
    validation_labels_path="data/nir/nir_clean_labels_valid_shfl.npy",
    test_spectra_path="data/nir/nir_clean_spectra_test_shfl.npy",
    test_labels_path="data/nir/nir_clean_labels_test_shfl.npy",
    class_names=["FGKM", "HotStar", "SB2", "YSO"]
)

def get_kwds(row):
    """
    Return the ApVisit keys we need from a row in the AllVisitSum file.
    """
    return dict(
        telescope=row["TELESCOPE"],
        field=row["FIELD"].lstrip(),
        plate=row["PLATE"].lstrip(),
        mjd=str(row["MJD"]),
        prefix=row["FILE"][:2],
        fiber=str(row["FIBERID"])
    )


# We will run this in batch mode.
# (We could hard-code keys here but it's good coding practice to have a single place of truth)
kwds = { key: [] for key in get_kwds(defaultdict(lambda: "")).keys() }

# Add all visits to the keywords.
for row in tqdm(all_visits):
    for k, v in get_kwds(row).items():
        kwds[k].append(v)

raise a


if __name__ == "__main__":

    from astropy.table import Table

    class ClassifyAllApVisitSpectra(luigi.WrapperTask, BaseTask):

        release = luigi.Parameter()
        apred = luigi.Parameter()
        use_remote = luigi.BoolParameter(
            default=False, 
            significant=False,
            visibility=ParameterVisibility.HIDDEN
        )
        
        def requires(self):
            all_visit_sum = LocalTargetTask(
                path=SDSSPath(release=self.release).full("allVisitSum", apred=self.apred)
            )
            yield all_visit_sum

            # Load in all the apVisit files.
            table = Table.read(all_visit_sum.path)
            
            # Get the keywords we need.
            kwds = self.get_common_param_kwargs(ClassifySourceGivenApVisitFile)
            for key in ("apred", "telescope", "field", "plate", "mjd", "prefix", "fiber"):
                kwds[key] = []

            for i, row in enumerate(table):
                kwds["apred"].append(self.apred)
                kwds["telescope"].append(row["TELESCOPE"])
                kwds["field"].append(row["FIELD"].lstrip())
                kwds["plate"].append(row["PLATE"].lstrip())
                kwds["mjd"].append(str(row["MJD"]))
                kwds["prefix"].append(row["FILE"][:2])
                kwds["fiber"].append(str(row["FIBERID"]))
            
            yield ClassifySourceGivenApVisitFile(**kwds)
            
        
        def on_success(self):
            # Overwrite the inherited method that will mark this wrapper task as done and never re-run it.
            pass

        def output(self):
            return None

    task = ClassifyAllApVisitSpectra(
        release="dr16",
        apred="r12",
        use_remote=True
    )

    task.run()