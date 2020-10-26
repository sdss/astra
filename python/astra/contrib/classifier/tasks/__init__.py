

from .train import (TrainNIRSpectrumClassifier, TrainOpticalSpectrumClassifier)
from .test import (ClassifyApVisitSpectrum, ClassifyAllApVisitSpectra)


if __name__ == "__main__":

    import luigi

    # ApVisit file
    #$APOGEE_REDUX/{apred}/{telescope}/{plate}/{mjd}/apVisit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits
    #To: $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Visit-{apred}-{plate}-{mjd}-{fiber:0>3}.fits
    apvisit_params = dict(
        use_remote=True, # Download remote paths if they don't exist.
        apred="r12",
        telescope="apo25m",
        field="000+02",
        plate="5815",
        mjd="56433",
        prefix="ap",
        fiber="221",
        release="dr16",
    )

    task = ClassifyApVisitSpectrum(
        **apvisit_params
    )
    task.run()

    
    #test_classifier()

