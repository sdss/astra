import numpy as np
import json
from astra import log
from astra.database.astradb import database, ZetaPayneOutput
from astra.base import TaskInstance, Parameter, TupleParameter
from astra.contrib.zetapayne.run_MWMPayne import fit_spectrum
from astra.contrib.zetapayne.Network import Network
from astra.contrib.zetapayne.FitLogger import FitLoggerDB
from astra.tools.spectrum import SpectrumList
from astra.tools.spectrum.utils import spectrum_overlaps
from astra.utils import expand_path, list_to_dict

from astra.sdss.datamodels.base import get_extname
from astra.sdss.datamodels.pipeline import create_pipeline_product


class ZetaPayne(TaskInstance):

    """
    Estimate stellar labels of hot stars using a single layer neural network
    and Chebyshev polynomials for modelling the pseudo-continuum.

    :param apogee_nn_path: [optional]
        The path to the pre-trained neural network for APOGEE spectra.
    
    :param apogee_wave_range: [optional]
        A two-length tuple containing the (lower, upper) wavelength range
        to use when fitting APOGEE spectra. Default is (15000, 17000).
    
    :param spectral_R: [optional]
        The spectral resolution to use when fitting APOGEE spectra. Default is 22,5000.

    :param apogee_N_chebyshev: [optional]
        The number of Chebyshev polynomials to use when fitting APOGEE spectra. Default is 15.
    
    :param apogee_N_presearch_iter: [optional]
        The number of iterations to use when pre-searching for the best APOGEE fit. Default is 1.
    
    :param apogee_N_presearch: [optional]
        The number of pre-searches to perform for APOGEE spectra. Default is 4000.
    
    :param boss_nn_path: [optional]
        The path to the pre-trained neural network for BOSS spectra.

    :param boss_wave_range: [optional]
        A two-length tuple containing the (lower, upper) wavelength range to use
        when fitting BOSS spectra. Default is (4500, 10170).
    
    :param boss_spectral_R: [optional]
        The spectral resolution to use when fitting BOSS spectra. Default is 2,000.
    
    :param boss_N_chebyshev: [optional]
        The number of Chebyshev polynomials to use when fitting BOSS spectra. Default is 15.
    
    :param boss_N_presearch_iter: [optional]
        The number of iterations to use when pre-searching for the best BOSS fit. Default is 1.

    :param boss_N_presearch: [optional]
        The number of pre-searches to perform for BOSS spectra. Default is 4000.
    """


    # APOGEE
    apogee_nn_path = Parameter(
        default="$MWM_ASTRA/component_data/ZetaPayne/OV_5K_n300.npz",
        bundled=True
    )
    apogee_wave_range = TupleParameter(default=(15_000, 17_000))
    apogee_spectral_R = Parameter(default=22_500)

    apogee_N_chebyshev = Parameter(default=15)
    apogee_N_presearch_iter = Parameter(default=1)
    apogee_N_presearch = Parameter(default=4000)

    # BOSS
    boss_nn_path = Parameter(
        default="$MWM_ASTRA/component_data/ZetaPayne/NN_OPTIC_n300_b1000_v0.1_27.npz",
        bundled=True
    )
    boss_wave_range = TupleParameter(default=(4_500, 10_170))
    boss_spectral_R = Parameter(default=2_000)

    boss_N_chebyshev = Parameter(default=15)
    boss_N_presearch_iter = Parameter(default=1)
    boss_N_presearch = Parameter(default=4000)

    def execute(self):

        # Load in the networks.
        log.info(f"Loading networks..")
        if self.apogee_nn_path is not None:
            apogee_nn = Network()
            apogee_nn.read_in(expand_path(self.apogee_nn_path))
        else:
            apogee_nn = None

        if self.boss_nn_path is not None:
            boss_nn = Network()
            boss_nn.read_in(expand_path(self.boss_nn_path))
        else:
            boss_nn = None

        log.info(f"Initializing logger..")
        log_dir = expand_path("$MWM_ASTRA/logs/") # temporary

        logger = FitLoggerDB(log_dir)
        logger.init_DB()
        logger.new_run(json.dumps({ k: v for k, (_, v, *_) in self._parameters.items() }))
        
        log.info(f"Running..")
        for task, (data_product, ), parameters in self.iterable():
            results, database_results = ({}, [])
            for spectrum in SpectrumList.read(data_product.path):

                if spectrum is None:
                    continue
                elif apogee_nn is not None and spectrum_overlaps(spectrum, parameters["apogee_wave_range"]):
                    network, prefix = (apogee_nn, "apogee_")
                elif boss_nn is not None and spectrum_overlaps(spectrum, parameters["boss_wave_range"]):
                    network, prefix = (boss_nn, "boss_")
                else:
                    # No network for this spectrum.
                    log.warn(f"No network for spectrum {spectrum} in data product {data_product}")
                    continue

                opt_keys = ("wave_range", "spectral_R", "N_chebyshev", "N_presearch_iter", "N_presearch")
                opt = { k: parameters[f"{prefix}{k}"] for k in opt_keys }

                result, meta, fit = fit_spectrum(spectrum, network, opt, logger)

                database_results.extend(result)
                hdu_results = list_to_dict(result)
                hdu_results.update(list_to_dict(meta))

                extname = get_extname(spectrum, data_product)
                results[extname] = hdu_results

            with database.atomic():
                task.create_or_update_outputs(ZetaPayneOutput, database_results)

            # Create astraStar/astraVisit data product and link it to this task.
            create_pipeline_product(task, data_product, results)



if __name__ == "__main__":

    '''
    import numpy as np
    from astropy.table import Table
    from astra.utils import expand_path
    t = Table.read(expand_path("$BOSS_SPECTRO_REDUX/v6_0_9/spAll-v6_0_9.fits"))
    is_ob = np.array(["_ob_" in carton for carton in t["FIRSTCARTON"]])

    catalogids = list(set(map(int, t["CATALOGID"][is_ob])))

    from astra.database.apogee_drpdb import Star, Visit

    q = Star.select(Star.catalogid).where(Star.catalogid.in_(catalogids))

    catalogid = q.first().catalogid # 27021598755571980

    # Let's ingest the (daily) apVisit files for this source,
    #   and then create a mwmVisit/mwmStar file
    #   and then test ZetaPayne on it.
    from astra.sdss.operators.apogee import get_or_create_data_product_from_apogee_drpdb

    visit_dps = []
    for visit in Visit.select().where(Visit.catalogid == catalogid):
        dp, created = get_or_create_data_product_from_apogee_drpdb(visit)
        visit_dps.append(dp)

    from astra.sdss.datamodels.mwm import create_mwm_data_products

    hdu_star, hdu_visit, meta = create_mwm_data_products(Source.get(catalogid))
    hdu_star.writeto(f"mwmStar-test-{catalogid}.fits")
    hdu_visit.writeto(f"mwmVisit-test-{catalogid}.fits")
    '''

    catalogid = 27021598755571980 # has been observed by both APOGEE and BOSS 
    # spAll file says FIRSTCARTON is mwm_ob_core 
    # apogee_drpdb says FIRSTCARTON is mwm_galactic_core ...
    
    # let's take a look.
    from astra.contrib.zetapayne.base import ZetaPayne
    t = ZetaPayne([f"mwmStar-test-{catalogid}.fits"])
    

    '''
    <DataProduct: id=506342>
    ipdb> task
    <Task: id=144221, name='astr
    '''



    '''
    # Get something from mwm_ob_*
    from astra.database.astradb import Source, DataProduct

    s = Source.get(27021598101135555)
    for dp in s.data_products:
        if dp.filetype == "mwmStar":
            break
    else:
        raise nope

    from astra.contrib.zetapayne.base import ZetaPayne
    t = ZetaPayne(
        [dp],
        NN_path="$MWM_ASTRA/component_data/ZetaPayne/BOSS.npz",
        wave_range=(4500, 10170),
        spectral_R=2000,
        N_chebyshev=15,
        N_presearch_iter=0,
        N_presearch=4000
    )
    
    # Let's find a source in mwm_ob_core that has been observed by APOGEE and BOSS
    '''




