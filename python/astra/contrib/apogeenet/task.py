
import torch
import numpy as np
from astra import log
from astra.task import ExecutableTask, Parameter
from astra.database.astradb import Output, TaskOutput
from astra.tools.spectrum import Spectrum1D
from astra.utils import timer

from astra.contrib.apogeenet.model import Model
from astra.contrib.apogeenet.database import ApogeeNet
from astra.contrib.apogeenet.utils import get_metadata



def create_output(model, task, **kwargs):
    output = Output.create()
    kwds = dict(task=task, output=output)
    TaskOutput.create(**kwds)
    return model.create(**kwds, **kwargs)


class StellarParameters(ExecutableTask):

    model_path = Parameter("model_path", bundled=True)
    num_uncertainty_draws = Parameter("num_uncertainty_draws", default=100)
    large_error = Parameter("large_error", default=1e10)
    

    def execute(self):
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Model(self.model_path, device)

        result = []
        for i, (task, data_products, parameters) in enumerate(self.iterable()):
            
            if i == 1:
                from time import sleep
                sleep(2)

            assert len(data_products) == 1

            spectrum = Spectrum1D.read(data_products[0].path)
            keys, metadata, metadata_norm = get_metadata(spectrum)

            N, P = spectrum.flux.shape
            flux = np.nan_to_num(spectrum.flux.value).astype(np.float32).reshape((N, 1, P))
            meta = np.tile(metadata_norm, N).reshape((N, -1))

            flux = torch.from_numpy(flux).to(device)
            meta = torch.from_numpy(meta).to(device)

            with torch.set_grad_enabled(False):
                predictions = model.predict_spectra(flux, meta)
                if device != "cpu":
                    predictions = predictions.cpu().data.numpy()
            
            # Replace infinites with non-finite.
            predictions[~np.isfinite(predictions)] = np.nan

            result.append(predictions)
            
        return result


    def post_execute(self):

        if not ApogeeNet.table_exists():
            log.info(f"Creating database table for ApogeeNet")
            ApogeeNet.create_table()

        # Create rows in the database.
        i = 0
        for (task, data_products, parameters), result in zip(self.iterable(), self.result):
            
            # S/N will be per entry in `result`.
            N = len(result)
            snrs = (data_products[0].metadata or {}).get("snr", [-1] * N)

            for snr, row in zip(snrs, result):
                log_teff, logg, fe_h = row
                
                create_output(
                    ApogeeNet, 
                    task,
                    snr=snr,
                    teff=10**log_teff,
                    logg=logg,
                    fe_h=fe_h,
                    u_teff=1,
                    u_logg=1,
                    u_fe_h=1,
                    teff_sample_median=1,
                    logg_sample_median=1,
                    fe_h_sample_median=1,
                    bitmask_flag=0
                )

