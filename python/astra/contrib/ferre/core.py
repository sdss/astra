

import numpy as np
import multiprocessing as mp
import os
import re
import subprocess
import sys
import threading
import time

from collections import OrderedDict
from glob import glob
from io import StringIO, BytesIO
from inspect import getfullargspec
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from tqdm import tqdm

from astra.utils import log
from astra.tools.spectrum import Spectrum1D
from astropy import units as u

from astra.contrib.ferre import utils
from collections import OrderedDict
from luigi.freezing import FrozenOrderedDict

class Ferre(object):

    def __init__(
            self, 
            grid_header_path,
            frozen_parameters=None,
            interpolation_order=1,
            init_algorithm_flag=1,
            error_algorithm_flag=0,
            continuum_flag=None,
            continuum_order=None,
            continuum_reject=None,
            continuum_observations_flag=0,
            optimization_algorithm_flag=3,
            wavelength_interpolation_flag=0,
            lsf_shape_flag=0,
            use_direct_access=True,
            n_threads=1,
            directory_kwds=None,        
            input_weights_path=None,
            input_lsf_path=None,
            debug=False,
            **kwargs
        ):        


        self.executable = kwargs.pop("executable", "ferre")

        # Parse headers.
        self.headers = utils.read_ferre_headers(grid_header_path)

        defaults = [
            ("synthfile_format_flag", 1),
            ("input_parameter_path", "params.input"),
            ("input_wavelength_path", "wavelength.input"),
            ("input_flux_path", "flux.input"),
            ("input_uncertainties_path", "uncertainties.input"),
            ("output_parameter_path", "params.output"),
            ("output_flux_path", "flux.output"),
            ("output_normalized_input_flux_path", "normalized_flux.output"),
            #("input_weights_path", "weights.input"),
            #("input_lsf_path", "lsf.input"),
            ("n_dimensions", self.headers[0]["N_OF_DIM"])
            # n_pixels,
            # n_runs
        ]

        kwds = dict()
        for key, default_value in defaults:
            kwds[key] = kwargs.get(key, default_value)

        # These parameters do not go to FERRE.
        ignore = ("self", "frozen_parameters", "debug", "directory_kwds")
        for keyword in getfullargspec(Ferre.__init__).args:
            if keyword not in ignore:
                kwds[keyword] = eval(keyword)

        self.frozen_parameters = frozen_parameters or None
        # We assume a parameter will be fit, unless it is frozen.
        if self.frozen_parameters is not None:
            unknown_parameters = set(self.frozen_parameters).difference(self.parameter_names)
            if unknown_parameters:
                raise ValueError(f"unknown parameter(s): {unknown_parameters} (available: {self.parameter_names})")
            
            indices = [i for i, pn in enumerate(self.parameter_names, start=1) if pn not in self.frozen_parameters]
            indices = sorted(indices)
            kwds["n_parameters"] = len(indices)
            kwds["parameter_search_indices"] = indices

        else:
            kwds["n_parameters"] = kwds["n_dimensions"]
            kwds["parameter_search_indices"] = 1 + np.arange(kwds["n_parameters"])

        # Things for context manager.
        self.context_manager = False
        # By default if we use a context manager, we want the input stream path to go to the 
        # input_parameter_path so that we can use the context manager as an interpolator.
        self._input_stream_path = kwargs.get("input_stream_path", "input_parameter_path")
        self._output_stream_path = kwargs.get("output_stream_path", "output_flux_path")

        # Directory.
        self._directory_kwds = directory_kwds

        self.kwds = kwds
        self.debug = debug

        return None


    @property
    def parameter_names(self):
        return self.headers[0]["LABEL"]

    @property
    def wavelength(self):
        return tuple([
            utils.wavelength_array_from_ferre_header(header) * u.Angstrom \
            for header in self.headers[1:]
        ])


    @property
    def directory(self):
        try:
            self._directory
        except AttributeError:
            kwds = self._directory_kwds or dict()
            # Make the parent directory if we need to.
            if kwds.get("dir", None) is not None:
                os.makedirs(kwds["dir"], exist_ok=True)
            self._directory = mkdtemp(**kwds)
            log.info(f"Created temporary directory {self._directory}")
        
        return self._directory


    @property
    def grid_mid_point(self):
        return self.headers[0]["LLIMITS"] + 0.5 * self.headers[0]["STEPS"] * self.headers[0]["N_P"]


    @property
    def grid_limits(self):
        return (
            self.headers[0]["LLIMITS"],
            self.headers[0]["LLIMITS"] + self.headers[0]["STEPS"] * self.headers[0]["N_P"]
        )

    @property
    def n_dimensions(self):
        return self.headers[0]["N_OF_DIM"]

    def wavelength_mask(self, wavelength):

        try:
            wl = wavelength.value
        except:
            wl = wavelength

        P = 0
        mask = np.zeros(wavelength.size, dtype=bool)
        for model_wavelength in self.wavelength:
            s_index, e_index = wl.searchsorted(model_wavelength[[0, -1]].value)
            mask[s_index:e_index + 1] = True
            P += model_wavelength.size
        
        assert np.sum(mask) == P
        return mask


    def in_grid_limits(self, points):
        lower, upper = self.grid_limits
        return (upper >= points) * (points >= lower)
            
    def parse_initial_parameters(self, initial_parameters, N):

        parsed_initial_parameters = np.tile(self.grid_mid_point, N).reshape((N, -1))
        if initial_parameters is not None and len(initial_parameters) > 0:
            self.kwds["init_algorithm_flag"] = 0

            # Parse the initial parameters.
            if isinstance(initial_parameters, (dict, OrderedDict, FrozenOrderedDict)):
                initial_parameters = [initial_parameters]
            
            D = self.headers[0]["N_OF_DIM"]
            for i, ip in enumerate(initial_parameters):
                for j, pn in enumerate(self.parameter_names):
                    try:
                        v = ip[pn]
                    except KeyError:
                        continue
                    else:
                        parsed_initial_parameters[i, j] = v
            
        # Apply frozen parameters.
        if self.frozen_parameters is not None:
            for pn, v in self.frozen_parameters.items():
                if v is not None:
                    index = self.parameter_names.index(pn)
                    parsed_initial_parameters[:, index] = v

        return parsed_initial_parameters


    def fit(self, spectra, initial_parameters=None, full_output=False, names=None):

        if isinstance(spectra, Spectrum1D):
            spectra = [spectra]

        N = len(spectra)
        parsed_initial_parameters = self.parse_initial_parameters(initial_parameters, N)

        # TODO: Apply edge clipping.

        wavelengths = np.vstack([spectrum.wavelength.value for spectrum in spectra])
        flux = np.vstack([spectrum.flux.value for spectrum in spectra])
        uncertainties = np.vstack([spectrum.uncertainty.array**-0.5 for spectrum in spectra])

        assert flux.shape[0] == len(spectra), "Mis-match in flux array"
        assert uncertainties.shape[0] == len(spectra), "Mis-match in uncertainties array"

        # Make sure we are not sending nans etc.
        bad = ~np.isfinite(flux) \
            + ~np.isfinite(uncertainties) \
            + (uncertainties == 0) \
            + (flux < 0) # TODO: Trying.....
        flux[bad] = 1.0
        uncertainties[bad] = 1e6

        # We only send specific set of pixels to FERRE.
        mask = self.wavelength_mask(wavelengths[0])

        # TODO: Should we be doing this if we are using wavelength_interpolation_flag > 0?
        wavelengths = wavelengths[:, mask]
        flux = flux[:, mask]
        uncertainties = uncertainties[:, mask]

        # Write wavelengths?
        if self.kwds["wavelength_interpolation_flag"] > 0:            
            utils.write_data_file(
                wavelengths,
                os.path.join(self.directory, self.kwds["input_wavelength_path"])
            )

        # Write flux.
        utils.write_data_file(
            flux,
            os.path.join(self.directory, self.kwds["input_flux_path"])
        )

        # Write uncertainties.
        utils.write_data_file(
            uncertainties,
            os.path.join(self.directory, self.kwds["input_uncertainties_path"])
        )

        # Write initial parameters to disk.
        if names is None:
            names = [f"idx_{i:.0f}" for i in range(len(parsed_initial_parameters))]
        
        with open(os.path.join(self.directory, self.kwds["input_parameter_path"]), "w") as fp:            
            #for i, each in enumerate(parsed_initial_parameters):
            #    fp.write(utils.format_ferre_input_parameters(*each, star_name=f"idx_{i:.0f}"))
            for star_name, ip in zip(names, parsed_initial_parameters):
                fp.write(utils.format_ferre_input_parameters(*ip, star_name=star_name))
            
        # Execute.
        self._execute(total=N)
        
        erroneous_output = -999.999

        # Parse outputs.
        try:
            # Parse parameters.
            param, param_errs, meta = utils.read_output_parameter_file(
                os.path.join(self.directory, self.kwds["output_parameter_path"]),
                n_dimensions=self.n_dimensions
            )

            any_bad = False
            for j, (p, e) in enumerate(zip(param, param_errs)):
                if np.all(p == erroneous_output) and np.all(e == erroneous_output):
                    log.warn(f"Error in output for index {j}")
            
            if any_bad:
                log.warn(f"FERRE STDOUT:\n{self.stdout}")

            if full_output:
                output_flux = np.atleast_2d(
                    np.loadtxt(
                        os.path.join(self.directory, self.kwds["output_flux_path"])
                    )
                )

                if self.kwds["continuum_flag"] is not None:

                    normalized_flux = np.atleast_2d(
                        np.loadtxt(
                            os.path.join(self.directory, self.kwds["output_normalized_input_flux_path"])
                        )
                    )

                    meta.update(normalized_input_flux=normalized_flux)

                return (param, param_errs, output_flux, meta)

            return (param, param_errs)
        
        except:
            log.info("Something went wrong. Could have been in FERRE.")
            log.info(self.stdout)
            log.error(self.stderr)
            raise 
        


    def __call__(self, *params, timeout=30):

        try:
            self.process
        
        except AttributeError:
            raise RuntimeError("this function can only be called when used as a context manager")
        
        # Write the parameters to the process stdin.
        content = utils.format_ferre_input_parameters(*params)
        with open(os.path.join(self.directory, "params.input"), "w") as fp:
            fp.write(content)

        self.process.stdin.write(content)
        self.process.stdin.flush()

        # Get outputs.
        content = self._queue.get(timeout=timeout)

        try:
            # TODO: What if output is params path.
            if self._output_stream_path == "output_flux_path":
                # Parse as flux.
                output = np.loadtxt(StringIO(content))

            elif self._output_stream_path == "output_parameter_path":
                # Parse as parameters.
                raise NotImplementedError

            else:
                raise ValueError("unknown")

        except ValueError:
            # An error probably occurred.
            # Check that there is no other output.
            while True:
                try:
                    content += self._queue.get_nowait()
                
                except mp.Empty:
                    break
                
            log.exception(f"FERRE returned the error:\n{content}")
            raise RuntimeError(f"Ferre failed to interpolate spectrum at {params}")
        
        return output


    def _write_ferre_input_file(self):
        """
        Write a FERRE input file.
        """
        
        ferre_kwds = utils.prepare_ferre_input_keywords(**self.kwds)

        input_path = os.path.join(self.directory, "input.nml")
        with open(input_path, "w") as fp:
            fp.write(utils.format_ferre_control_keywords(ferre_kwds))

        return input_path
    

    def _execute(self, bufsize=1, encoding="utf-8", interactive=False, total=None):
        """
        Write an input file and execute FERRE.
        """

        self._setup()

        self._write_ferre_input_file()

        try:
            self.process = subprocess.Popen(
                [self.executable],
                cwd=self.directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=bufsize,
                encoding=encoding,
                close_fds="posix" in sys.builtin_module_names
            )

        except subprocess.CalledProcessError:
            log.exception(f"Exception when calling {self.executable} in {self.directory}")
            raise

        else:
            if interactive:
                return self.process

            self._monitor_progress(total=total)
                
            return None


    def _monitor_progress(self, total=None, interval=1):
    
        self.stdout, self.stderr = ("", "")

        # Monitor progress.
        # Note: FERRE normally writes fluxes before parameters, so we will use the flux path to
        # monitor progress.
        output_flux_path = os.path.join(self.directory, self.kwds["output_flux_path"])

        done = 0
        with tqdm(total=total, desc="FERRE", unit="star") as pbar:
            while total > done:
                if self.process is not None:
                    self._communicate(timeout=interval)
                else:
                    sleep(interval)

                if os.path.exists(output_flux_path):
                    lines = utils.line_count(output_flux_path)
                    pbar.update(lines - done)
                    done = lines
                
                error_count = self.stderr.lower().count("error")
                if error_count > 0:
                    pbar.set_description(f"FERRE ({error_count:.0f} errors)")

                pbar.refresh()

        
        # Do a final I/O communicate because we are done.
        if self.process is not None:
            self._communicate()
        return None


    def _communicate(self, timeout=None):
        try:
            stdout, stderr = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            None
        else:
            self.stdout += stdout
            self.stderr += stderr
        return None


    def _setup(self):
        # Set up temporary directory.
        assert os.path.exists(self.directory)

        # Copy input weights file.
        if self.kwds["input_weights_path"] is not None:
            basename = os.path.basename(self.kwds["input_weights_path"])
            copyfile(
                self.kwds["input_weights_path"],
                os.path.join(self.directory, basename)
            )
            # Replace with relative path.
            self.kwds["input_weights_path"] = basename

        return None


    def teardown(self):
        if not self.debug:
            rmtree(self.directory)
            return True
            
        else:
            log.warning(f"Not tearing down {self.directory} because debug = {self.debug}")
            return False
        


    def __enter__(self):

        # If we are using the `with Ferre() as model:` context statement then we are going to have
        # something with many inputs and outputs.
        self.context_manager = True

        # This means we need to execute FERRE now, and we should already know where the 
        # "input stream path" goes to, because that has to be defined for the input Ferre file.
        self.kwds[self._input_stream_path] = "/dev/stdin"
        self.kwds[self._output_stream_path] = "/dev/stderr"

        if self._input_stream_path == "input_parameter_path":
            # Need to set nov = 0 for interpolation mode for FERRE.
            self.kwds["n_parameters"] = 0
        
        # Execute FERRE.
        process = self._execute(interactive=True)

        # Set up non-blocking thread.
        self._queue = mp.Queue()
        self._thread = threading.Thread(
            target=_non_blocking_pipe_read,
            args=(process.stderr, self._queue),
            daemon=True
        )
        self._thread.needed = True
        self._thread.start()

        return self


    def __exit__(self, type, value, traceback):
        # Kill off our non-blocking pipe read.        
        self._thread.needed = False

        # Capture outputs for verbosity.
        self.stdout, self.stderr = self.process.communicate()

        self.teardown()
        return None



class FerreQueue(Ferre):

    def _execute(self, bufsize=1, encoding="utf-8", total=None, **kwargs):
        """
        Write an input file and execute FERRE, using the PBS/slurm queue submission system.
        """

        self._setup()
        self._write_ferre_input_file()

        # It's bad practice to import things here, but we do so to avoid import errors on
        # non-Utah systems, since the pbs package is not a requirement and only available
        # at Utah.
        import pbs

        kwds = dict(
            label="FERRE",
            nodes=1,
            ppn=16,
            walltime='24:00:00',
            alloc='sdss-kp',
            verbose=True
        )        
        kwds.update(self.queue_kwds)

        queue = pbs.queue()
        queue.create(**kwds)
        queue.append(
            self.executable,
            dir=self.directory
        )
        queue.commit(hard=True, submit=True)
        print(f"PBS queue key: {queue.key}")
        self.process = None

        self._monitor_progress()
        return None
    



def _non_blocking_pipe_read(stream, queue):
    """ 
    A non-blocking and non-destructive stream reader for long-running interactive jobs. 
    
    :param stream:
        The stream to read (e.g., process.stderr).
    
    :param queue:
        The multiprocessing queue to put the output from the stream to.        
    """

    thread = threading.currentThread()
    while getattr(thread, "needed", True):
        f = stream.readline()
        queue.put(f)

    return None



if __name__ == "__main__":


    from astra.tools.spectrum import Spectrum1D

    spectrum = Spectrum1D.read(
        "/home/ubuntu/data/sdss/dr16/apogee/spectro/redux/r12/stars/apo25m/120-60/apStar-r12-2M00463666+0347457.fits",
        format="SDSS APOGEE apStar"
    )
    # Grid header path found by dispatcher.
    grid_header_path = "/home/andy/data/sdss/apogeework/apogee/spectro/speclib/turbospec/marcs/solarisotopes/tdM_180901_lsfa_l33/p_apstdM_180901_lsfa_l33_012_075.hdr"


    from astropy.io import fits

    image = fits.open("/home/ubuntu/data/sdss/dr16/apogee/spectro/aspcap/r12/l133/apo25m/070+69_MGA/aspcapStar-r12-2M14212225+3908580.fits")

    #  Teff, logg, vmicro, [M/H], [C/M], [N/M], [alpha/M], vsini/vmacro, [O/M]
    teff, logg, vmicro, m_h, c_m, n_m, alpha_m, vsini, o_m = image[4].data["PARAM"][0]

    #model.parameter_names                                                                                                                                                                                                                                                                                                          
    #Out[78]: ['TEFF', 'LOGG', 'METALS', 'O Mg Si S Ca Ti', 'N', 'C', 'LOG10VDOP']

    point = np.array([teff, logg, m_h, alpha_m, n_m, c_m, vsini])

    spectrum = Spectrum1D.read(
        "/home/ubuntu/data/sdss/dr16/apogee/spectro/redux/r12/stars/apo25m/070+69_MGA/apStar-r12-2M14212225+3908580.fits",
        format="SDSS APOGEE apStar"
    )


    grid_header_path = "/home/andy/data/sdss/apogeework/apogee/spectro/speclib/turbospec/marcs/giantisotopes/tgGK_180901_lsfa_l33/p_apstgGK_180901_lsfa_l33_012_075.hdr"

    #point = np.array([ 6.1508e+03,  4.5033e+00,  6.2916e-02,  0, 0, 0, 0])
    
    kwds = dict(
        grid_header_path=grid_header_path, 
        debug=True,
        continuum_flag=1,
        continuum_order=4,
        continuum_reject=0.1,
        optimization_algorithm_flag=3,
        wavelength_interpolation_flag=0,
        input_weights_path="global_mask_v02.txt"
    )

    '''

    with Ferre(grid_header_path=grid_header_path) as model:
        mid_point_except_teff_logg_feh = model.grid_mid_point[:-3]

        teff_logg_feh = np.random.uniform(
            model.grid_limits[0][-3:],
            model.grid_limits[1][-3:],
            size=(30, 3)
        )

        ok = []
        for each in teff_logg_feh:
            p = list(mid_point_except_teff_logg_feh) + list(each)
            flux = model(*p[::-1])
            ok.append(np.any(flux > -1000))
            print(p, ok[-1])

    '''

    with Ferre(**kwds) as model:
        interpolated_flux = model(*point[::-1])
        interpolated_flux2 = model(*point)


    

    from time import time


    point = np.array([
        5.911906e+03,
        4.349000e+00,
        7.700000e-02,
        2.000000e-02,
        -9.000000e-03,
        1.290000e-01,
        -5.000000e-01
    ])

    model = Ferre(**kwds)
    t_init = time()
    p_opt, p_cov, model_flux, meta = model.fit(
        spectrum, 
        initial_parameters=point[::-1],
        full_output=True,
    )
    print(time() - t_init, p_opt[0], meta["log_chisq_fit"])

    model = Ferre(**kwds)
    t_init = time()
    p_opt, p_cov, model_flux, meta = model.fit(
        spectrum, 
        full_output=True,
    )    
    print(time() - t_init, p_opt[0], meta["log_chisq_fit"])

    print(p_opt)


    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import NullFormatter

    fig, ax = plt.subplots()

    ax.plot(
        spectrum.wavelength,
        spectrum.flux.value[0, :],
        c="#000000"
    )

    sigma = spectrum.uncertainty.array**-0.5
    ax.fill_between(
        spectrum.wavelength,
        spectrum.flux.value[0, :] - sigma[0, :],
        spectrum.flux.value[0, :] + sigma[0, :],
        facecolor="#cccccc",
        zorder=-1
    )

    ax.set_ylim(0, 3500)

    fig.savefig("tmp1.png")



    '''
    fig, axes = plt.subplots(
        ncols=1, nrows=3,
        constrained_layout=True,
        figsize=(15, 5)
    )


    v = np.nanmedian(spectrum.flux)

    P = 0
    for i, (ax, wavelength) in enumerate(zip(axes, model.wavelength)):
        p = wavelength.size
        ax.plot(
            wavelength, 
            model_flux[0][P:P+p],
            c="tab:red"
        )
        
        #ax.plot(
        #    wavelength,
        #    interpolated_flux[P:P+p],
        #    c="tab:blue",
        #)


        if "normalized_input_flux" in meta:
            ax.plot(
                wavelength,
                meta["normalized_input_flux"][0][P:P+p],
                c="#000000"
            )

        else:
            ax.plot(
                spectrum.wavelength,
                spectrum.flux.value[0, :],
                c="#000000"
            )

        ax.set_xlim(
            *wavelength.value[[0, -1]]
        )
        ax.set_ylim(0, 1.2)

        P += p

    fig.savefig("tmp.png", dpi=600)
    
    '''


    width_ratios = np.array([np.ptp(wl.value) for wl in model.wavelength])
    width_ratios = width_ratios/np.min(width_ratios)

    fig, axes = plt.subplots(
        ncols=3, nrows=2, figsize=(40, 5),
        gridspec_kw=dict(width_ratios=width_ratios, height_ratios=[1, 5])
    )

    diff_axes = axes[0]
    plot_axes = axes[1]

    P = 0
    for i, (ax, ax_diff, wavelength) in enumerate(zip(plot_axes, diff_axes, model.wavelength)):
        p = wavelength.size
        ax.plot(
            wavelength, 
            model_flux[0][P:P+p],
            c="tab:red",
            lw=1,
        )

        ax.plot(
            wavelength,
            interpolated_flux[P:P+p],
            c="tab:blue",
            lw=1
        )


        if "normalized_input_flux" in meta:
            ax.plot(
                wavelength,
                meta["normalized_input_flux"][0][P:P+p],
                c="#000000",
                lw=1,
            )

            ax_diff.plot(
                wavelength,
                meta["normalized_input_flux"][0][P:P+p] - model_flux[0][P:P+p],
                c="#000000",
                lw=1,
            )


            s_index, e_index = spectrum.wavelength.value.searchsorted(wavelength[[0, -1]]) 
            
            continuum = spectrum.flux.value[0, s_index:e_index + 1] / meta["normalized_input_flux"][0][P:P+p]

            sigma = spectrum.uncertainty.array[0, s_index:e_index + 1]**-0.5 / continuum

            ax.fill_between(
                wavelength, 
                meta["normalized_input_flux"][0][P:P+p] - sigma,
                meta["normalized_input_flux"][0][P:P+p] + sigma,
                facecolor="#CCCCCC",
                edgecolor="#CCCCCC",
                zorder=-1
            )

            ax_diff.fill_between(
                wavelength,
                -sigma,
                +sigma,
                facecolor="#CCCCCC",
                edgecolor="#CCCCCC",
                zorder=-1
            )


        else:
            ax.plot(
                spectrum.wavelength,
                spectrum.flux.value[0, :],
                c="#000000"
            )
        


        ax_diff.set_ylim(-0.1, 0.1)
        ax.set_ylim(0.5, 1.1)

        ax_diff.axhline(0, c="#666666", ls=":", zorder=-10)
        

        for ax_ in (ax_diff, ax):

            ax_.set_xlim(*wavelength.value[[0, -1]])

            if not ax.is_last_row():
                ax.set_xticks([])

            if len(model.wavelength) > 0:
                if not ax_.is_last_col():
                    ax_.spines["right"].set_visible(False)
                    ax_.tick_params(right=False, which="both")
                    
                if not ax_.is_first_col():
                    ax_.spines["left"].set_visible(False)
                    ax_.tick_params(left=False, which="both")
                    ax_.yaxis.set_major_formatter(NullFormatter())

            '''
            if ii == 0 and not nregions == 1:
                thisax_.spines['right'].set_visible(False)
                thisax_.tick_params(right=False,which='both')
            elif ii == (nregions-1) and not nregions == 1:
                thisax_.spines['left'].set_visible(False)
                thisax_.tick_params(labelleft='off')
                thisax_.tick_params(left=False,which='both')
            elif not nregions == 1:
                thisax_.spines['left'].set_visible(False)
                thisax_.spines['right'].set_visible(False)
                thisax_.tick_params(labelleft='off')
                thisax_.tick_params(left=False,which='both')
                thisax_.tick_params(right=False,which='both')
            '''

            # Plot cut-out markers
            cutOutkwargs = dict(transform=ax_.transAxes,color='k',
                                clip_on=False)

            nregions = len(model.wavelength)
            dx = np.ones(nregions)
            skipdx = 0.015

            d = .015 # how big to make the diagonal lines in axes coordinates
            #d = 0.015 if ax.is_last_row() else 0.015/5
            skipdx = 0.015
            slope= 1./(dx[i]+0.2*skipdx)/3.
            slope = slope if ax.is_last_row() else 5 * slope 
            if i == 0 and not nregions == 1:
                ax_.plot((1-slope*d,1+slope*d),(-d,+d), **cutOutkwargs)
                ax_.plot((1-slope*d,1+slope*d),(1-d,1+d), **cutOutkwargs)
            elif i == (nregions-1) and not nregions == 1:
                ax_.plot((-slope*d,+slope*d),(-d,+d), **cutOutkwargs)
                ax_.plot((-slope*d,+slope*d),(1-d,1+d), **cutOutkwargs)
            elif not nregions == 1:
                ax_.plot((1-slope*d,1+slope*d),(-d,+d), **cutOutkwargs)
                ax_.plot((1-slope*d,1+slope*d),(1-d,1+d), **cutOutkwargs)
                ax_.plot((-slope*d,+slope*d),(-d,+d), **cutOutkwargs)
                ax_.plot((-slope*d,+slope*d),(1-d,1+d), **cutOutkwargs)

        P += p
    
    fig.tight_layout()
    fig.savefig("tmp.png", dpi=600)

    raise a





    grid_header_path = "/home/andy/data/sdss/apogeework/apogee/spectro/speclib/turbospec/marcs/giantisotopes/tgGK_180901_lsfa_l33/p_apstgGK_180901_lsfa_l33_012_075.hdr"

    import numpy as np
    from astropy.units import Quantity
    from astropy.nddata import InverseVariance
    from astropy import units as u
    units = u.Unit("1e-17 erg / (Angstrom cm2 s)")

    from time import time

    if False:
            
        with Ferre(grid_header_path) as model:
            points = np.random.uniform(*model.grid_limits, size=(30, 7))

        for interpolation_order in (0, 1, 2, 3, 4):

            with Ferre(grid_header_path, interpolation_order=interpolation_order) as model:
                print(model.directory)

                times = []
                for point in points:
                    model(*point[::-1])
                    times.append(time())

                diffs = np.diff(times)
                print(f"Interpolation order {interpolation_order}: median={np.median(diffs):.3f} (stddev={np.std(diffs):.3f}, min={np.min(diffs):.3f}, max={np.max(diffs):.3f})")

        
    point = (5000., 2.9, -0.9, 0.2, 0.1, 0.1, 0.0)
    point2 = (5100., 2.8, -0.9, 0.2, 0.1, 0.1, 0.0)

    debug = False

    with Ferre(grid_header_path, debug=debug) as model:
        flux = model(*point)
        flux2 = model(*point2)

        spectra = [
            Spectrum1D(
                spectral_axis=np.hstack(model.wavelength),
                flux=flux * units,
                uncertainty=InverseVariance(1e6 * np.ones_like(flux))),
            Spectrum1D(
                spectral_axis=np.hstack(model.wavelength),
                flux=flux2 * units,
                uncertainty=InverseVariance(1e6 * np.ones_like(flux)))
        ]


    second_model = Ferre(grid_header_path, debug=debug)

    result = second_model.fit(
        spectra,
        initial_parameters=[point, point2]
        )

    result2 = second_model.fit(
        spectra[::-1],
        initial_parameters=[point2, point]
        )



