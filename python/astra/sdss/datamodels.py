import numpy as np
import datetime
from astropy.io import fits

from astra import __version__ as astra_version

from .catalog import (get_sky_position, get_gaia_dr2_photometry)

class BaseDataModel:
    pass


class MilkyWayMapperStar(BaseDataModel):

    pass

def get_fits_format_code(value):
    fits_format_code = {
        bool: "L",
        int: "K",
        str: "A",
        float: "E",
        type(None): "A"
    }.get(type(value))
    assert fits_format_code is not None
    if fits_format_code == "A" and value is not None:
        return f"{len(value)}A"
    return fits_format_code



class AstraStar(BaseDataModel):
    
    """
    AstraStar files contain output parameters and best-fit model(s) from one pipeline for a single star.
    """

    def __init__(
        self,
        catalog_id,
        input_data_products,        
        wavelength, 
        model_flux,
        model_ivar=None,
        rectified_flux=None,
    ) -> None:
        self.catalog_id = catalog_id
        self.input_data_products = input_data_products
        self.wavelength = wavelength
        self.model_flux = model_flux
        self.model_ivar = model_ivar
        self.rectified_flux = rectified_flux
        return None

    @cached_property
    def sky_position(self):
        """Return the sky position given the known catalog identifier."""
        return get_sky_position(self.catalog_id)

    # get healpix 
    @cached_property
    def healpix(self):
        """Return the HEALPix given the known catalog identifier."""
        ra, dec = self.sky_position
        import healpy as hp
        return hp.ang2pix(
            nside=128, 
            ra=ra,
            dec=dec,
            lonlat=True
        )
    

    @property
    def path(self):
        return (
            f"$MWM_ASTRA/{astra_version}/healpix/{healpix % 100}/{healpix}/"
            f"astraStar-{component_name}-{catalog_id}.fits"
        )


    def write(self):
        """
        Write to disk.
        """
        # TODO: This will be refactored once we have a better understanding of the problem.

        catalog_id = 1
        ra = 1.0
        dec = 0.0
        healpix = 5
        object_id = "J018398+234723"

        files = "aslkjfa.fits,that.fits"

        crval, cdelt, crpix = (1, 2, 3)
        pipeline = "SLAM v1.0"


        parameters = {
            'continuum_flag': 1,
        'continuum_observations_flag': 1,
        'continuum_order': 4,
        'continuum_reject': 0.3,
        'continuum_segment': None,
        'error_algorithm_flag': 1,
        'f_access': None,
        'f_format': 1,
        'ferre_kwds': {'TYPETIE': 1,
        'INDTIE(1)': 2,
        'TTIE0(1)': 0,
        'TTIE(1,6)': -1,
        'INDTIE(2)': 3,
        'TTIE0(2)': 0,
        'TTIE(2,6)': -1,
        'INDTIE(3)': 4,
        'TTIE0(3)': 0,
        'TTIE(3,6)': -1,
        'NTIE': 3},
        'frozen_parameters': {'LOG10VDOP': True,
        'C': True,
        'N': True,
        'O Mg Si S Ca Ti': True,
        'LGVSINI': True,
        'METALS': False,
        'LOGG': True,
        'TEFF': True},
        'full_covariance': False,
        'initial_parameters': [{'teff': 4360.509,
            'logg': 4.207,
            'metals': -0.145,
            'o_mg_si_s_ca_ti': -0.019,
            'lgvsini': 0.947,
            'c': -0.035,
            'n': -0.045,
            'log10vdop': -0.507}],
        'interpolation_order': 3,
        'lsf_shape_flag': 0,
        'lsf_shape_path': None,
        'n_threads': 32,
        'normalization_kwds': {'median_filter_from_task': 1326},
        'normalization_method': 'astra.contrib.aspcap.continuum.MedianFilterNormalizationWithErrorInflation',
        'optimization_algorithm_flag': 3,
        'parent_dir': '$MWM_ASTRA/0.2.2/ferre/',
        'pca_chi': False,
        'pca_project': False,
        'slice_args': None,
        'wavelength_interpolation_flag': 0,
        'weight_path': '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/component_data/aspcap/masks/Al.mask',
        'header_path': '$SAS_BASE_DIR/dr17/apogee/spectro/speclib/synth/synspec/marcs/solarisotopes/sdGK_200921nlte_lsfc/p_apssdGK_200921nlte_lsfc_012_075.hdr'}

        flux = np.random.uniform(size=10_000)
        ivar = np.random.uniform(size=10_000)
        rectified_flux = np.random.uniform(size=10_000)

        task_id = 100
        task_name = "astra.contrib.ferre.base.Ferre"


        hdu = fits.PrimaryHDU(
            header=fits.Header(
                [
                    (
                        "DATE", 
                        datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                        "File creation date (UTC)"
                    ),
                    (   
                        "ASTRAVER", 
                        astra_version,
                        "Software version of Astra"
                    ),
                    (
                        "CAT_ID", # '#CATALOG'? 'CATID'?
                        catalog_id,
                        "SDSS-V catalog identifier"
                    ),
                    (
                        "OBJ",
                        object_id,
                        "Object identifier"
                    ),
                    (
                        "RA",
                        ra,
                        "RA (J2000)" # TODO: What source of RA?
                    ),
                    (
                        "DEC",
                        dec,
                        "DEC (J2000)" # TODO: What source of DEC
                    ),
                    (
                        "HEALPIX",
                        healpix,
                        "HEALPix location"
                    ),
                    (
                        "INPUTS",
                        files,
                        "Input data products"
                    )
                ]
            )
        )



        header = fits.Header([
            ("PIPELINE", pipeline),
            ("CRVAL1", crval),
            ("CDELT1", cdelt),
            ("CRPIX1", crpix),
            ("CTYPE1", "LOG-LINEAR"),
            ("DC-FLAG", 1),            
        ])



        flux_col = fits.Column(name="model_flux", format="E", array=flux)
        ivar_col = fits.Column(name="model_ivar", format="E", array=ivar)
        rectified_flux_col = fits.Column(name="rectified_flux", format="E", array=rectified_flux)

        hdu_spectrum = fits.BinTableHDU.from_columns(
            [flux_col, ivar_col, rectified_flux_col],
            header=header
        )

        # Task parameters
        task_columns = [
            fits.Column(name="task_id", format="I", array=[task_id]),
            fits.Column(name="task_name", format="30A", array=[task_name])
        ]


        for parameter_name, parameter_value in parameters.items():

            if isinstance(parameter_value, (dict, list)):
                # Json-ify.
                parameter_value = json.dumps(parameter_value)

            format_code = get_fits_format_code(parameter_value)
            if parameter_value is None:
                parameter_value = ""
            task_columns.append(
                fits.Column(
                    name=parameter_name,
                    format=format_code,
                    array=[parameter_value]
                )
            )

        # Store the outputs from each task.
        hdu_tasks = fits.BinTableHDU.from_columns(
            task_columns,
        )

        # Results in the hdu_tasks?
        # We can use Table(.., descriptions=(..., )) to give comments for each column type
        image = fits.HDUList([
            hdu,
            hdu_spectrum,
            hdu_tasks
        ])


        # HDU1 contains columns flux, ivar, rectified_flux
