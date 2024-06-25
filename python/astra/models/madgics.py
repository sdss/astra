import numpy as np
import h5py
import concurrent
import pickle
from peewee import (
    AutoField,
    ForeignKeyField,
    IntegerField,
    TextField, 
    FloatField,
    BigIntegerField
)
    
from tqdm import tqdm
from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.fields import PixelArray, PixelArrayAccessorHDF, BasePixelArrayAccessor, LogLambdaArrayAccessor
from astra.specutils.resampling import wave_to_pixel, sincint

from astra.glossary import Glossary
from astra.models.apogee import ApogeeVisitSpectrum


class MadgicsPixelArrayAccessor(BasePixelArrayAccessor):

    """A class to access pixel arrays stored in a HDF-5 file."""

    def __get__(self, instance, model, instance_type=None, **kwargs):
        if instance is not None:
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                with h5py.File(self.kwargs["path"], "r") as fp:
                    value = fp[self.kwargs["key"]][instance.map2madgics - 1]
                    if self.transform is not None:
                        value = self.transform(value, instance)
                                    
                    instance.__pixel_data__.setdefault(self.name, value)            
            finally:
                return instance.__pixel_data__[self.name]

        return self.field


class ResampledMadgicsPixelArrayAccessor(BasePixelArrayAccessor):

    """A class to access pixel arrays stored in a HDF-5 file."""

    def __get__(self, instance, model, instance_type=None, **kwargs):
        if instance is not None:
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                with h5py.File(self.kwargs["path"], "r") as fp:
                    value = fp[f"{self.kwargs['key']}/{instance.map2madgics}"][:]
                    if self.transform is not None:
                        value = self.transform(value, instance)
                                    
                    instance.__pixel_data__.setdefault(self.name, value)            
            finally:
                return instance.__pixel_data__[self.name]

        return self.field

class StarPickledPixelArrayAccessor(BasePixelArrayAccessor):
    def __get__(self, instance, model, instance_type=None, **kwargs):
        if instance is not None:
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:

                with h5py.File(self.kwargs["path"], "r") as fp:
                    value = fp[f"{self.kwargs['key']}/{instance.source_pk}/{instance.telescope}"][:]
                                
                    if self.transform is not None:
                        value = self.transform(value, instance)
                                    
                    instance.__pixel_data__.setdefault(self.name, value)                 
            finally:
                return instance.__pixel_data__[self.name]

        return self.field



class BaseApogeeMADGICSSpectrum(BaseModel, SpectrumMixin):

    pk = AutoField()
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
    )
    drp_spectrum_pk = ForeignKeyField(
        ApogeeVisitSpectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
        field=ApogeeVisitSpectrum.spectrum_pk,
        help_text=Glossary.drp_spectrum_pk
    )    
    source = ForeignKeyField(
        Source, 
        null=True, 
        index=True,
        column_name="source_pk",
        backref="apogee_madgics_visit_spectra",
    )

    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
    )

    release = TextField()
    telescope = TextField()
    field = TextField()
    plate = IntegerField()
    mjd = IntegerField()
    fiber = IntegerField()

    
    # All of these are 1-indexed
    map2visit = IntegerField()
    map2star = IntegerField()
    map2madgics = IntegerField()
    
    rv_pixels = FloatField()
    rv_velocity = FloatField()
    
    # Store things that are recorded by apMADGICS, in case we decide not to link them ourselves, or to check for discrepancies.
    meta_apogee_id = TextField(null=True)
    meta_ra = FloatField(null=True)
    meta_dec = FloatField(null=True)
    meta_glat = FloatField(null=True)
    meta_glon = FloatField(null=True)
    meta_sfd = FloatField(null=True)
    meta_dr17_teff = FloatField(null=True)
    meta_dr17_logg = FloatField(null=True)
    meta_dr17_x_h = FloatField(null=True)    
    meta_dr17_vsini = FloatField(null=True)
    meta_drp_snr = FloatField(null=True)
    meta_drp_vhelio = FloatField(null=True)
    meta_drp_vrel = FloatField(null=True)
    meta_drp_vrelerr = FloatField(null=True)
    meta_gaiaedr3_parallax = FloatField(null=True)
    meta_gaiaedr3_source_id = BigIntegerField(null=True)
            
            
    @property
    def path(self):
        return ""
        
    class Meta:
        indexes = (
            (
                (
                    "release",
                    "telescope",
                    "field",
                    "plate",
                    "mjd",
                    "fiber",
                ),
                True,
            ),
        )



    
class ApMADGICSTheoryStarSpectrum(BaseModel, SpectrumMixin):
    
    pk = AutoField()
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
    )
    source = ForeignKeyField(
        Source, 
        null=True, 
        index=True,
        column_name="source_pk",
        backref="apogee_madgics_th_star_spectrum",
    )

    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
    )

    release = TextField()
    telescope = TextField()
    mean_fiber = FloatField()
    
    meta_dr17_teff = FloatField(null=True)
    meta_dr17_logg = FloatField(null=True)
    meta_dr17_x_h = FloatField(null=True)    
                
            
    @property
    def path(self):
        return ""
        
    class Meta:
        indexes = (
            (
                (
                    "source_pk",
                    "release",
                    "telescope",
                ),
                True,
            ),
        )
    
    flux = PixelArray(
        accessor_class=StarPickledPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/2024_03_16_th_stack.h5",
            key="flux",
        ),
        help_text=Glossary.flux
    )
    
    ivar = PixelArray(
        accessor_class=StarPickledPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/2024_03_16_th_stack.h5",
            key="ivar",
        ),
    )    




class ApMADGICSDataDrivenStarSpectrum(BaseModel, SpectrumMixin):
    
    pk = AutoField()
    
    #> Identifiers
    spectrum_pk = ForeignKeyField(
        Spectrum,
        null=True,
        index=True,
        unique=True,
        lazy_load=False,
    )
    source = ForeignKeyField(
        Source, 
        null=True, 
        index=True,
        column_name="source_pk",
        backref="apogee_madgics_dd_star_spectrum",
    )

    wavelength = PixelArray(
        accessor_class=LogLambdaArrayAccessor,
        accessor_kwargs=dict(
            crval=4.179,
            cdelt=6e-6,
            naxis=8575,
        ),
        help_text=Glossary.wavelength
    )

    release = TextField()
    telescope = TextField()
    mean_fiber = FloatField()
    
    meta_dr17_teff = FloatField(null=True)
    meta_dr17_logg = FloatField(null=True)
    meta_dr17_x_h = FloatField(null=True)    
            
    @property
    def path(self):
        return ""
        
    class Meta:
        indexes = (
            (
                (
                    "source_pk",
                    "release",
                    "telescope",
                ),
                True,
            ),
        )
    
    flux = PixelArray(
        accessor_class=StarPickledPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/2024_03_16_dd_stack.h5",
            key="flux",
        ),
        help_text=Glossary.flux
    )
    
    ivar = PixelArray(
        accessor_class=StarPickledPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/2024_03_16_dd_stack.h5",
            key="ivar",
        ),
    )    



class ApMADGICSTheorySpectrum(BaseApogeeMADGICSSpectrum):
            
    filetype = TextField(default="apmadgics_th")
    
    # Flux accessor needs to load the file and access from map2madgics (after accounting for 1-indexing) and do 1-starLines
    flux = PixelArray(
        accessor_class=MadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2024_03_16/outdir_wu_th/apMADGICS_out_x_starLines_restFrame_v0.h5",
            key="x_starLines_restFrame_v0",
        ),
        transform=lambda v, *_: 1 + v[125:], # clip off the first 125 pixels to match the apStar/FERRE sampling
        help_text=Glossary.flux
    )
    
    def transform(v, i):
        z = np.zeros(8575)
        y = v[(125 + int(np.round(i.rv_pixels))):]**-2
        z[:min(y.size, 8575)] = y[:8575]
        return z
        
    ivar = PixelArray(
        accessor_class=MadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2024_03_16/outdir_wu_th/apMADGICS_out_x_starLines_err_v0.h5",
            key="x_starLines_err_v0"
        ),
        transform=transform
    )


class ApMADGICSDataDrivenSpectrum(BaseApogeeMADGICSSpectrum):
    
    filetype = TextField(default="apmadgics_dd")
    
    # Flux accessor needs to load the file and access from map2madgics (after accounting for 1-indexing) and do 1-starLines
    flux = PixelArray(
        accessor_class=MadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2024_03_16/outdir_wu_dd/apMADGICS_out_x_starLines_restFrame_v0.h5",
            key="x_starLines_restFrame_v0",
        ),
        transform=lambda v, *_: 1 + v[125:], # clip off the first 125 pixels to match the apStar/FERRE sampling
        help_text=Glossary.flux
    )
    
    def transform(v, i):
        z = np.zeros(8575)
        y = v[(125 + int(np.round(i.rv_pixels))):]**-2
        z[:min(y.size, 8575)] = y[:8575]
        return z
        
    ivar = PixelArray(
        accessor_class=MadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2024_03_16/outdir_wu_dd/apMADGICS_out_x_starLines_err_v0.h5",
            key="x_starLines_err_v0"
        ),
        transform=transform
    )    
    
class ApMADGICSDataDrivenVisitSpectrum(BaseApogeeMADGICSSpectrum):
    
    filetype = TextField(default="apmadgics_dd_apVisit")    

    flux = PixelArray(
        accessor_class=ResampledMadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/20240402_apMADGICS/2024_03_16_dd_apVisit_v0_resampled.h5",
            key="flux",
        ),
        help_text=Glossary.flux
    )
    ivar = PixelArray(
        accessor_class=ResampledMadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/20240402_apMADGICS/2024_03_16_dd_apVisit_v0_resampled.h5",
            key="ivar",
        ),
        help_text=Glossary.ivar
    )

    

    
class ApMADGICSTheoryVisitSpectrum(BaseApogeeMADGICSSpectrum):
    
    filetype = TextField(default="apmadgics_th_apVisit")    

    # Flux accessor needs to load the file and access from map2madgics (after accounting for 1-indexing) and do 1-starLines
    flux = PixelArray(
        accessor_class=ResampledMadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/20240402_apMADGICS/2024_03_16_th_apVisit_v0_resampled.h5",
            key="flux",
        ),
        help_text=Glossary.flux
    )
    ivar = PixelArray(
        accessor_class=ResampledMadgicsPixelArrayAccessor,
        accessor_kwargs=dict(
            path="/uufs/chpc.utah.edu/common/home/u6020307/sas_home/20240402_apMADGICS/2024_03_16_th_apVisit_v0_resampled.h5",
            key="ivar",
        ),
        help_text=Glossary.ivar
    )

    

    


    
    
    

        
    


def resample_apmadgics(spectrum, n_res):       
    x = np.arange(spectrum.wavelength.size) 

    pixel = wave_to_pixel(x + spectrum.v_rad_pixel, x)
    (finite, ) = np.where(np.isfinite(pixel))

    ((finite_flux, finite_e_flux), ) = sincint(
        pixel[finite], n_res, [
            [spectrum.flux, 1/spectrum.ivar]
        ]
    )

    flux = np.nan * np.ones(spectrum.wavelength.size)
    e_flux = np.nan * np.ones(spectrum.wavelength.size)
    flux[finite] = finite_flux
    e_flux[finite] = finite_e_flux
    
    spectrum.flux = flux
    spectrum.ivar = e_flux**-2
    bad = ~np.isfinite(spectrum.ivar)
    spectrum.ivar[bad] = 0
    
    return None


def _shift_and_resample_to_rest_frame(index, v_rad, flux, flux_var):
    
    resampled_flux = np.zeros_like(flux)
    resampled_ivar = np.zeros_like(flux)

    # split by chips
    chip_indices = (3392, 6184)
    indices = np.hstack([0, np.repeat(chip_indices, 2), 8575]).reshape((-1, 2))
    
    for i, n_res in enumerate((5, 4.25, 3.5)):
        si, ei = indices[i]
                            
        x = np.arange(flux[si:ei].size)
        # only doing shifts in terms of pixels
        pixel = x + v_rad
        pixel[(pixel < 0) | (pixel > x[-1])] = np.nan

        (finite, ) = np.where(np.isfinite(pixel))

        ((finite_flux, finite_e_flux), ) = sincint(
            pixel[finite], n_res, [
                [flux[si:ei], flux_var[si:ei]]
            ]
        )
        
        resampled_flux[si:ei][finite] = finite_flux
        resampled_ivar[si:ei][finite] = finite_e_flux**-2

    finite_ivar = finite_e_flux**-2
    finite_ivar[~np.isfinite(finite_ivar)] = 0

    return (index, resampled_flux, resampled_ivar)


def shift_and_resample_to_rest_frame(flux_path, flux_err_path, rv_pixoff_path, n_res=4.25, max_workers=4):
    import os
    print("loading flux")
    flux_fp = h5py.File(flux_path, "r")
    print("loading err")
    flux_err_fp = h5py.File(flux_err_path, "r")
    print("loading rv")
    rv_pixoff_fp = h5py.File(rv_pixoff_path, "r")
    get_key = lambda x: os.path.basename(x).split(".")[0][14:] 

    print("parsing vrads")
    v_rads = rv_pixoff_fp[get_key(rv_pixoff_path)]
    print("parsing fluxes")
    fluxes = flux_fp[get_key(flux_path)]
    print("parsing vars")
    e_fluxes = flux_err_fp[get_key(flux_err_path)]
    print("done")

    N, P = flux_fp[get_key(flux_path)].shape
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers)

    futures = []
    for i, (v_rad, flux, e_flux) in enumerate(tqdm(zip(v_rads, fluxes, e_fluxes), total=N, desc="Submitting")):
        futures.append(
            executor.submit(_shift_and_resample_to_rest_frame, i, v_rad, 1 + flux, e_flux**2, n_res)
        )

    print("preparing flux array")
    resampled_fluxs = np.zeros((N, P), dtype=float)
    print("prepareing ivar array")
    resampled_ivars = np.zeros((N, P), dtype=float)
    print("ok")

    with tqdm(total=N, desc="Resampling in parallel") as pb:
        for future in concurrent.futures.as_completed(futures):
            index, resampled_flux, resampled_ivar = future.result()
            resampled_fluxs[index] = resampled_flux
            resampled_ivars[index] = resampled_ivar
            pb.update()
                
    return resampled_fluxs, resampled_ivars


    # save to new file

if __name__ == "__main__":

    from astropy.table import Table
    from tqdm import tqdm

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_21/outdir_wu_K/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_21/outdir_wu_K/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_21/outdir_wu_K/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230721_Konventional/"
    os.makedirs(output_dir, exist_ok=True)

    N, P = resampled_flux.shape
    for i in range(N):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230721-{i}.fits", overwrite=True)

    '''
    from astropy.table import Table
    from tqdm import tqdm

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_21/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_21/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_21/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230721/"
    os.makedirs(output_dir, exist_ok=True)

    N, P = resampled_flux.shape
    for i in range(N):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230721-{i}.fits", overwrite=True)
    '''

    '''
    from astropy.table import Table
    from tqdm import tqdm

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_20/outdir_wu_injectStellarParams/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_20/outdir_wu_injectStellarParams/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_20/outdir_wu_injectStellarParams/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230720/"
    os.makedirs(output_dir, exist_ok=True)

    N, P = resampled_flux.shape
    for i in range(N):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230720-{i}.fits", overwrite=True)
    '''

    '''
    from astropy.table import Table
    from tqdm import tqdm

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_13/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_13/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_13/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230713/"
    os.makedirs(output_dir, exist_ok=True)

    N, P = resampled_flux.shape
    for i in range(N):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230713-{i}.fits", overwrite=True)
    '''
    '''

    from astropy.table import Table
    from tqdm import tqdm

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_12/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_12/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_12/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230712/"
    os.makedirs(output_dir, exist_ok=True)

    N, P = resampled_flux.shape
    for i in range(N):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230712-{i}.fits", overwrite=True)
    '''


    '''
    from astropy.table import Table
    from tqdm import tqdm

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_06/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_06/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_07_06/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230706/"
    os.makedirs(output_dir, exist_ok=True)

    N, P = resampled_flux.shape
    for i in range(N):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230706-{i}.fits", overwrite=True)
    '''

    '''

    from astropy.table import Table
    from tqdm import tqdm

    meta = Table.read(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt",
        names=("index", "telescope", "field", "plate", "mjd", "basename", "plate_path", "fiber_index"),
        format="ascii"
    )
    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_25/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_25/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_25/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230625/"
    os.makedirs(output_dir, exist_ok=True)

    meta["fiber"] = [6] * len(meta)
    for i, row in enumerate(tqdm(meta)):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230625-dr17-{row['telescope']}-{row['field']}-{row['plate']}-{row['mjd']}-{row['fiber']:0>3.0f}.fits", overwrite=True)
    '''
        

    '''
    from astropy.table import Table
    from tqdm import tqdm

    meta = Table.read(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt",
        names=("index", "telescope", "field", "plate", "mjd", "basename", "plate_path", "fiber_index"),
        format="ascii"
    )
    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_20/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_20/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_20/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230620/"
    os.makedirs(output_dir, exist_ok=True)

    meta["fiber"] = [6] * len(meta)
    for i, row in enumerate(tqdm(meta)):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230620-dr17-{row['telescope']}-{row['field']}-{row['plate']}-{row['mjd']}-{row['fiber']:0>3.0f}.fits", overwrite=True)
    '''

    '''
    from astropy.table import Table
    from tqdm import tqdm

    meta = Table.read(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt",
        names=("index", "telescope", "field", "plate", "mjd", "basename", "plate_path", "fiber_index"),
        format="ascii"
    )
    
    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230618_stddev/"
    os.makedirs(output_dir, exist_ok=True)

    meta["fiber"] = [6] * len(meta)
    for i, row in enumerate(tqdm(meta)):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-20230618-stddev-dr17-{row['telescope']}-{row['field']}-{row['plate']}-{row['mjd']}-{row['fiber']:0>3.0f}.fits", overwrite=True)
    '''


    '''
    from astropy.table import Table
    from tqdm import tqdm

    meta = Table.read(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt",
        names=("index", "telescope", "field", "plate", "mjd", "basename", "plate_path", "fiber_index"),
        format="ascii"
    )
    def shift_and_resample_to_rest_frame(flux_path, flux_var_path, rv_pixoff_path, n_res=4.25):
        import os
        flux_fp = h5py.File(flux_path, "r")
        flux_err_fp = h5py.File(flux_var_path, "r")
        rv_pixoff_fp = h5py.File(rv_pixoff_path, "r")
        get_key = lambda x: os.path.basename(x).split(".")[0][14:] 

        v_rads = rv_pixoff_fp[get_key(rv_pixoff_path)][:]
        fluxs = 1 + flux_fp[get_key(flux_path)][:]
        flux_vars = flux_err_fp[get_key(flux_var_path)][:]

        N, P = fluxs.shape
        
        executor = concurrent.futures.ProcessPoolExecutor(4)

        all_args = [(i, v_rad, flux, flux_var, n_res) for i, (v_rad, flux, flux_var) in enumerate(zip(v_rads, fluxs, flux_vars))]
        futures = [executor.submit(_shift_and_resample_to_rest_frame, *args) for args in all_args]

        resampled_fluxs = np.zeros((N, P), dtype=float)
        resampled_ivars = np.zeros((N, P), dtype=float)

        with tqdm(total=N, desc="Resampling in parallel") as pb:
            for future in concurrent.futures.as_completed(futures):
                index, resampled_flux, resampled_ivar = future.result()
                resampled_fluxs[index] = resampled_flux
                resampled_ivars[index] = resampled_ivar
                pb.update()
                
        
        return resampled_fluxs, resampled_ivars


    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu/apMADGICS_out_fluxerr2.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu/apMADGICS_out_RV_pixoff_final.h5"
    )    

    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    import os
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_20230618_fluxvar/"
    os.makedirs(output_dir, exist_ok=True)

    meta["fiber"] = [6] * len(meta)
    for i, row in enumerate(tqdm(meta)):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/aapMADGICS-20230618-fluxvar-dr17-{row['telescope']}-{row['field']}-{row['plate']}-{row['mjd']}-{row['fiber']:0>3.0f}.fits", overwrite=True)
    
    '''


    '''
    from astropy.table import Table
    from tqdm import tqdm

    meta = Table.read(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt",
        names=("index", "telescope", "field", "plate", "mjd", "basename", "plate_path", "fiber_index"),
        format="ascii"
    )

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_04/outdir_wu_50_295only/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_04/outdir_wu_50_295only/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_04/outdir_wu_50_295only/apMADGICS_out_RV_pixoff_final.h5"
    )
    
    # Clip off the first 125 pixels in order to match the apStar/FERRE sampling
    si = 125
    resampled_flux = resampled_flux[:, si:]
    resampled_ivar = resampled_ivar[:, si:]

    import os
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS_2023_06_04_wu_50_indexbugfixed/"
    os.makedirs(output_dir, exist_ok=True)

    meta["fiber"] = [6] * len(meta)
    for i, row in enumerate(tqdm(meta)):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-2023_06_04-wu_50-dr17-{row['telescope']}-{row['field']}-{row['plate']}-{row['mjd']}-{row['fiber']:0>3.0f}.fits", overwrite=True)
    '''


    '''
    from astropy.table import Table
    from tqdm import tqdm

    meta = Table.read(
        "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt",
        names=("index", "telescope", "field", "plate", "mjd", "basename", "plate_path", "fiber_index"),
        format="ascii"
    )

    dir = "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_18/outdir_wu"
    

    def shift_and_resample_to_rest_frame(flux_path, flux_err_path, rv_pixoff_path, n_res=4.25):
        
        flux_fp = h5py.File(flux_path, "r")
        flux_err_fp = h5py.File(flux_err_path, "r")
        rv_pixoff_fp = h5py.File(rv_pixoff_path, "r")

        v_rads = rv_pixoff_fp["RV_pixoff_final"][:]
        fluxs = 1 + flux_fp["x_starLines_v0"][:, 125:]
        flux_vars = flux_err_fp["apMADGICS_out_fluxerr2"][:, 25:]**2

        raise a
        N, P = fluxs.shape
        


        executor = concurrent.futures.ProcessPoolExecutor(4)

        all_args = [(i, v_rad, flux, flux_var, n_res) for i, (v_rad, flux, flux_var) in enumerate(zip(v_rads, fluxs, flux_vars))]
        futures = [executor.submit(_shift_and_resample_to_rest_frame, *args) for args in all_args]

        resampled_fluxs = np.zeros((N, P), dtype=float)
        resampled_ivars = np.zeros((N, P), dtype=float)

        with tqdm(total=N, desc="Resampling in parallel") as pb:
            for future in concurrent.futures.as_completed(futures):
                index, resampled_flux, resampled_ivar = future.result()
                resampled_fluxs[index] = resampled_flux
                resampled_ivars[index] = resampled_ivar
                pb.update()
                
        
        return resampled_fluxs, resampled_ivars
        

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        f"{dir}/apMADGICS_out_x_starLines_v0.h5",
        f"{dir}/apMADGICS_out_fluxerr2.h5",
        f"{dir}/apMADGICS_out_RV_pixoff_final.h5"
    )
    

    import os
    from astropy.io import fits
    output_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6020307/apMADGICS/"
    os.makedirs(output_dir, exist_ok=True)

    meta["fiber"] = [6] * len(meta)
    for i, row in enumerate(tqdm(meta)):
        Table(data=dict(flux=resampled_flux[i], ivar=resampled_ivar[i])).write(f"{output_dir}/apMADGICS-2023_06_04-wu_50-dr17-{row['telescope']}-{row['field']}-{row['plate']}-{row['mjd']}-{row['fiber']:0>3.0f}.fits", overwrite=True)
    '''