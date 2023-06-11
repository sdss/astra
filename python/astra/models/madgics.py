import numpy as np
from peewee import (
    ForeignKeyField,
    IntegerField,
    TextField
)

from astra.models.base import BaseModel
from astra.models.source import Source
from astra.models.spectrum import (Spectrum, SpectrumMixin)
from astra.models.fields import PixelArray, BasePixelArrayAccessor
from astra.specutils.resampling import wave_to_pixel, sincint


class PixelArrayAccessorHDF(BasePixelArrayAccessor):

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except AttributeError:
                # Load them all.
                instance.__pixel_data__ = {}
                import h5py
                with h5py.File(instance.path, "r") as fp:
                    for name, accessor in instance._meta.pixel_fields.items():
                        value = fp[accessor.column_name][instance.row_index]
                        if accessor.transform is not None:
                            value = accessor.transform(value)
                        
                        instance.__pixel_data__.setdefault(name, value)
                
                return instance.__pixel_data__[self.name]

        return self.field



class ApogeeMADGICSSpectrum(BaseModel, SpectrumMixin):

    """An APOGEE spectrum from the MADGICS pipeline."""

    # TODO: We set the referencing up like this so that we can do analysis without *requiring* the lazy loading to Source (and that source exists)
    sdss_id = ForeignKeyField(Source, lazy_load=False, index=True, backref="apogee_madgics_spectra")
    spectrum_id = ForeignKeyField(Spectrum, lazy_load=False, index=True)
    
    # TODO: replace this with something that is recognised as a field?
    @property
    def wavelength(self):
        return 10**(4.179 + 6e-6 * np.arange(8575))

    flux = PixelArray(
        column_name="x_starLines_v0",
        accessor_class=PixelArrayAccessorHDF,
        transform=lambda x: 1 + x[125:]
    )
    ivar = PixelArray(
        column_name="x_starLines_err_v0",
        accessor_class=PixelArrayAccessorHDF,
        transform=lambda x: x[125:]**-2
    )

    row_index = IntegerField(index=True)
    v_rad_pixel = PixelArray(column_name="RV_pixoff_final", accessor_class=PixelArrayAccessorHDF)

    release = TextField()
    telescope = TextField()
    field = TextField()
    plate = IntegerField()
    mjd = IntegerField()
    fiber = IntegerField()


    @property
    def path(self):
        return "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/outdir_wu/apMADGICS_out.h5"




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


def _shift_and_resample_to_rest_frame(index, v_rad, flux, flux_var, n_res=4.25):
    x = np.arange(flux.size)
    pixel = wave_to_pixel(x + v_rad, x)

    (finite, ) = np.where(np.isfinite(pixel))

    ((finite_flux, finite_e_flux), ) = sincint(
        pixel[finite], n_res, [
            [flux, flux_var]
        ]
    )

    finite_ivar = finite_e_flux**-2
    finite_ivar[~np.isfinite(finite_ivar)] = 0

    resampled_flux = np.zeros_like(flux)
    resampled_ivar = np.zeros_like(flux)
    resampled_flux[finite] = finite_flux
    resampled_ivar[finite] = finite_ivar

    return (index, resampled_flux, resampled_ivar)


def shift_and_resample_to_rest_frame(flux_path, flux_err_path, rv_pixoff_path, n_res=4.25):
    import h5py
    
    flux_fp = h5py.File(flux_path, "r")
    flux_err_fp = h5py.File(flux_err_path, "r")
    rv_pixoff_fp = h5py.File(rv_pixoff_path, "r")

    v_rads = rv_pixoff_fp["RV_pixoff_final"][:]
    fluxs = 1 + flux_fp["x_starLines_v0"][:, 125:]
    flux_vars = flux_err_fp["x_starLines_err_v0"][:, 25:]**2

    N, P = fluxs.shape
    
    import concurrent
    from tqdm import tqdm

    executor = concurrent.futures.ProcessPoolExecutor(4)

    all_args = [(i, v_rad, flux, flux_var, n_res) for i, (v_rad, flux, flux_var) in enumerate(zip(v_rads, fluxs, flux_vars))]
    futures = [executor.submit(_shift_and_resample_to_rest_frame, *args) for args in all_args]

    resampled_fluxs = np.zeros((N, P), dtype=float)
    resampled_ivars = np.zeros((N, P), dtype=float)

    with tqdm(total=N) as pb:
        for future in concurrent.futures.as_completed(futures):
            index, resampled_flux, resampled_ivar = future.result()
            resampled_fluxs[index] = resampled_flux
            resampled_ivars[index] = resampled_ivar
            pb.update()
            
    
    return resampled_fluxs, resampled_ivars

    
    # save to new file

if __name__ == "__main__":

    resampled_flux, resampled_ivar = shift_and_resample_to_rest_frame(
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_04/outdir_wu_50_295only/apMADGICS_out_x_starLines_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_04/outdir_wu_50_295only/apMADGICS_out_x_starLines_err_v0.h5",
        "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2023_06_04/outdir_wu_50_295only/apMADGICS_out_RV_pixoff_final.h5"
    )