
class ApogeeMADGICSSpectrum(BaseModel, Spectrum):

    """An APOGEE spectrum from the MADGICS pipeline."""

    source = ForeignKeyField(Source, index=True, backref="apogee_visit_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

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

    