

from astra.pipelines.drp import utils
from astropy.io import fits

def create_visit_hdu(
    spectra,
    crval,
    cdelt,
    num_pixels,
    num_pixels_per_resolution_element,   
    median_filter_size: int = 501,
    median_filter_mode: str = "reflect",
    gaussian_filter_size: int = 100,
    scale_by_pseudo_continuum: bool = False,
    instrument = None,
    fill_values = None,
    upper=True,
):
    """
    Create a HDU of visit spectra sampled on a common rest frame.

    :param spectra:
        A query that will retrieve spectra to store in this HDU.
    
    :param crval:

    :param cdelt:

    :param num_pixels:

    :param num_pixels_per_resolution_element:

    :param median_filter_size:

    :param median_filter_mode:

    :param gaussian_filter_size:

    :param scale_by_pseudo_continuum:


    """
    if instrument is None:
        possible_instruments = ("APOGEE", "BOSS")
        for instrument in possible_instruments:
            if spectra.model._meta.table_name.contains(instrument.lower()):
                break
        else:
            raise ValueError("No `instrument` given, and we failed to infer it!")
    
    # Find where the first pixel array is defined, and lump them all together
    fields, pixel_fields = ({}, {})            
    for name, field in spectra.model._meta.fields.items():
        if name not in fields: # Don't duplicate fields
            fields[name] = field

    for name, field in spectra.model._meta.pixel_fields.items():
        fields[name] = field
        pixel_fields[name] = field

    sampling_kwargs = dict(
        num_pixels_per_resolution_element=num_pixels_per_resolution_element,
        median_filter_size=median_filter_size,
        median_filter_mode=median_filter_mode,
        gaussian_filter_size=gaussian_filter_size,
        scale_by_pseudo_continuum=scale_by_pseudo_continuum
    )

    # TODO: 
    observatory = utils.get_observatory(spectra[0].telescope)

    new_wavelength = utils.log_lambda_dispersion(crval, cdelt, num_pixels)
    wavelength_cards = utils.wavelength_cards(crval=crval, cdelt=cdelt, num_pixels=num_pixels)
    spectrum_sampling_cards = utils.spectrum_sampling_cards(**sampling_kwargs)

    observatory_cards = utils.metadata_cards(observatory, instrument)

    cards = [
        *observatory_cards,
        *spectrum_sampling_cards,
        *wavelength_cards,
        utils.FILLER_CARD
    ]

    # Re-sample all the data
    column_fill_values = { 
        field.name: utils.get_fill_value(field, fill_values) \
            for field in fields.values() 
    }

    data = { name: [] for name in fields.keys() }
    for spectrum in spectra:
        for name, value in spectrum.__data__.items():
            if name not in data:
                continue
            if value is None:
                value = column_fill_values[name]                    
            data[name].append(value)
        
        for name in pixel_fields.keys():
            data[name].append(getattr(spectrum, name).flatten()) # TODO: don't flatten

    # This is the point where we should resample.
    

    # Create the columns.
    original_names, columns = ({}, [])
    for name, field in fields.items():
        kwds = utils.fits_column_kwargs(field, data[name], upper=upper)
        # Keep track of field-to-HDU names so that we can add help text.
        original_names[kwds['name']] = name
        columns.append(fits.Column(**kwds))

    # Create the HDU.
    header = fits.Header(cards)
    hdu = fits.BinTableHDU.from_columns(columns, header=header)

    # Add comments for 
    for i, name in enumerate(hdu.data.dtype.names, start=1):
        field = fields[original_names[name]]
        hdu.header.comments[f"TTYPE{i}"] = field.help_text

    # Add category groupings.
    utils.add_category_headers(hdu, (spectra.model, ), original_names, upper)

    # TODO: Add comments for flag definitions?
    
    # Add checksums.
    hdu.add_checksum()
    hdu.header.insert("CHECKSUM", utils.BLANK_CARD)
    hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
    hdu.add_checksum()    

    raise a


if __name__ == "__main__":



    from astra.models.source import Source
    source = Source.get(1000)


    create_visit_hdu(
        source.apogee_visit_spectra,
        4.179,
        6e-6,
        8575,
        (5, 4.25, 3)
    )