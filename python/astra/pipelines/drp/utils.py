import numpy as np
from astropy.io import fits
from peewee import (
    TextField,
    FloatField,
    BooleanField,
    IntegerField,
    AutoField,
    BigIntegerField,
    ForeignKeyField,
    DateTimeField,
    JOIN
)
from astra.utils import log
from astra.models.fields import BitField, BasePixelArrayAccessor

from typing import Union, List, Tuple

BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)


def log_lambda_dispersion(crval, cdelt, num_pixels):
    return 10 ** (crval + cdelt * np.arange(num_pixels))

def calculate_snr(flux, flux_error, axis=None):
    snr_pixel = np.clip(flux, 0, np.inf) / np.clip(flux_error, 0, np.inf)
    bad_pixels = ~np.isfinite(snr_pixel) | (flux_error == 0)
    snr_pixel[bad_pixels] = 0
    return np.mean(snr_pixel, axis=axis)


def get_observatory(telescope):
    return telescope.upper()[:3]

def wavelength_cards(
    crval: Union[int, float], 
    cdelt: Union[int, float], 
    num_pixels: int, 
    decimals: int = 6, 
    **kwargs
) -> List:
    return [
        BLANK_CARD,
        (" ", "WAVELENGTH INFORMATION (VACUUM)", None),
        ("CRVAL", np.round(crval, decimals), None),
        ("CDELT", np.round(cdelt, decimals), None),
        ("CTYPE", "LOG-LINEAR", None),
        ("CUNIT", "Angstrom (Vacuum)", None),
        ("CRPIX", 1, None),
        ("DC-FLAG", 1, None),
        ("NPIXELS", num_pixels, "Number of pixels per spectrum"),
    ]


def spectrum_sampling_cards(
    num_pixels_per_resolution_element: Union[int, float],
    median_filter_size: Union[int, float],
    gaussian_filter_size: Union[int, float],
    scale_by_pseudo_continuum: bool,
    **kwargs,
) -> List:
    if isinstance(num_pixels_per_resolution_element, (float, int)):
        nres = f"{num_pixels_per_resolution_element}"
    else:
        nres = " ".join(list(map(str, num_pixels_per_resolution_element)))
    return [
        BLANK_CARD,
        (" ", "SPECTRUM SAMPLING AND STACKING"),
        ("NRES", nres),
        ("FILTSIZE", median_filter_size),
        ("NORMSIZE", gaussian_filter_size),
        ("CONSCALE", scale_by_pseudo_continuum),
    ]    

def remove_filler_card(hdu):
    if FILLER_CARD_KEY is not None:
        try:
            del hdu.header[FILLER_CARD_KEY]
        except:
            None    


def _get_extname(instrument, observatory):
    return f"{instrument}/{observatory}"


def get_extname(spectrum, data_product):
    raise a
    if data_product.filetype in ("specLite", "specFull"):
        observatory, instrument = ("APO", "BOSS")
    elif data_product.filetype in ("apStar", "apStar-1m", "apVisit"):
        instrument = "APOGEE"
        if data_product.kwargs["telescope"] == "lco25m":
            observatory = "LCO"
        else:
            observatory = "APO"
    elif data_product.filetype in ("mwmVisit", "mwmStar"):
        observatory, instrument = (spectrum.meta["OBSRVTRY"], spectrum.meta["INSTRMNT"])
    else:
        # Could be a `full` filetype. Let's just try:
        try:
            observatory, instrument = (spectrum.meta["OBSRVTRY"], spectrum.meta["INSTRMNT"])
        except:
            raise ValueError(f"Cannot get extension name for file {data_product}")
    return _get_extname(instrument, observatory)



def metadata_cards(observatory: str, instrument: str) -> List:
    return [
        BLANK_CARD,
        (" ", "METADATA"),
        ("EXTNAME", _get_extname(instrument, observatory)),
        ("OBSRVTRY", observatory),
        ("INSTRMNT", instrument),
    ]


def create_empty_hdu(observatory: str, instrument: str, is_data=True) -> fits.BinTableHDU:
    """
    Create an empty HDU to use as a filler.
    """

    x = "data" if is_data else "results"
    y = "source" if is_data else "pipeline"

    cards = metadata_cards(observatory, instrument)
    cards.extend(
        [
            BLANK_CARD,
            (
                "COMMENT",
                f"No {instrument} {x} available from {observatory} for this {y}.",
            ),
        ]
    )
    return fits.BinTableHDU(
        header=fits.Header(cards),
    )



def add_category_headers(hdu, models, original_names, upper):
    category_headers_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for header, field_name in model.field_category_headers:
            if field_name in category_headers_added:
                continue
            index = 1 + list_original_names.index(field_name)
            key = f"TTYPE{index}"
            hdu.header.insert(key, BLANK_CARD)
            hdu.header.insert(key, (" ", header.upper() if upper else header))
            category_headers_added.append(field_name)
    
    return None


def _fits_array_kwargs(v):
    v = np.array(v)
    V, P = v.shape
    format = "E" # 32 bit floats
    kwds = dict()
    if v.ndim == 2:
        kwds["format"] = f"{P:.0f}{format}"
        kwds["dim"] = f"({P})"
    else:
        kwds["format"] = format
    return kwds
    



# TODO: move to somewhere else
def fits_column_kwargs(field, values, upper, warn_comment_length=47, warn_total_length=65):
    mappings = {
        # Require at least one character for text fields
        TextField: lambda v: dict(format="A{}".format(max(1, max(len(_) for _ in v)) if len(v) > 0 else 1)),
        BooleanField: lambda v: dict(format="L"),
        IntegerField: lambda v: dict(format="J"),
        FloatField: lambda v: dict(format="E"), # single precision
        AutoField: lambda v: dict(format="K"),
        BigIntegerField: lambda v: dict(format="K"),
        # We are assuming here that all foreign key fields are big integers
        ForeignKeyField: lambda v: dict(format="K"),
        BitField: lambda v: dict(format="J"), # integer
        DateTimeField: lambda v: dict(format="A26")
    }
    if isinstance(field, BasePixelArrayAccessor):
        callable = _fits_array_kwargs
    else:
        callable = mappings[type(field)]

    if isinstance(field, DateTimeField):
        array = []
        for value in values:
            try:
                array.append(value.isoformat())
            except:
                array.append(value)
    else:
        array = values

    kwds = dict(
        name=field.name.upper() if upper else field.name,
        array=array,
        unit=None,
    )
    kwds.update(callable(values))
    return kwds


def warn_on_long_name_or_comment(field, warn_comment_length=47, warn_total_length=65):
    total = len(field.name)
    if field.help_text is not None:
        if len(field.help_text) > warn_comment_length:
            log.warning(f"Field {field} help text is too long for FITS header ({len(field.help_text)} > {warn_comment_length}).")
        total += len(field.help_text)
    if total > warn_total_length:
        log.warning(f"Field {field} name and help text are too long for FITS header ({total} > {warn_total_length}).")
    return None


def get_fill_value(field, given_fill_values):
    try:
        return given_fill_values[field.name]
    except:
        try:
            if field.default is not None:
                return field.default
        except:
            None
        finally:
            if isinstance(field, BasePixelArrayAccessor):
                return np.nan

            default_fill_values = {
                TextField: "",
                BooleanField: False,
                IntegerField: -1,
                AutoField: -1,
                BigIntegerField: -1,
                FloatField: np.nan,
                ForeignKeyField: -1,
                DateTimeField: "",
                BitField: 0            
            }
            return default_fill_values[type(field)]
        