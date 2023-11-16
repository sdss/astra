import os
import datetime
import numpy as np
from peewee import (
    Field,
    Model,
    PostgresqlDatabase,
    TextField,
    FloatField,
    BooleanField,
    IntegerField,
    AutoField,
    BigIntegerField,
    ForeignKeyField,
    DateTimeField,
    FieldAccessor,
    BigBitField,
    JOIN
)
try:
    from playhouse.postgres_ext import ArrayField
except:
    ArrayField = ...

from collections import OrderedDict

from astropy.io import fits
from tqdm import tqdm
from astra import __version__
from astra.utils import log, flatten, expand_path
from astra import models as astra_models
from astra.models.fields import BitField, BasePixelArrayAccessor
from typing import Union
from astra.glossary import Glossary

DATETIME_FMT = "%y-%m-%d %H:%M:%S"
BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)

INSTRUMENT_COMMON_DISPERSION_VALUES = {
    "apogee": (4.179, 6e-6, 8575),
    "boss": (3.5523, 1e-4, 4648)
}
INCLUDE_SOURCE_FIELD_NAMES = (
    ("sdss_id", ),
    ("sdss4_apogee_id", "apogeeid"),
    ("gaia_dr2_source_id", "gaia2_id"),
    ("gaia_dr3_source_id", "gaia3_id"),
    ("tic_v8_id", "tic_id"),
    ("healpix", ),
    
    ("carton_0", ),
    ("lead", ),
    ("version_id", "ver_id"),
    ("catalogid", "cat_id"),
    ("catalogid21", "cat_id21"),
    ("catalogid25", "cat_id25"),
    ("catalogid31", "cat_id31"),
    ("n_associated", "n_assoc"),
    ("n_neighborhood", "n_neigh"),
    
    ("ra", ),
    ("dec", ),
    ("l", ),
    ("b", ),
    ("plx", ),
    ("e_plx", ),
    ("pmra", ),
    ("e_pmra", ),
    ("pmde", ),
    ("e_pmde", ),
    ("gaia_v_rad", "v_rad"),
    ("gaia_e_v_rad", "e_v_rad"),

    ("g_mag", ),
    ("bp_mag", ),
    ("rp_mag", ),

    ("j_mag", ),
    ("e_j_mag", ),
    ("h_mag", ),
    ("e_h_mag", ),
    ("k_mag", ),
    ("e_k_mag", ),
    ("ph_qual", ),
    ("bl_flg", ),
    ("cc_flg", ),

    ("w1_mag", ),
    ("e_w1_mag", ),
    ("w1_flux", ),
    ("w1_dflux", ),
    ("w1_frac", ),
    ("w2_mag", ),
    ("e_w2_mag", ),
    ("w2_flux", ),
    ("w2_dflux", ),
    ("w2_frac", ),
    ("w1uflags", ),
    ("w2uflags", ),
    ("w1aflags", ),
    ("w2aflags", ),

    ("mag4_5", ),
    ("d4_5m", ),
    ("rms_f4_5", ),
    ("sqf_4_5", ),
    ("mf4_5", ),
    ("csf", ),

    ("n_boss_visits", "n_boss"),
    ("boss_min_mjd", "b_minmjd"),
    ("boss_max_mjd", "n_maxmjd"),
    ("n_apogee_visits", "n_apogee"),
    ("apogee_min_mjd", "a_minmjd"),
    ("apogee_max_mjd", "a_maxmjd"),
)


def create_source_primary_hdu(
    source, 
    field_names=INCLUDE_SOURCE_FIELD_NAMES,
    upper=False
):
    """
    Create a primary (header only) HDU with basic source-level information only.
    
    The information to include are identifiers, astrometry, and photometry.
    
    Note that the names for some fields are shortened so that they are not prefixed by the FITS 'HIERARCH' cards.
    """

    cards, original_names = create_source_primary_hdu_cards(source, field_names, upper)
    return create_source_primary_hdu_from_cards(source, cards, original_names, upper)
    



def create_source_primary_hdu_from_cards(source, cards, original_names, upper=False):

    hdu = fits.PrimaryHDU(header=fits.Header(cards=cards))
    
    # Add category groupings.
    add_category_headers(hdu, (source.__class__, ), original_names, upper, use_ttype=False)
    add_category_comments(hdu, (source.__class__, ), original_names, upper, use_ttype=False)

    s = lambda v: v.upper() if upper else v

    # Add checksums.
    hdu.add_checksum()
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.header.insert("CHECKSUM", (" ", s("Data Integrity")))
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.add_checksum()
    return hdu
    

def create_source_primary_hdu_cards(
    source,
    field_names=INCLUDE_SOURCE_FIELD_NAMES,
    upper=False,
    context="results"
):
    
    created = datetime.datetime.utcnow().strftime(DATETIME_FMT)

    shortened_names = { k[0]: k[0] if len(k) == 1 else k[1] for k in field_names }
    s = lambda v: v.upper() if upper else v

    cards = [
        BLANK_CARD,
        (" ", s("Metadata"), None),
        BLANK_CARD,
        (s("v_astra"), __version__, Glossary.v_astra),
        (s("created"), created, f"File creation time (UTC {DATETIME_FMT})"),
    ]
    original_names = {}
    for name, field in source._meta.fields.items():
        try:
            use_name = shortened_names[name]
        except:
            continue
        else:                        
            use_name = s(use_name)
            
            value = getattr(source, name)
            if value is not None and isinstance(value, float) and not np.isfinite(value):
                # FITS header cards can't have numpy NaNs
                value = 'NaN'
            cards.append((use_name, value, field.help_text))
            original_names[use_name] = name

        warn_on_long_name_or_comment(field)

    cards.extend([
        BLANK_CARD,
        (" ", s("HDU Descriptions"), None),
        BLANK_CARD,
        ("COMMENT", "HDU 0: Summary information only", None),
        ("COMMENT", f"HDU 1: BOSS {context} from Apache Point Observatory"),
        ("COMMENT", f"HDU 2: BOSS {context} from Las Campanas Observatory"),
        ("COMMENT", f"HDU 3: APOGEE {context} from Apache Point Observatory"),
        ("COMMENT", f"HDU 4: APOGEE {context} from Las Campanas Observatory"),
    ])
    
    return (cards, original_names)
    

def resolve_model(model_or_model_name):
    if isinstance(model_or_model_name, str):
        return getattr(astra_models, model_or_model_name)
    else:
        return model_or_model_name

def get_fields_and_pixel_arrays(models, name_conflict_strategy=None, ignore_field_names=None):
    
    if name_conflict_strategy is None:
        def default_conflict_strategy(fields, name, field, model):
            # Overwrite the previous field, since the hierarchy of models is usually
            # Source -> spectrum_level -> pipeline_level
            del fields[name]
            fields[name] = field
            return None

        name_conflict_strategy = default_conflict_strategy

    fields = OrderedDict([])
    for model in models:
        for name, field in model.__dict__.items():
            if (
                (ignore_field_names is not None and name in ignore_field_names)
            or  (not isinstance(field, (FieldAccessor, BasePixelArrayAccessor)))
            ):
                continue

            if isinstance(field, FieldAccessor):
                field = model._meta.fields.get(name, field)

            if name in fields:
                name_conflict_strategy(fields, name, field, model)                
            else:
                fields[name] = field
            
            warn_on_long_name_or_comment(field)
    
    return fields


def get_fields(models, name_conflict_strategy=None, ignore_field_name_callable=None):
    fields = OrderedDict([])
    for model in models:
        for name, field in model._meta.fields.items():
            if ignore_field_name_callable is not None and ignore_field_name_callable(name):
                continue

            if name in fields:
                if name_conflict_strategy is None:
                    continue
                
                # Execute the name conflict strategy, which might re-name fields, etc
                name_conflict_strategy(fields, name, field, model)
                # TODO: warn on the new name/comment.
            else:
                fields[name] = field        
                warn_on_long_name_or_comment(field)

    return fields




def get_binary_table_hdu(
    q,
    models,
    fields,
    limit=None,
    header=None,
    upper=True,
    fill_values=None,
):
    column_fill_values = {
        "sdss5_target_flags": np.zeros((0, 0), dtype=np.uint8),
    }
    
    for name, field in fields.items():
        try:
            column_fill_values[name] = get_fill_value(field, fill_values)
        except KeyError:
            # May not be a problem yet.
            continue

    data = { name: [] for name in fields.keys() }
    for result in tqdm(q.iterator(), desc="Collecting results", total=limit or q.count()):
        
        if isinstance(result, dict):        
            for name, value in result.items():
                if value is None:
                    value = column_fill_values[name]                    
                data[name].append(value)
        else:
            for name, field in fields.items():
                value = getattr(result, name)
                if value is None:
                    value = get_fill_value(field, None)
                data[name].append(value)
    
    # Create the columns.
    original_names, columns, = ({}, [])
    for name, field in fields.items():
        if isinstance(field, ArrayField):
            # Do a hack to deal with delta_ra, delta_dec
            if len(data[name]) > 0:
                P = max(map(len, data[name]))
                # TODO: we are assuming floats here for this ArrayField
                assert field.field_type == "FLOAT"
                value = np.nan * np.ones((len(data[name]), P), dtype=np.float32)
                for i, item in enumerate(data[name]):
                    value[i, :len(item)] = item
            else:
                value = np.ones((0, 0), dtype=np.float32)
        else:
            value = data[name]
        kwds = fits_column_kwargs(field, value, upper=upper, name=name)
        # Keep track of field-to-HDU names so that we can add help text.
        original_names[kwds['name']] = name
        columns.append(fits.Column(**kwds))    

    hdu = fits.BinTableHDU.from_columns(columns, header=header)

    # Add comments for each field.
    for i, name in enumerate(hdu.data.dtype.names, start=1):
        field = fields[original_names[name]]
        hdu.header.comments[f"TTYPE{i}"] = field.help_text

    # Add category groupings.
    add_category_headers(hdu, models, original_names, upper)
    add_category_comments(hdu, models, original_names, upper)

    # TODO: Add comments for flag definitions?
    
    # Add checksums.
    hdu.add_checksum()
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.add_checksum()
    return hdu


def add_category_comments(hdu, models, original_names, upper, use_ttype=True):
    category_comments_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for comment, field_name in model.category_comments:
            if field_name in category_comments_added:
                continue
            try:
                index = 1 + list_original_names.index(field_name)
            except:
                continue
            if use_ttype:
                key = f"TFORM{index}"
            else:
                key = [k for k, v in original_names.items() if v == field_name][0]
            hdu.header.insert(key, ("COMMENT", comment), after=True)
            category_comments_added.append(field_name)
    return None

def add_category_headers(hdu, models, original_names, upper, use_ttype=True, suppress_warnings=False):
    category_headers_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for header, field_name in model.category_headers:
            if field_name in category_headers_added:
                continue
            try:
                index = 1 + list_original_names.index(field_name)
            except:
                if not suppress_warnings:
                    log.warning(f"Cannot find field {field_name} to put category header above it")
                continue
            if use_ttype:
                key = f"TTYPE{index}"
            else:
                key = [k for k, v in original_names.items() if v == field_name][0]
            hdu.header.insert(key, BLANK_CARD)
            hdu.header.insert(key, (" ", header.upper() if upper else header))
            hdu.header.insert(key, BLANK_CARD)
            category_headers_added.append(field_name)
    
    return None


def fits_column_kwargs(field, values, upper, name=None, default_n_pixels=0, warn_comment_length=47, warn_total_length=65):
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
        DateTimeField: lambda v: dict(format="A26"),
    }
    if isinstance(field, BasePixelArrayAccessor):
        def callable(v):
            V, P = np.atleast_2d(v).shape
            if P == 0:
                P = default_n_pixels
                # TODO: it would be nice to have the expected number of pixels, but this is a lot of hack to make that happen
                #P = getattr(field, "pixels", None) or P # try to get the expected number of pixels if we have none
            return dict(format=f"{P:.0f}E", dim=f"({P})")
        
    elif isinstance(field, ArrayField):
        
        def callable(v):
            V, P = np.atleast_2d(v).shape
            format_code = "E" if field.field_type == "FLOAT" else "J"
            return dict(format=f"{P:.0f}{format_code}", dim=f"({P})")    
    
    elif isinstance(field, BigBitField):
        def callable(v):
            V, P = np.atleast_2d(v).shape
            return dict(format=f"{P:.0f}B", dim=f"({P})")
    else:
        callable = mappings[type(field)]

    if isinstance(field, DateTimeField):
        array = []
        for value in values:
            try:
                array.append(value.isoformat())
            except:
                array.append(value)
    elif isinstance(field, BigBitField):
        N = len(values)
        if N > 0:
            F = max(1, max(len(item) for item in values))
        else:
            F = 1 # otherwise this doesn't play well with the FITS standard
        
        array = np.zeros((N, F), dtype=np.uint8)
        for i, item in enumerate(values):
            array[i, :len(item)] = np.frombuffer(item.tobytes(), dtype=np.uint8)
    elif isinstance(field, ArrayField):
        if len(values) > 0:
            P = max(map(len, values))
            # TODO: we are assuming floats here for this ArrayField
            assert field.field_type == "FLOAT"
            array = np.nan * np.ones((len(values), P), dtype=np.float32)
            for i, item in enumerate(values):
                array[i, :len(item)] = item
        else:
            array = np.ones((0, 0), dtype=np.float32)                
    else:
        array = values

    if isinstance(field, BasePixelArrayAccessor):
        name = name or field.name
    else:
        name = name or field.column_name or field.name
        
    kwds = dict(
        name=name.upper() if upper else name,
        array=array,
        unit=None,
    )
    kwds.update(callable(array))
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



def wavelength_cards(
    crval: Union[int, float], 
    cdelt: Union[int, float], 
    num_pixels: int, 
    decimals: int = 6, 
    upper: bool = False,
    **kwargs
):
    h = "Wavelength Information (Vacuum)"
    h = h.upper() if upper else h
    return [
        BLANK_CARD,
        (" ", h, None),
        BLANK_CARD,
        ("CRVAL", np.round(crval, decimals), Glossary.crval),
        ("CDELT", np.round(cdelt, decimals), Glossary.cdelt),
        ("CTYPE", "LOG-LINEAR", Glossary.ctype),
        ("CUNIT", "Angstrom (Vacuum)", Glossary.cunit),
        ("CRPIX", 1, Glossary.crpix),
        ("DC-FLAG", 1, Glossary.dc_flag),
        # We can't use NAXIS because it is a reserved keyword when we use a binary table,
        # and there is just NO way around this because NAXIS[...] are used to set the table size.
        ("NPIXELS", num_pixels, Glossary.npixels),
    ]

def dispersion_array(instrument):
    crval, cdelt, num_pixels = INSTRUMENT_COMMON_DISPERSION_VALUES[instrument.lower().strip()]
    return 10**(crval + cdelt * np.arange(num_pixels))

def get_basic_header(
    pipeline=None,
    observatory=None,
    instrument=None,
    include_dispersion_cards=None,
    include_hdu_descriptions=False,
    upper=False
):
    created = datetime.datetime.utcnow().strftime(DATETIME_FMT)

    s = lambda v: v.upper() if upper else v
    cards = [
        BLANK_CARD,
        (" ", s("Metadata"), None),
        BLANK_CARD,
    ]

    if instrument is not None and observatory is not None:
        cards.append(("EXTNAME", _get_extname(instrument, observatory), "Extension name"))
    
    if observatory is not None:
        cards.append(("OBSRVTRY", observatory.upper(), "Observatory"))
        
    if instrument is not None:
        cards.append(("INSTRMNT", instrument.upper(), "Instrument"))

    if pipeline is not None:
        cards.append(("PIPELINE", pipeline, "Pipeline name"))
    
    cards.extend([
        ("V_ASTRA", __version__, Glossary.v_astra),
        ("CREATED", created, f"File creation time (UTC {DATETIME_FMT})"),
    ])

    if include_dispersion_cards:
        try:
            cards.extend(wavelength_cards(*INSTRUMENT_COMMON_DISPERSION_VALUES[instrument.lower().strip()], upper=upper))
        except KeyError:
            raise ValueError(f"Unknown instrument '{instrument}': not among {', '.join(INSTRUMENT_COMMON_DISPERSION_VALUES.keys())}")

    if include_hdu_descriptions:
        cards.extend([
            BLANK_CARD,
            (" ", s("HDU Descriptions"), None),
            BLANK_CARD,
            ("COMMENT", "HDU 0: Summary information only", None),
            ("COMMENT", "HDU 1: BOSS spectra taken at Apache Point Observatory"),
            ("COMMENT", "HDU 2: BOSS spectra taken at Las Campanas Observatory"),
            ("COMMENT", "HDU 3: APOGEE spectra taken at Apache Point Observatory"),
            ("COMMENT", "HDU 4: APOGEE spectra taken at Las Campanas Observatory"),
        ])        
    return fits.Header(cards)    





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
            if isinstance(field, ArrayField):
                if field.field_type == "FLOAT":
                    return [np.nan]
                else:
                    return [-1]
                
            default_fill_values = {
                TextField: "",
                BooleanField: False,
                IntegerField: -1,
                AutoField: -1,
                BigIntegerField: -1,
                FloatField: np.nan,
                ForeignKeyField: -1,
                DateTimeField: "",
                BitField: 0,
            }
            return default_fill_values[type(field)]
                


def _resolve_model(model_name):
    if isinstance(model_name, str):    
        return getattr(astra_models, model_name) 
    else:
        return model_name

def _get_extname(instrument, observatory):
    # parse instrument
    if instrument.strip().lower().startswith("apogee"):
        instrument_str = "APOGEE"
    elif instrument.strip().lower().startswith("boss"):
        instrument_str = "BOSS"
    else:
        raise ValueError(f"Unknown instrument '{instrument}'")
    return f"{instrument_str}/{observatory.upper().strip()}"


def check_path(path, overwrite):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise OSError(f"File {path} already exists. If you mean to replace it then use the argument \"overwrite=True\".")
