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
    JOIN
)
from collections import OrderedDict

from astropy.io import fits
from tqdm import tqdm
from astra import __version__
from astra.utils import log, flatten, expand_path
from astra import models as astra_models
from astra.models.fields import BitField, BasePixelArrayAccessor

from astra.glossary import Glossary

DATETIME_FMT = "%y-%m-%d %H:%M:%S"
BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)


def resolve_model(model_or_model_name):
    if isinstance(model_or_model_name, str):
        return getattr(astra_models, model_or_model_name)
    else:
        return model_or_model_name

def get_fields_and_pixel_arrays(models, name_conflict_strategy=None, ignore_field_names=None):
    fields = OrderedDict([])
    for model in models:
        for name, field in model.__dict__.items():
            if (
                (ignore_field_names is not None and name in ignore_field_names)
            or  (not isinstance(field, (FieldAccessor, BasePixelArrayAccessor)))
            or  (name in fields and name_conflict_strategy is None)
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


def get_fields(models, name_conflict_strategy=None, ignore_field_names=None):
    fields = OrderedDict([])
    for model in models:
        for name, field in model._meta.fields.items():
            if ignore_field_names is not None and name in ignore_field_names:
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




def get_binary_table_hdu(q, models, fields, limit=None, header=None, upper=True, fill_values=None):
    column_fill_values = {}
    for field in fields.values():
        try:
            column_fill_values[field.name] = get_fill_value(field, fill_values)
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
    original_names, columns = ({}, [])
    for name, field in fields.items():
        kwds = fits_column_kwargs(field, data[name], upper=upper)
        # Keep track of field-to-HDU names so that we can add help text.
        original_names[kwds['name']] = name
        columns.append(fits.Column(**kwds))    

    hdu = fits.BinTableHDU.from_columns(columns, header=header)

    # Add comments for
     
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

def add_category_headers(hdu, models, original_names, upper, use_ttype=True):
    category_headers_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for header, field_name in model.category_headers:
            if field_name in category_headers_added:
                continue
            try:
                index = 1 + list_original_names.index(field_name)
            except:
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
        def callable(v):
            V, P = np.atleast_2d(v).shape
            if P == 0:
                P = getattr(field, "pixels", None) or P # try to get the expected number of pixels if we have none
            return dict(format=f"{P:.0f}E", dim=f"({P})")
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


def get_basic_header(
    pipeline=None,
    observatory=None,
    instrument=None,
    include_hdu_descriptions=False
):
    created = datetime.datetime.utcnow().strftime(DATETIME_FMT)

    cards = [
        BLANK_CARD,
        (" ", "METADATA", None),
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
    if include_hdu_descriptions:
        cards.extend([
            BLANK_CARD,
            (" ", "HDU DESCRIPTIONS", None),
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
                


def _resolve_model(model_name):
    if isinstance(model_name, str):    
        return getattr(astra_models, model_name) 
    else:
        return model_name

def _get_extname(instrument, observatory):
    return f"{instrument.upper().strip()}/{observatory.upper().strip()}"


def check_path(path, overwrite):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise OSError(f"File {path} already exists. If you mean to replace it then use the argument \"overwrite=True\".")
