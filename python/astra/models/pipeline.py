
import numpy as np
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
from astra.models.fields import BitField

BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)

class PipelineOutputMixin:

    @classmethod
    def to_hdu(cls, spectrum_model, where=None, header=None, fill_values=None, upper=True):
        """
        Create a FITS binary HDU of pipeline results for the given spectrum model.

        :param spectrum_model:
            The spectrum model to select on (e.g., `astra.models.apogee.ApogeeVisitSpectrum`).

        :param where: [optional]
            Supply an optional `where` clause to filter the results.

        :param header: [optional]
            The base header to use for the HDU. This can contain things like the `EXTNAME`,
            `INSTRMNT`, and `OBSRVTRY`.
        
        :param fill_values: [optional]
            A `dict` where field names are keys and fill values are values.
        
        :param upper: [optional]
            If `True` (default), then all column names will be converted to upper case.
        """

        from astropy.io import fits
        from astra.models.base import Source

        fields = {}
        models = (Source, spectrum_model, cls)
        for model in models:
            for name, field in model._meta.fields.items():
                if name not in fields: # Don't duplicate fields
                    fields[name] = field
                warn_on_long_name_or_comment(field)

        # Do left outer joins on spectrum_model so that we get every spectrum even if there
        # isn't a corresponding pipeline result.
        # If the user wants something different, they can use the `where` clause.
        q = (
            spectrum_model
            .select(*tuple(fields.values()))
            .join(Source)
            .switch(spectrum_model)
            .join(
                cls, 
                JOIN.LEFT_OUTER,
                on=(cls.spectrum_id == spectrum_model.spectrum_id),
            )
            .dicts()
        )

        if where is not None:   
            q = q.where(where)

        column_fill_values = { 
            field.name: get_fill_value(field, fill_values) \
                for field in fields.values() 
        }

        data = { name: [] for name in fields.keys() }
        for result in q:
            for name, value in result.items():
                if value is None:
                    value = column_fill_values[name]                    
                data[name].append(value)
        
        # Create the columns.
        original_names, columns = ({}, [])
        for name, field in fields.items():
            kwds = fits_column_kwargs(field, data[name], upper=upper)
            # Keep track of field-to-HDU names so that we can add help text.
            original_names[kwds['name']] = name
            columns.append(fits.Column(**kwds))

        # Create the HDU.
        hdu = fits.BinTableHDU.from_columns(columns, header=header)

        # Add comments for 
        for i, name in enumerate(hdu.data.dtype.names, start=1):
            field = fields[original_names[name]]
            hdu.header.comments[f"TTYPE{i}"] = field.help_text

        # Add category groupings.
        add_category_headers(hdu, models, original_names, upper)

        # TODO: Add comments for flag definitions?
        
        # Add checksums.
        hdu.add_checksum()
        hdu.header.insert("CHECKSUM", BLANK_CARD)
        hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
        hdu.add_checksum()

        raise NotImplementedError



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


# TODO: move to somewhere else
def fits_column_kwargs(field, values, upper, warn_comment_length=47, warn_total_length=65):
    mappings = {
        # Require at least one character for text fields
        TextField: lambda v: dict(format="A{}".format(max(1, max(len(_) for _ in v)))),
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


# TODO: Put this elsewhere.
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
        