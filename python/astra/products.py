import os
import numpy as np
import datetime
from astropy.io import fits
from tqdm import tqdm
from typing import Iterable, Union, Optional, Tuple
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
    JOIN
)
from astra import __version__
from astra import models
from astra.models import BaseModel, Source, SpectrumMixin, ApogeeVisitSpectrum
from astra.utils import log, expand_path
from astra.models.fields import BitField

from astra.glossary import Glossary


DATETIME_FMT = "%y-%m-%d %H:%M:%S"
BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)


def create_hdu(
    spectrum_model=None, 
    pipeline_model=None, 
    drp_spectrum_model=None, 
    where=None, 
    header=None, 
    ignore_fields=("carton_flags", "source", "pk"),
    fill_values=None, 
    upper=True,
    limit=None,
    **kwargs
):
    """
    Create a FITS binary HDU of database results, either from a pipeline or just from spectrum model.
     
    :param spectrum_model: [optional]
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

    pipeline_model = _resolve_model(pipeline_model)
    spectrum_model = _resolve_model(spectrum_model)
    drp_spectrum_model = _resolve_model(drp_spectrum_model)

    fields = {}
    models = [Source]
    for model in (spectrum_model, drp_spectrum_model, pipeline_model):
        if model is not None:
            models.append(model)
    
    for model in models:
        for name, field in model._meta.fields.items():
            if name not in fields and (ignore_fields is None or name not in ignore_fields): # Don't duplicate fields
                fields[name] = field
            warn_on_long_name_or_comment(field)

    # Do left outer joins on spectrum_model so that we get every spectrum even if there
    # isn't a corresponding pipeline result.
    # If the user wants something different, they can use the `where` clause.
    #distinct_on = [pipeline_model]
    #if spectrum_model is not None:
    #    distinct_on.append(spectrum_model)
    
    '''
    q = (
        pipeline_model
        .select(*tuple(fields.values()))
    )
    #    .distinct(*distinct_on)
    #)

    if pipeline_model != Source:
        q = (
            q
            .join(
                Source,
                on=(Source.id == pipeline_model.source_id)
            )
            .switch(pipeline_model)
        )

    if spectrum_model is not None:
        q = (
            q
            .join(
                spectrum_model, 
                on=(Source.id == spectrum_model.source_id)
            )
        )

    if drp_spectrum_model is not None:
        # Here we ignore the JOIN_TYPE because otherwise we'd
        q = (
            q
            .join(
                drp_spectrum_model, 
                on=(drp_spectrum_model.spectrum_id == spectrum_model.drp_spectrum_id)
            )
            .switch(spectrum_model)
        )
    
    """
    if pipeline_model != Source:                
        q = (
            q
            .join(
                pipeline_model, 
                join_type,
                on=(pipeline_model.spectrum_id == spectrum_model.spectrum_id),
            )
        )
    """
    
    if where is not None:   
        q = q.where(where)

    q = q.limit(limit).dicts()
    '''

    # TODO: reconsider if we need this switch
    if spectrum_model is None and drp_spectrum_model is None and pipeline_model is None:
        q = (
            Source
            .select(*tuple(fields.values()))
        )
    else:
        q = (
            spectrum_model 
            .select(*tuple(fields.values()))
            .join(pipeline_model, on=(pipeline_model.spectrum_id == spectrum_model.spectrum_id))
            .switch(spectrum_model)
            .join(Source, on=(Source.id == spectrum_model.source_id))
        )
        if drp_spectrum_model is not None:
            q = (
                q
                .join(drp_spectrum_model, on=(drp_spectrum_model.spectrum_id == spectrum_model.drp_spectrum_id))
            )
    
    q = (
        q
        .where(where)
        .limit(limit)
        .dicts()
    )

    column_fill_values = { 
        field.name: get_fill_value(field, fill_values) \
            for field in fields.values() 
    }

    total = limit or q.count()
    data = { name: [] for name in fields.keys() }
    for result in tqdm(q.iterator(), desc="Collecting results", total=total):
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
    add_category_comments(hdu, models, original_names, upper)

    # TODO: Add comments for flag definitions?
    
    # Add checksums.
    hdu.add_checksum()
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.add_checksum()

    return hdu


def create_summary_star_product(
    overwrite: Optional[bool] = False,
    full_output: Optional[bool] = False,
    **kwargs
):
    """
    Create a summary product containing information about all stars.

    :param overwrite: [optional]
        If the output path already exists, overwrite it.
    
    :param full_output: [optional]
        If given, return a two-length tuple containing the path, and the HDU list.

    :returns:
        The path where the product was written. If `full_output=True`, then the HDU list will also be returned.    
    """
    path = _get_summary_path(f"allStar-{__version__}.fits")
    _prevent_summary_overwrite(path, overwrite)

    hdu_list = fits.HDUList([
        fits.PrimaryHDU(),
        create_hdu(**kwargs)
    ])
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path


def create_summary_visit_product(
    overwrite: Optional[bool] = False,
    full_output: Optional[bool] = False,     
    **kwargs   
):
    """
    Create a summary product containing information about all visits.

    :param overwrite: [optional]
        If the output path already exists, overwrite it.
    
    :param full_output: [optional]
        If given, return a two-length tuple containing the path, and the HDU list.

    :returns:
        The path where the product was written. If `full_output=True`, then the HDU list will also be returned.    
    """
    path = _get_summary_path(f"allVisit-{__version__}.fits")
    _prevent_summary_overwrite(path, overwrite)

    # TODO: Revisit this when the BOSS models are solidified, and clarify apo1m placement.
    hdu_list = fits.HDUList([
        fits.PrimaryHDU(),
        create_hdu(
            spectrum_model=ApogeeVisitSpectrum,
            where=(ApogeeVisitSpectrum.telescope == "apo25m"),
            **kwargs
        ),
        create_hdu(
            spectrum_model=ApogeeVisitSpectrum,
            where=(ApogeeVisitSpectrum.telescope == "lco25m"),
            **kwargs
        ),
        create_hdu(
            spectrum_model=ApogeeVisitSpectrum,
            where=(ApogeeVisitSpectrum.telescope == "apo1m"),
            **kwargs
        ),
    ])
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path


def create_summary_pipeline_product(
    pipeline_model: Optional[Union[BaseModel, str]],
    spectrum_model: Union[SpectrumMixin, str],
    basename_prefix: str, 
    drp_spectrum_model: Optional[Union[BaseModel, str]] = None,
    overwrite: Optional[bool] = False,
    full_output: Optional[bool] = False,
    **kwargs           
):
    """
    Create a summary product containing pipeline results for some given spectral model.
     
    :param pipeline_model:
        The data model for pipeline results.
    
    :param spectrum_model:
        The data model for the spectra to cross-match with. If `None` is given, only source-
        level results will be given (irrespective of what is given to `drp_spectrum_model`
        and `pipeline_model`).

    :param basename_prefix:
        The prefix to use in the basename. For example, `prefix='allVisit'` for Astra version
        0.4.0 will translate to a basename of 'allVisit-0.4.0.fits'
        
    :param drp_spectrum_model: [optional]
        If the `pipeline_model` is a derivative spectrum product (e.g., a re-sampled spectrum)
        then it may not have all the data reduction pipeline metadata available. If you supply
        `drp_spectrum_model`, then all the metadata from the original data reduction pipeline
        product will be included.
    
    :param overwrite: [optional]
        If the output path already exists, overwrite it.
    
    :param full_output: [optional]
        If given, return a two-length tuple containing the path, and the HDU list.

    :returns:
        The path where the product was written. If `full_output=True`, then the HDU list will also be returned.    
    """
    path = _get_summary_path(f"{basename_prefix}-{__version__}.fits")
    _prevent_summary_overwrite(path, overwrite)    

    primary_hdu = create_summary_pipeline_primary_hdu(pipeline_model)

    # TODO: this is hacky, but we can revise when we have to deal with pipelines that Do It All(tm)
    is_apogee = spectrum_model.__name__.lower().startswith("apogee")
    is_boss = spectrum_model.__name__.lower().startswith("boss")
    if (not is_apogee and not is_boss) or (is_apogee and is_boss):
        raise NotImplementedError("can't figure out this")

    kwds = dict(
        spectrum_model=spectrum_model,
        pipeline_model=pipeline_model,
        drp_spectrum_model=drp_spectrum_model,
        **kwargs
    )

    if is_apogee:
        # make empty boss hdus
        hdu_boss_apo = create_empty_hdu("APO", "BOSS", False)
        hdu_boss_lco = create_empty_hdu("LCO", "BOSS", False)

        hdu_apogee_apo = create_hdu(where=spectrum_model.telescope.startswith("apo"), header=fits.Header(cards=metadata_cards("APO", "APOGEE")), **kwds)
        hdu_apogee_lco = create_hdu(where=spectrum_model.telescope.startswith("lco"), header=fits.Header(cards=metadata_cards("LCO", "APOGEE")), **kwds)

    else:
        hdu_apogee_apo = create_empty_hdu("APO", "APOGEE", False)
        hdu_apogee_lco = create_empty_hdu("LCO", "APOGEE", False)

        hdu_boss_apo = create_hdu(where=spectrum_model.telescope.startswith("apo"), header=fits.Header(cards=metadata_cards("APO", "BOSS")), **kwds)
        hdu_boss_lco = create_hdu(where=spectrum_model.telescope.startswith("lco"), header=fits.Header(cards=metadata_cards("LCO", "BOSS")), **kwds)

    hdu_list = fits.HDUList([
        primary_hdu,
        hdu_boss_apo,
        hdu_boss_lco,
        hdu_apogee_apo,
        hdu_apogee_lco,
    ])
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path


#def create_mwmvisit_mwmstar_



def metadata_cards(observatory: str, instrument: str):
    return [
        BLANK_CARD,
        (" ", "METADATA"),
        ("EXTNAME", _get_extname(instrument, observatory), "Extension name"),
        ("OBSRVTRY", observatory, "Observatory"),
        ("INSTRMNT", instrument, "Instrument"),
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

    

def create_summary_pipeline_primary_hdu(pipeline_model):
    # I would like to use .isoformat(), but it is too long and makes headers look disorganised.
    # Even %Y-%m-%d %H:%M:%S is one character too long! ARGH!
    created = datetime.datetime.utcnow().strftime(DATETIME_FMT)

    cards = [
        BLANK_CARD,
        (" ", "METADATA", None),
        BLANK_CARD,
        ("V_ASTRA", __version__, Glossary.v_astra),
        ("CREATED", created, f"File creation time (UTC {DATETIME_FMT})"),
        ("PIPELINE", pipeline_model.__name__, "Pipeline name"),
        BLANK_CARD,
        (" ", "HDU DESCRIPTIONS", None),
        BLANK_CARD,
        ("COMMENT", "HDU 0: Summary information only", None),
        ("COMMENT", "HDU 1: Results from BOSS spectra taken at Apache Point Observatory"),
        ("COMMENT", "HDU 2: Results from BOSS spectra taken at Las Campanas Observatory"),
        ("COMMENT", "HDU 3: Results from APOGEE spectra taken at Apache Point Observatory"),
        ("COMMENT", "HDU 4: Results from APOGEE spectra taken at Las Campanas Observatory"),
    ]
    primary_hdu = fits.PrimaryHDU(header=fits.Header(cards))
    primary_hdu.add_checksum()
    primary_hdu.header.insert("CHECKSUM", BLANK_CARD)
    primary_hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
    primary_hdu.header.insert("CHECKSUM", BLANK_CARD)
    primary_hdu.add_checksum()
    return primary_hdu


def _resolve_model(model_name):
    if isinstance(model_name, str):    
        return getattr(models, model_name) 
    else:
        return model_name

def _get_extname(instrument, observatory):
    return f"{instrument}/{observatory}"

def _get_summary_path(basename):
    return expand_path(f"$MWM_ASTRA/{__version__}/summary/{basename}")


def _prevent_summary_overwrite(path, overwrite):
    if os.path.exists(path) and not overwrite:
        raise OSError(f"File {path} already exists. If you mean to replace it then use the argument \"overwrite=True\".")


def add_category_comments(hdu, models, original_names, upper):
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
            key = f"TFORM{index}"
            hdu.header.insert(key, ("COMMENT", comment), after=True)
            category_comments_added.append(field_name)
    return None

def add_category_headers(hdu, models, original_names, upper):
    category_headers_added = []
    list_original_names = list(original_names.values())
    for model in models:
        for header, field_name in model.category_headers:
            if field_name in category_headers_added:
                continue
            try:
                index = 1 + list_original_names.index(field_name)
            except:
                continue
            key = f"TTYPE{index}"
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
                