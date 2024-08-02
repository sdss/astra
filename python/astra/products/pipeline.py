""""Functions for creating pipeline products (e.g., astraStarASPCAP, astraVisitASPCAP)."""

import os
import warnings
from astropy.io import fits
from peewee import JOIN
from tqdm import tqdm
from astra import __version__
from astra.glossary import Glossary
from astra.utils import log, expand_path
from astra.models import Source, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar, BossVisitSpectrum
from astra.products.utils import (
    BLANK_CARD,
    create_source_primary_hdu,
    get_fields_and_pixel_arrays, get_basic_header,
    fits_column_kwargs, get_fill_value, check_path, resolve_model,
    add_category_headers, add_category_comments, dispersion_array
)

ASTRA_STAR_TEMPLATE = "astraStar{pipeline}-{version}-{sdss_id}.fits"
ASTRA_VISIT_TEMPLATE = "astraVisit{pipeline}-{version}-{sdss_id}.fits"
DEFAULT_STAR_IGNORE_FIELD_NAMES = ("pk", "sdss5_target_flags", "source", "wresl", "flux", "ivar", "pixel_flags", "source_pk_id", "source_pk")
DEFAULT_VISIT_IGNORE_FIELD_NAMES = tuple(list(DEFAULT_STAR_IGNORE_FIELD_NAMES) + ["wavelength"])

def create_star_pipeline_products_for_all_sources(
    pipeline_model,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeCoaddedSpectrumInApStar,
    boss_where=None,
    apogee_where=None,
    ignore_field_names=DEFAULT_STAR_IGNORE_FIELD_NAMES,
    name_conflict_strategy=None,
    upper=False,
    limit=None,
    fill_values=None,
    overwrite=False,    
):
    """
    Iterate over every source and create all the `astraStar<PIPELINE>` product which contains the 
    results from a `<PIPELINE>` for the star-level (coadded) spectra from each telescope and instrument.

    :param pipeline_model:
        The pipeline database model to retrieve results from.

    :param boss_spectrum_model:
        The BOSS spectrum database model.
    
    :param apogee_spectrum_model:
        The APOGEE spectrum database model.

    :param boss_where: [optional]
        A `where` clause for the `boss_spectrum_model` query.
    
    :param apogee_where: [optional]
        A `where` clause for the `apogee_spectrum_model` query.

    :param ignore_field_names: [optional]
        Ignore the given field names.

    :param name_conflict_strategy: [optional]
        A callable that expects

            `name_conflict_strategy(fields, name, field, model)`

        where:
    
        - `fields` is a dictionary with field names as keys and fields as values,
        - `name` is the conflicting field name (e.g., it appears already in `fields`),
        - `field` is the conflicting field,
        - `model` is the model where the conflicting `field` is bound to.
    
        This callable should update `fields` (or not).
    
    :param upper: [optional]
        Specify all field names to be in upper case.
    
    :param limit: [optional]
        Specify a limit on the number of results per HDU.
        
    :param fill_values: [optional]
        Specify a fill value for each kind of column type.
    
    :param overwrite: [optional]
        Overwrite the path if it already exists.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _create_pipeline_products_for_all_sources(
            pipeline_model=pipeline_model,
            output_template=ASTRA_STAR_TEMPLATE,
            boss_spectrum_model=boss_spectrum_model,
            apogee_spectrum_model=apogee_spectrum_model,        
            boss_where=boss_where,
            apogee_where=apogee_where,
            ignore_field_names=ignore_field_names,
            name_conflict_strategy=name_conflict_strategy,
            include_dispersion_cards=False,
            upper=upper,
            limit=limit,
            fill_values=fill_values,
            overwrite=overwrite
        )

def create_visit_pipeline_products_for_all_sources(
    pipeline_model,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeVisitSpectrumInApStar,
    boss_where=None,
    apogee_where=None,
    ignore_field_names=DEFAULT_VISIT_IGNORE_FIELD_NAMES,
    name_conflict_strategy=None,
    upper=False,
    limit=None,
    fill_values=None,
    overwrite=False,    
):
    """
    Iterate over every source and create all the `astraVisit<PIPELINE>` product which contains the 
    results from a `<PIPELINE>` for the visit-level spectra from each telescope and instrument.

    :param pipeline_model:
        The pipeline database model to retrieve results from.

    :param boss_spectrum_model:
        The BOSS spectrum database model.
    
    :param apogee_spectrum_model:
        The APOGEE spectrum database model.

    :param boss_where: [optional]
        A `where` clause for the `boss_spectrum_model` query.
    
    :param apogee_where: [optional]
        A `where` clause for the `apogee_spectrum_model` query.

    :param ignore_field_names: [optional]
        Ignore the given field names.

    :param name_conflict_strategy: [optional]
        A callable that expects

            `name_conflict_strategy(fields, name, field, model)`

        where:
    
        - `fields` is a dictionary with field names as keys and fields as values,
        - `name` is the conflicting field name (e.g., it appears already in `fields`),
        - `field` is the conflicting field,
        - `model` is the model where the conflicting `field` is bound to.
    
        This callable should update `fields` (or not).
    
    :param upper: [optional]
        Specify all field names to be in upper case.
    
    :param limit: [optional]
        Specify a limit on the number of results per HDU.
        
    :param fill_values: [optional]
        Specify a fill value for each kind of column type.
    
    :param overwrite: [optional]
        Overwrite the path if it already exists.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _create_pipeline_products_for_all_sources(
            pipeline_model=pipeline_model,
            output_template=ASTRA_VISIT_TEMPLATE,
            boss_spectrum_model=boss_spectrum_model,
            apogee_spectrum_model=apogee_spectrum_model,        
            boss_where=boss_where,
            apogee_where=apogee_where,
            include_dispersion_cards=True,
            ignore_field_names=ignore_field_names,
            name_conflict_strategy=name_conflict_strategy,
            upper=upper,
            limit=limit,
            fill_values=fill_values,
            overwrite=overwrite
        )


def create_star_pipeline_product(
    source,
    pipeline_model,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeCoaddedSpectrumInApStar,
    boss_where=None,
    apogee_where=None,    
    ignore_field_names=DEFAULT_STAR_IGNORE_FIELD_NAMES,
    name_conflict_strategy=None,
    upper=False,
    limit=None,
    fill_values=None,
    overwrite=False,
    full_output=False,        
):
    """
    Create an `astraStar<PIPELINE>` product that contains the results from a `<PIPELINE>`
    for the co-added (star-level) spectra from each telescope and instrument.

    :param source:
        The astronomical source (`astra.models.Source`).
    
    :param pipeline_model:
        The pipeline database model to retrieve results from.

    :param boss_spectrum_model:
        The BOSS star-level (coadded) spectrum database model.
    
    :param apogee_spectrum_model:
        The APOGEE star-level (coadded) spectrum database model.

    :param boss_where: [optional]
        A `where` clause for the `boss_spectrum_model` query.
    
    :param apogee_where: [optional]
        A `where` clause for the `apogee_spectrum_model` query.

    :param ignore_field_names: [optional]
        Ignore the given field names.

    :param name_conflict_strategy: [optional]
        A callable that expects

            `name_conflict_strategy(fields, name, field, model)`

        where:
    
        - `fields` is a dictionary with field names as keys and fields as values,
        - `name` is the conflicting field name (e.g., it appears already in `fields`),
        - `field` is the conflicting field,
        - `model` is the model where the conflicting `field` is bound to.
    
        This callable should update `fields` (or not).
    
    :param upper: [optional]
        Specify all field names to be in upper case.
    
    :param limit: [optional]
        Specify a limit on the number of results per HDU.
        
    :param fill_values: [optional]
        Specify a fill value for each kind of column type.
    
    :param overwrite: [optional]
        Overwrite the path if it already exists.
    
    :param full_output: [optional]
        If `True`, return a two-length tuple containing the path and the HDU list,
        otherwise just return the path.     
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _create_pipeline_product(
            source=source,
            pipeline_model=pipeline_model,
            output_template=ASTRA_STAR_TEMPLATE,
            boss_spectrum_model=boss_spectrum_model,
            apogee_spectrum_model=apogee_spectrum_model,
            boss_where=boss_where,
            apogee_where=apogee_where,    
            ignore_field_names=ignore_field_names,
            name_conflict_strategy=name_conflict_strategy,
            include_dispersion_cards=True,
            upper=upper,
            fill_values=fill_values,
            limit=limit,
            overwrite=overwrite,
            full_output=full_output,                  
        )

    
def create_visit_pipeline_product(
    source,
    pipeline_model,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeVisitSpectrumInApStar,
    boss_where=None,
    apogee_where=None,    
    ignore_field_names=DEFAULT_VISIT_IGNORE_FIELD_NAMES,
    name_conflict_strategy=None,
    upper=False,
    limit=None,
    fill_values=None,
    overwrite=False,
    full_output=False,           
):
    """
    Create an `astraVisit<PIPELINE>` product that contains the results from a `<PIPELINE>`
    for the visit-level spectra from each telescope and instrument.

    :param source:
        The astronomical source (`astra.models.Source`).
    
    :param pipeline_model:
        The pipeline database model to retrieve results from.

    :param boss_spectrum_model:
        The BOSS visit-level spectrum database model.
    
    :param apogee_spectrum_model:
        The APOGEE visit-level spectrum database model.
        
    :param boss_where: [optional]
        A `where` clause for the `boss_spectrum_model` query.
    
    :param apogee_where: [optional]
        A `where` clause for the `apogee_spectrum_model` query.

    :param ignore_field_names: [optional]
        Ignore the given field names.

    :param name_conflict_strategy: [optional]
        A callable that expects

            `name_conflict_strategy(fields, name, field, model)`

        where:
    
        - `fields` is a dictionary with field names as keys and fields as values,
        - `name` is the conflicting field name (e.g., it appears already in `fields`),
        - `field` is the conflicting field,
        - `model` is the model where the conflicting `field` is bound to.
    
        This callable should update `fields` (or not).
    
    :param upper: [optional]
        Specify all field names to be in upper case.

    :param limit: [optional]
        Specify a limit on the number of results per HDU.

    :param fill_values: [optional]
        Specify a fill value for each kind of column type.
    
    :param overwrite: [optional]
        Overwrite the path if it already exists.
    
    :param full_output: [optional]
        If `True`, return a two-length tuple containing the path and the HDU list,
        otherwise just return the path.     
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _create_pipeline_product(
            source=source,
            pipeline_model=pipeline_model,
            output_template=ASTRA_VISIT_TEMPLATE,
            boss_spectrum_model=boss_spectrum_model,
            apogee_spectrum_model=apogee_spectrum_model,
            boss_where=boss_where,
            apogee_where=apogee_where,    
            ignore_field_names=ignore_field_names,
            name_conflict_strategy=name_conflict_strategy,
            include_dispersion_cards=True,
            upper=upper,
            fill_values=fill_values,
            limit=limit,
            overwrite=overwrite,
            full_output=full_output,                  
        )


def _get_output_product_path(output_template, pipeline, sdss_id, source_pk=None):
    if sdss_id is None:
        log.warning(f"Source {source_pk} has no SDSS_ID")
        sdss_id_groups = "no_sdss_id"
        sdss_id = f"source_pk={source_pk}"
        
    else:
        chars = str(sdss_id)[-4:]
        sdss_id_groups = f"{chars[:2]}/{chars[2:]}"
        sdss_id = f"{sdss_id}"
    
    if output_template.lower().startswith("astrastar"):
        star_or_visit = "star"
    elif output_template.lower().startswith("astravisit"):
        star_or_visit = "visit"
    else:
        raise ValueError(f"Could not figure out whether this is `star` or `visit` level for the folder structure. Please use `output_template` to start with 'astraStar' or 'astraVisit'")
    
    output_dir = f"$MWM_ASTRA/{__version__}/results/{star_or_visit}/{sdss_id_groups}/"
    path = expand_path(
        os.path.join(
            output_dir,
            output_template.format(pipeline=pipeline, version=__version__, sdss_id=sdss_id)
        )
    )    
    return path


def _create_pipeline_product(
    source,
    pipeline_model,
    output_template,
    boss_spectrum_model,
    apogee_spectrum_model,
    boss_where,
    apogee_where,    
    ignore_field_names,
    name_conflict_strategy,
    include_dispersion_cards,
    upper,
    fill_values,
    limit,
    overwrite,
    full_output,                  
):

    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__

    path = _get_output_product_path(output_template, pipeline, source.sdss_id, source.pk)
    check_path(path, overwrite)
    
    kwds = dict(upper=upper, fill_values=fill_values)
    hdus = [create_source_primary_hdu(source, upper=upper)]
        
    struct = [
        (
            boss_spectrum_model, 
            "apo", 
            "boss",
            boss_where,
            boss_spectrum_model.telescope.startswith("apo")
        ),
        (
            boss_spectrum_model, 
            "lco", 
            "boss",
            boss_where,
            boss_spectrum_model.telescope.startswith("lco")
        ),              
        (
            apogee_spectrum_model, 
            "apo", 
            "apogee",
            apogee_where,
            apogee_spectrum_model.telescope.startswith("apo")
        ),
        (
            apogee_spectrum_model, 
            "lco", 
            "apogee",
            apogee_where,
            apogee_spectrum_model.telescope.startswith("lco")
        ),      
    ]

    all_fields = {}
    for spectrum_model, observatory, instrument, instrument_where, telescope_where in struct:

        models = (spectrum_model, pipeline_model)
        try:
            fields = all_fields[spectrum_model]
        except KeyError:
            fields = all_fields[spectrum_model] = get_fields_and_pixel_arrays(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_names=ignore_field_names
            )

        header = get_basic_header(
            observatory=observatory,
            instrument=instrument,
            include_dispersion_cards=include_dispersion_cards,
            upper=upper,
        )

        q = (
            pipeline_model
            .select(*models)
            #.distinct(pipeline_model.spectrum_pk) # Require distinct per spectrum_pk? if so also update order_by
            .join(spectrum_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk), attr="__spectrum")
            #.switch(pipeline_model)
            #.join(Source, on=(pipeline_model.source_pk_id == Source.pk), attr="__source")
            .where(spectrum_model.source_pk == source.pk) # note: spectrum_model.source_pk but pipeline_models are soure_pk_id eeeek whoops sorry
            .where(telescope_where)
            .where(pipeline_model.v_astra == __version__)
        )
        if instrument_where is not None:
            q = q.where(instrument_where)

        q = (
            q
            .order_by(pipeline_model.task_pk.desc())
            .limit(limit)
        )

        if q.count() > 1:
            log.warning(f"More than 1 star-level result ({q.count()}) for {pipeline} and {spectrum_model} for source {source} (observatory={observatory}, instrument={instrument})")
        
        data = { name: [] for name in fields.keys() }
        for result in q.iterator():
            for name, field in fields.items():
                if field.model == pipeline_model:
                    value = getattr(result, name)
                elif field.model == spectrum_model:
                    value = getattr(result.__spectrum, name)
                elif field.model == Source:
                    value = getattr(result.__source, name)
                else:
                    raise RuntimeError("AHH")

                if value is None:
                    value = get_fill_value(field, fill_values)
                data[name].append(value)

        original_names, columns = ({}, [])
        for name, field in fields.items():
            kwds = fits_column_kwargs(field, data[name], upper=upper)
            # Keep track of field-to-HDU names so that we can add help text.
            original_names[kwds['name']] = name
            columns.append(fits.Column(**kwds))    

        hdu = fits.BinTableHDU.from_columns(columns, header=header)
        for i, name in enumerate(hdu.data.dtype.names, start=1):
            field = fields[original_names[name]]
            hdu.header.comments[f"TTYPE{i}"] = field.help_text

        # Add category groupings.
        add_category_headers(hdu, models, original_names, upper)
        add_category_comments(hdu, models, original_names, upper)

        # Add checksums.
        hdu.add_checksum()
        hdu.header.insert("CHECKSUM", BLANK_CARD)
        hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
        hdu.header.insert("CHECKSUM", BLANK_CARD)
        hdu.add_checksum()

        hdus.append(hdu)

    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path


def _create_pipeline_products_for_all_sources(
    pipeline_model,
    output_template,
    boss_spectrum_model,
    apogee_spectrum_model,
    boss_where,
    apogee_where,
    ignore_field_names,
    name_conflict_strategy,
    include_dispersion_cards,
    upper,
    fill_values,
    limit,
    overwrite,
):
    # Find sources that have results from this pipeline with EITHER the boss_spectrum_model OR the apogee_spectrum_model
    # TODO: I'm not 100% sure this will work. Should try it, particularly with `boss_where` and `apogee_where`

    pipeline_model = resolve_model(pipeline_model)
    
    q = (
        Source
        .select()
        .distinct()
        .join(pipeline_model, on=(pipeline_model.source_pk == Source.pk))
        .join(boss_spectrum_model, JOIN.LEFT_OUTER, on=(pipeline_model.spectrum_pk == boss_spectrum_model.spectrum_pk))
        .switch(pipeline_model)
        .join(apogee_spectrum_model, JOIN.LEFT_OUTER, on=(pipeline_model.spectrum_pk == apogee_spectrum_model.spectrum_pk))
        .where(pipeline_model.v_astra == __version__)
    )
    if boss_where is not None and apogee_where is not None:
        q = q.where((boss_where) | (apogee_where))
    elif boss_where is not None:
        q = q.where(boss_where)
    elif apogee_where is not None:
        q = q.where(apogee_where)
        
    if limit is not None:
        q = q.limit(limit)

    # TODO: Parallelize and chunk this.
    N_created, N_skipped, failures = (0, 0, [])
    for source in tqdm(q, desc="Creating pipeline products"):        
        try:     
            new_path = _create_pipeline_product(
                source,
                pipeline_model,
                output_template,
                boss_spectrum_model,
                apogee_spectrum_model,
                boss_where=boss_where,
                apogee_where=apogee_where,    
                ignore_field_names=ignore_field_names,
                name_conflict_strategy=name_conflict_strategy,
                include_dispersion_cards=include_dispersion_cards,
                upper=upper,
                fill_values=fill_values,
                limit=limit,
                overwrite=overwrite,
                full_output=False
            )
        except OSError:
            if overwrite:
                # Then it failed for some other reason.
                failures.append(source)
            else:
                N_skipped += 1
        else:
            log.info(f"Created {new_path}")
            N_created += 1

    return (N_created, N_skipped, failures)




'''
old visit stuff:


    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__
    sdss_id = source.sdss_id or source.pk
    if source.sdss_id is None:
        print("WARNING SOURCE ID IS NONE")
        raise a

    print("WARNING: using 00/00 as sdss_id_groups")
    sdss_id_groups = "00/00"

    path = expand_path(
        f"$MWM_ASTRA/{__version__}/results/star/{sdss_id_groups}/"
        f"astraVisit-{pipeline}-{__version__}-{sdss_id}.fits"
    )
    check_path(path, overwrite)

    if boss_spectrum_model is None:
        log.warning("Defaulting boss_spectrum_model in astra.products.pipeline_summary.create_visit_pipeline_product")
        boss_spectrum_model = BossVisitSpectrum
    
    if apogee_spectrum_model is None:
        log.warning(f"Defaulting apogee_spectrum_model in astra.products.pipeline_summary.create_visit_pipeline_product")
        apogee_spectrum_model = ApogeeVisitSpectrumInApStar
    
    drp_spectrum_models = {
        boss_spectrum_model: BossVisitSpectrum,
        apogee_spectrum_model: ApogeeVisitSpectrum
    }

    kwds = dict(upper=upper, fill_values=fill_values)
    hdus = [create_source_primary_hdu(source)]
    
    struct = [
        (
            boss_spectrum_model, 
            "apo", 
            "boss",
            (
                boss_spectrum_model.telescope.startswith("apo")
            #&   (boss_spectrum_model.run2d == run2d)
            )
        ),
        (
            boss_spectrum_model, 
            "lco", 
            "boss",
            (
                boss_spectrum_model.telescope.startswith("lco")
            #&   (boss_spectrum_model.run2d == run2d)
            )
        ),              
        (
            apogee_spectrum_model, 
            "apo", 
            "apogee",
            (
                apogee_spectrum_model.telescope.startswith("apo")
            #&   (apogee_spectrum_model.apred == apred)
            )
        ),
        (
            apogee_spectrum_model, 
            "lco", 
            "apogee",
            (
                apogee_spectrum_model.telescope.startswith("lco")
            #&   (apogee_spectrum_model.apred == apred)
            )
        ),      
    ]

    all_fields = {}
    for spectrum_model, observatory, instrument, hdu_where in struct:

        drp_spectrum_model = drp_spectrum_models[spectrum_model]

        models = (Source, spectrum_model, drp_spectrum_model, pipeline_model)
        try:
            fields = all_fields[spectrum_model]
        except KeyError:
            fields = all_fields[spectrum_model] = get_fields_and_pixel_arrays(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_names=ignore_field_names
            )

        # TODO: Put log-lambda dispersion header cards in.
        header = get_basic_header(observatory=observatory, instrument=instrument)

        # TODO: Should we do an OUTER join based on spectrum_model?
        q = (
            pipeline_model
            .select(*models)
            .join(spectrum_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk), attr="__spectrum")
            .switch(pipeline_model)
        )
        if drp_spectrum_model != spectrum_model:
            q = (
                q
                .join(drp_spectrum_model, on=(spectrum_model.drp_spectrum_pk == drp_spectrum_model.spectrum_pk), attr="__drp_spectrum")
                .switch(pipeline_model)
            )
        q = (
            q
            .join(Source, on=(pipeline_model.source_pk == Source.pk), attr="__source")
            .where(hdu_where & (Source.pk == source.pk))
        )

        if q.count() > 1:
            log.warning(f"More than 1 star-level result ({q.count()}) for {pipeline} and {spectrum_model} for source {source} (observatory={observatory}, instrument={instrument})")
        
        data = { name: [] for name in fields.keys() }
        for result in q.iterator():
            for name, field in fields.items():
                if field.model == pipeline_model:
                    value = getattr(result, name)
                elif field.model == spectrum_model:
                    value = getattr(result.__spectrum, name)
                elif field.model == drp_spectrum_model:
                    value = getattr(result.__drp_spectrum, name)
                elif field.model == Source:
                    value = getattr(result.__source, name)
                else:
                    raise RuntimeError("AHH")

                if value is None:
                    value = get_fill_value(field, fill_values)
                data[name].append(value)

        original_names, columns = ({}, [])
        for name, field in fields.items():
            kwds = fits_column_kwargs(field, data[name], upper=upper)
            # Keep track of field-to-HDU names so that we can add help text.
            original_names[kwds['name']] = name
            columns.append(fits.Column(**kwds))    

        hdu = fits.BinTableHDU.from_columns(columns, header=header)
        for i, name in enumerate(hdu.data.dtype.names, start=1):
            field = fields[original_names[name]]
            hdu.header.comments[f"TTYPE{i}"] = field.help_text

        # Add category groupings.
        add_category_headers(hdu, models, original_names, upper)
        add_category_comments(hdu, models, original_names, upper)

        # Add checksums.
        hdu.add_checksum()
        hdu.header.insert("CHECKSUM", BLANK_CARD)
        hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
        hdu.header.insert("CHECKSUM", BLANK_CARD)
        hdu.add_checksum()

        hdus.append(hdu)

    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path
'''