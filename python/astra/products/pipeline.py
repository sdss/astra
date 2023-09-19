""""Functions for creating pipeline products (e.g., astraStarASPCAP, astraVisitASPCAP)."""

from astropy.io import fits
from astra import __version__
from astra.utils import log, expand_path
from astra.models import Source, ApogeeVisitSpectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar, BossVisitSpectrum
from astra.products.utils import (
    BLANK_CARD,
    get_fields_and_pixel_arrays, get_basic_header, fits_column_kwargs, get_fill_value, get_binary_table_hdu, check_path, resolve_model,
    add_category_headers, add_category_comments
)

# TODO: Probably can merge the logic from these two functions into one

def create_star_pipeline_product(
    source,
    pipeline_model,
    boss_spectrum_model=None,
    apogee_spectrum_model=None,
    ignore_field_names=("sdss5_target_flags", "source", "flux", "ivar", "pixel_flags"),
    name_conflict_strategy=None,
    upper=True,
    fill_values=None,
    limit=1,
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
    
    :param fill_values: [optional]
        Specify a fill value for each kind of column type.
    
    :param overwrite: [optional]
        Overwrite the path if it already exists.
    
    :param full_output: [optional]
        If `True`, return a two-length tuple containing the path and the HDU list,
        otherwise just return the path.     
    """

    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__

    sdss_id = source.sdss_id or source.pk
    if source.sdss_id is None:
        print("WARNING SOURCE ID IS NONE")
        raise a

    chars = str(source.sdss_id)[-4:]
    sdss_id_groups = f"{chars[:2]}/{chars[2:]}"
    
    path = expand_path(
        f"$MWM_ASTRA/{__version__}/results/star/{sdss_id_groups}/"
        f"astraStar-{pipeline}-{__version__}-{sdss_id}.fits"
    )
    check_path(path, overwrite)

    if boss_spectrum_model is None:
        log.warning("Defaulting boss_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        boss_spectrum_model = BossVisitSpectrum
    
    if apogee_spectrum_model is None:
        log.warning(f"Defaulting apogee_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        apogee_spectrum_model = ApogeeCoaddedSpectrumInApStar
    

    # TODO: Put the source info in the primary HDU, and remove the Source from the models set
    kwds = dict(upper=upper, fill_values=fill_values)
    hdus = [
        fits.PrimaryHDU(header=get_basic_header(pipeline=pipeline, include_hdu_descriptions=True))
    ]
    
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

        models = (Source, spectrum_model, pipeline_model)
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

        q = (
            pipeline_model
            .select(*models)
            .distinct(pipeline_model.spectrum_pk)
            .join(spectrum_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk), attr="__spectrum")
            .switch(pipeline_model)
            .join(Source, on=(pipeline_model.source_pk == Source.pk), attr="__source")
            .where(hdu_where & (Source.pk == source.pk))
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

    
def create_visit_pipeline_product(
    source,
    pipeline_model,
    boss_spectrum_model=None,
    apogee_spectrum_model=None,
    ignore_field_names=("sdss5_target_flags", "source", "wavelength", "flux", "ivar", "pixel_flags"),
    name_conflict_strategy=None,
    upper=True,
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
    
    :param fill_values: [optional]
        Specify a fill value for each kind of column type.
    
    :param overwrite: [optional]
        Overwrite the path if it already exists.
    
    :param full_output: [optional]
        If `True`, return a two-length tuple containing the path and the HDU list,
        otherwise just return the path.     
    """

    # TODO: Allow apred/run2d?


    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__


    #assert source.sdss_id is not None, f"Source {source} has no sdss_id"
    sdss_id = source.sdss_id or source.pk
    if source.sdss_id is None:
        print("WARNING SOURCE ID IS NONE")

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
    hdus = [
        fits.PrimaryHDU(header=get_basic_header(pipeline=pipeline, include_hdu_descriptions=True))
    ]
    
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