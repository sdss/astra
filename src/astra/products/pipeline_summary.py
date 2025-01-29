""""Functions for creating summary pipeline products (e.g., astraAllStarASPCAP, astraAllVisitASPCAP)."""

import os
import warnings
from collections import OrderedDict
from peewee import BooleanField, JOIN
from astropy.io import fits
from astra import __version__
from astra.utils import log, expand_path, version_string_to_integer
from astra.models.source import Source
from astra.models.apogee import ApogeeVisitSpectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar
from astra.models.boss import BossVisitSpectrum
from astra.models.mwm import BossCombinedSpectrum
from astra.products.utils import (get_fields, get_basic_header, get_binary_table_hdu, check_path, resolve_model)

get_path = lambda bn, gzip: expand_path(f"$MWM_ASTRA/{__version__}/summary/{bn}" + (".gz" if gzip else ""))

def ignore_field_name_callable(field_name):
    return (
        (field_name.lower() in ("pk", "input_spectrum_pks")) 
    or  field_name.lower().startswith("rho_")
    )

def create_astra_best_product(
    where=None,
    limit=None,
    output_template="astraFrankenstein-{version_major_minor}.fits",
    ignore_field_name_callable=ignore_field_name_callable,
    name_conflict_strategy=None,
    distinct_spectrum_pk=True,
    upper=False,
    fill_values=None,
    gzip=True,
    overwrite=False,
    full_output=False,
):
    """
    Create an `astraBest` product containing the results from the 'best' pipeline per source.

    :param where: [optional]
        A `where` clause for the `Source.select()` query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows.
    
    :param ignore_field_name_callable: [optional]
        A callable that returns `True` if a field name should be ignored.

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
        
    from astra.models.best import MWMBest as pipeline_model

    path = get_path(
        output_template.format(
            version=__version__,
            version_major_minor=".".join(__version__.split(".")[:2])
        ), 
        gzip
    )
    check_path(path, overwrite)
    
    kwds = dict(
        upper=upper, 
        fill_values=fill_values, 
        limit=limit,
    )
    hdus = [
        fits.PrimaryHDU(header=get_basic_header())
    ]
        
    og_fields = get_fields(
        (Source, pipeline_model),
        name_conflict_strategy=name_conflict_strategy,
        ignore_field_name_callable=ignore_field_name_callable
    )

    # Try and insert fields for `flag_warn` and `flag_bad`.
    fields = OrderedDict([])
    for k, v in og_fields.items():
        fields[k] = v
        if k == "result_flags":
            if hasattr(pipeline_model, "flag_warn"):
                fields["flag_warn"] = BooleanField(default=False, help_text="Warning flag for results")
            if hasattr(pipeline_model, "flag_bad"):
                fields["flag_bad"] = BooleanField(default=False, help_text="Bad flag for results")        

    header = get_basic_header()
    
    select_fields = []
    for n, f in fields.items():
        if n in ("flag_warn", "flag_bad"):
            select_fields.append(getattr(pipeline_model, n).alias(n))
        else:
            select_fields.append(f)

    current_version = version_string_to_integer(__version__) // 1000

    q = (
        pipeline_model
        .select(*select_fields)
        .join(Source)
        .where(pipeline_model.v_astra_major_minor == current_version)
    )   

    if where: # Need to check, otherwise it requires AND with previous where.
        q = q.where(where)
    
    q = q.limit(limit).dicts()
    
    hdu = get_binary_table_hdu(
        q,
        header=header,
        models=(Source, pipeline_model),
        fields=fields,
        **kwds
    )
    hdus.append(hdu)

    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(path, overwrite=overwrite)
    if gzip:
        os.system(f"gzip -f {path}")
        path += ".gz"
    
    return (path, hdu_list) if full_output else path    


def create_all_star_product(
    pipeline_model,
    where=None,
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossCombinedSpectrum,
    apogee_spectrum_model=ApogeeCoaddedSpectrumInApStar,
    output_template="astraAllStar{pipeline}-{version_major_minor}.fits",
    ignore_field_name_callable=ignore_field_name_callable,
    name_conflict_strategy=None,
    distinct_spectrum_pk=True,
    upper=False,
    fill_values=None,
    gzip=True,
    overwrite=False,
    full_output=False,
):
    """
    Create an `astraAllStar<PIPELINE>` product containing the results from a `<PIPELINE>`
    that was executed on co-added (star-level) spectra, but does NOT include the spectral data.

    :param pipeline_model:
        The pipeline database model to retrieve results from.

    :param boss_spectrum_model:
        The BOSS star-level (coadded) spectrum database model.
    
    :param apogee_spectrum_model:
        The APOGEE star-level (coadded) spectrum database model.
    
    :param where: [optional]
        A `where` clause for the `Source.select()` query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows.
    
    :param ignore_field_name_callable: [optional]
        A callable that returns `True` if a field name should be ignored.

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

    path = get_path(
        output_template.format(
            pipeline=pipeline, 
            version=__version__,
            version_major_minor=".".join(__version__.split(".")[:2])
        ), 
        gzip
    )
    check_path(path, overwrite)

    pipeline_model = resolve_model(pipeline_model)
    boss_spectrum_model = resolve_model(boss_spectrum_model)
    apogee_spectrum_model = resolve_model(apogee_spectrum_model)
    
    pipeline = pipeline_model.__name__

    kwds = dict(
        upper=upper, 
        fill_values=fill_values, 
        limit=limit,
    )
    hdus = [
        fits.PrimaryHDU(header=get_basic_header(pipeline=pipeline, include_hdu_descriptions=True))
    ]
    
    struct = [
    #    (boss_spectrum_model, "apo", "boss", boss_where),
    #    (boss_spectrum_model, "lco", "boss", boss_where),
    #    (apogee_spectrum_model, "apo", "apogee", apogee_where),
    #    (apogee_spectrum_model, "lco", "apogee", apogee_where),
        (boss_spectrum_model, "boss", boss_where),
        (apogee_spectrum_model, "apogee", apogee_where)
    ]
    
    all_fields = {}
    for spectrum_model, instrument, instrument_where in struct:

        models = (Source, spectrum_model, pipeline_model)
        try:
            fields = all_fields[spectrum_model]
        except KeyError:
            og_fields = get_fields(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_name_callable=ignore_field_name_callable
            )

            # Try and insert fields for `flag_warn` and `flag_bad`.
            fields = OrderedDict([])
            for k, v in og_fields.items():
                fields[k] = v
                if k == "result_flags":
                    if hasattr(pipeline_model, "flag_warn"):
                        fields["flag_warn"] = BooleanField(default=False, help_text="Warning flag for results")
                    if hasattr(pipeline_model, "flag_bad"):
                        fields["flag_bad"] = BooleanField(default=False, help_text="Bad flag for results")
                
            all_fields[spectrum_model] = fields            

        header = get_basic_header(
            pipeline=pipeline, 
        #    observatory=observatory, 
            instrument=instrument
        )
        
        select_fields = []
        for n, f in fields.items():
            if n in ("flag_warn", "flag_bad"):
                select_fields.append(getattr(pipeline_model, n).alias(n))
            else:
                select_fields.append(f)
                
        q = (
            spectrum_model
            .select(*select_fields)
        )
        if distinct_spectrum_pk:
            q = q.distinct(spectrum_model.spectrum_pk)

        current_version = version_string_to_integer(__version__) // 1000
        q = (
            q
            .join(pipeline_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk))
            .switch(spectrum_model)
            .join(Source, on=(Source.pk == spectrum_model.source_pk))
            #.where(spectrum_model.telescope.startswith(observatory))
            .where(pipeline_model.v_astra_major_minor == current_version)
        )
        if where: # Need to check, otherwise it requires AND with previous where.
            q = q.where(where)
        if instrument_where:
            q = q.where(instrument_where)
        
        q = q.limit(limit).dicts()

        hdu = get_binary_table_hdu(
            q,
            header=header,
            models=models,
            fields=fields,
            **kwds
        )
        hdus.append(hdu)

    written_path = get_path(
        output_template.format(
            pipeline=pipeline, 
            version=__version__,
            version_major_minor=".".join(__version__.split(".")[:2])
        ),
        False
    )
    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(written_path, overwrite=overwrite)
    if gzip:
        os.system(f"gzip -f {written_path}")
        written_path += ".gz"
    
    return (written_path, hdu_list) if full_output else written_path    


def create_all_visit_product(
    pipeline_model,
    where=None,
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeVisitSpectrumInApStar,
    output_template="astraAllVisit{pipeline}-{version_major_minor}.fits",    
    ignore_field_name_callable=ignore_field_name_callable,
    name_conflict_strategy=None,
    distinct_spectrum_pk=True,
    upper=False,
    fill_values=None,
    gzip=True,
    overwrite=False,
    full_output=False,
):
    """
    Create an `astraAllVisit<PIPELINE>` product containing the results from a `<PIPELINE>`
    that was executed on visit-level spectra, but does NOT include the spectral data.

    :param pipeline_model:
        The pipeline database model to retrieve results from.

    :param boss_spectrum_model:
        The BOSS star-level (coadded) spectrum database model.
    
    :param apogee_spectrum_model:
        The APOGEE star-level (coadded) spectrum database model.
    
    :param where: [optional]
        A `where` clause for the `Source.select()` query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows.
    
    :param ignore_field_name_callable: [optional]
        A callable that returns `True` if a field name should be ignored.

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
        
    if boss_spectrum_model is None:
        log.warning("Defaulting boss_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        boss_spectrum_model = BossVisitSpectrum
    
    if apogee_spectrum_model is None:
        log.warning(f"Defaulting apogee_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        apogee_spectrum_model = ApogeeVisitSpectrumInApStar

    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__

    path = get_path(
        output_template.format(
            pipeline=pipeline, 
            version=__version__,
            version_major_minor=".".join(__version__.split(".")[:2])
        ), 
        gzip
    )
    check_path(path, overwrite)

    pipeline_model = resolve_model(pipeline_model)
    boss_spectrum_model = resolve_model(boss_spectrum_model)
    apogee_spectrum_model = resolve_model(apogee_spectrum_model)
    
    pipeline = pipeline_model.__name__

    kwds = dict(
        upper=upper, 
        fill_values=fill_values, 
        limit=limit,
    )
    hdus = [
        fits.PrimaryHDU(header=get_basic_header(pipeline=pipeline, include_hdu_descriptions=True))
    ]

    drp_spectrum_models = {
        boss_spectrum_model: BossVisitSpectrum,
        apogee_spectrum_model: ApogeeVisitSpectrum
    }
    
    struct = [
    #    (boss_spectrum_model, "apo", "boss", boss_where),
    #    (boss_spectrum_model, "lco", "boss", boss_where),
    #    (apogee_spectrum_model, "apo", "apogee", apogee_where),
    #    (apogee_spectrum_model, "lco", "apogee", apogee_where),
        (boss_spectrum_model, "boss", boss_where),
        (apogee_spectrum_model, "apogee", apogee_where)
    ]
    
    all_fields = {}
    for spectrum_model, instrument, instrument_where in struct:

        drp_spectrum_model = drp_spectrum_models[spectrum_model]
        
        models = [Source]
        if drp_spectrum_model != spectrum_model:
            models.append(drp_spectrum_model)
        models.extend([spectrum_model, pipeline_model])
        
        try:
            fields = all_fields[spectrum_model]
        except KeyError:
            og_fields = get_fields(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_name_callable=ignore_field_name_callable
            )
                    
            # Try and insert fields for `flag_warn` and `flag_bad`.
            fields = OrderedDict([])
            for k, v in og_fields.items():
                fields[k] = v
                if k == "result_flags":
                    if hasattr(pipeline_model, "flag_warn"):
                        fields["flag_warn"] = BooleanField(default=False, help_text="Warning flag for results")
                    if hasattr(pipeline_model, "flag_bad"):
                        fields["flag_bad"] = BooleanField(default=False, help_text="Bad flag for results")
                                
            all_fields[spectrum_model] = fields
            
        header = get_basic_header(
            pipeline=pipeline, 
            #observatory=observatory, 
            instrument=instrument
        )

        select_fields = []
        for n, f in fields.items():
            if n in ("flag_warn", "flag_bad"):
                select_fields.append(getattr(pipeline_model, n).alias(n))
            else:
                select_fields.append(f)

        q = (
            spectrum_model
            .select(*select_fields)
        )
        if distinct_spectrum_pk:
            q = q.distinct(spectrum_model.spectrum_pk)
        
        if drp_spectrum_model != spectrum_model:
            # LEFT OUTER join 
            q = (
                q
                .join(drp_spectrum_model, JOIN.LEFT_OUTER, on=(spectrum_model.drp_spectrum_pk == drp_spectrum_model.spectrum_pk))
                .switch(spectrum_model)
            )

        current_version = version_string_to_integer(__version__) // 1000
            
        q = (
            q
            .join(pipeline_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk))
            .switch(spectrum_model)
            .join(Source, on=(Source.pk == spectrum_model.source_pk))
            .where(pipeline_model.v_astra_major_minor == current_version)
            #.where(spectrum_model.telescope.startswith(observatory))
        )
        if where: # Need to check, otherwise it requires AND with previous where.
            q = q.where(where)
        
        if instrument_where:
            q = q.where(instrument_where)
        
        q = q.limit(limit).dicts()
        
        hdu = get_binary_table_hdu(
            q,
            header=header,
            models=models,
            fields=fields,
            **kwds
        )
        hdus.append(hdu)

    written_path = get_path(
        output_template.format(
            pipeline=pipeline, 
            version=__version__,
            version_major_minor=".".join(__version__.split(".")[:2])
        ),
        False
    )
    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(written_path, overwrite=overwrite)
    if gzip:
        os.system(f"gzip -f {written_path}")
        written_path += ".gz"
    
    return (written_path, hdu_list) if full_output else written_path    
