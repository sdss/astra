""""Functions for creating summary pipeline products (e.g., astraAllStarASPCAP, astraAllVisitASPCAP)."""

from astropy.io import fits
from astra import __version__
from astra.utils import log, expand_path
from astra.models import Source, ApogeeVisitSpectrum, ApogeeVisitSpectrumInApStar, ApogeeCoaddedSpectrumInApStar, BossVisitSpectrum
from astra.products.utils import (
    get_fields, get_basic_header, get_binary_table_hdu, check_path,
    resolve_model    
)

get_path = lambda bn: expand_path(f"$MWM_ASTRA/{__version__}/summary/{bn}")

def create_all_star_pipeline_product(
    pipeline_model,
    boss_spectrum_model=None, 
    apogee_spectrum_model=None,
    where=None,
    limit=None,
    ignore_field_names=("sdss5_target_flags", ),
    name_conflict_strategy=None,
    upper=True,
    fill_values=None,
    overwrite=False,
    full_output=False,
):
    """
    Create an `astraAllStar<PIPELINE>` product containing the results from a `<PIPELINE>`
    that was executed on co-added (star-level) spectra.

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
        
    if boss_spectrum_model is None:
        log.warning("Defaulting boss_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        boss_spectrum_model = BossVisitSpectrum
    
    if apogee_spectrum_model is None:
        log.warning(f"Defaulting apogee_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        apogee_spectrum_model = ApogeeCoaddedSpectrumInApStar

    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__

    path = get_path(f"astraAllStar{pipeline}-{__version__}.fits")
    check_path(path, overwrite)

    pipeline_model = resolve_model(pipeline_model)
    boss_spectrum_model = resolve_model(boss_spectrum_model)
    apogee_spectrum_model = resolve_model(apogee_spectrum_model)
    
    pipeline = pipeline_model.__name__

    kwds = dict(upper=upper, fill_values=fill_values, limit=limit)
    hdus = [
        fits.PrimaryHDU(header=get_basic_header(pipeline=pipeline, include_hdu_descriptions=True))
    ]
    
    # TODO: Allow specification of `run2d` and `apred`
    struct = [
        (boss_spectrum_model, "apo", "boss"),
        (boss_spectrum_model, "lco", "boss"),
        (apogee_spectrum_model, "apo", "apogee"),
        (apogee_spectrum_model, "lco", "apogee"),
    ]
    
    all_fields = {}
    for spectrum_model, observatory, instrument in struct:

        models = (Source, spectrum_model, pipeline_model)
        try:
            fields = all_fields[spectrum_model]
        except KeyError:
            fields = all_fields[spectrum_model] = get_fields(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_names=ignore_field_names
            )

        header = get_basic_header(pipeline=pipeline, observatory=observatory, instrument=instrument)

        q = (
            spectrum_model
            .select(*tuple(fields.values()))
            .join(pipeline_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk))
            .switch(spectrum_model)
            .join(Source, on=(Source.pk == spectrum_model.source_pk))
            .where(spectrum_model.telescope.startswith(observatory))
        )
        if where: # Need to check, otherwise it requires AND with previous where.
            q = q.where(where)
        
        q = q.limit(limit).dicts()

        hdu = get_binary_table_hdu(
            q,
            header=header,
            models=models,
            fields=fields,
            **kwds
        )
        hdus.append(hdu)

    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path    



def create_all_visit_pipeline_product(
    pipeline_model,
    boss_spectrum_model=None, 
    apogee_spectrum_model=None,
    where=None,
    limit=None,
    ignore_field_names=("sdss5_target_flags", ),
    name_conflict_strategy=None,
    upper=True,
    fill_values=None,
    overwrite=False,
    full_output=False,
):
    """
    Create an `astraAllVisit<PIPELINE>` product containing the results from a `<PIPELINE>`
    that was executed on visit-level spectra.

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
    
    # TODO: Allow run2d and apred
    
    if boss_spectrum_model is None:
        log.warning("Defaulting boss_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        boss_spectrum_model = BossVisitSpectrum
    
    if apogee_spectrum_model is None:
        log.warning(f"Defaulting apogee_spectrum_model in astra.products.pipeline_summary.create_all_star_pipeline_product")
        apogee_spectrum_model = ApogeeVisitSpectrumInApStar

    pipeline_model = resolve_model(pipeline_model)
    pipeline = pipeline_model.__name__

    path = get_path(f"astraAllVisit{pipeline}-{__version__}.fits")
    check_path(path, overwrite)

    pipeline_model = resolve_model(pipeline_model)
    boss_spectrum_model = resolve_model(boss_spectrum_model)
    apogee_spectrum_model = resolve_model(apogee_spectrum_model)
    
    pipeline = pipeline_model.__name__

    kwds = dict(upper=upper, fill_values=fill_values, limit=limit)
    hdus = [
        fits.PrimaryHDU(header=get_basic_header(pipeline=pipeline, include_hdu_descriptions=True))
    ]

    drp_spectrum_models = {
        boss_spectrum_model: BossVisitSpectrum,
        apogee_spectrum_model: ApogeeVisitSpectrum
    }
    
    # TODO: Allow specification of `run2d` and `apred`
    struct = [
        (boss_spectrum_model, "apo", "boss"),
        (boss_spectrum_model, "lco", "boss"),
        (apogee_spectrum_model, "apo", "apogee"),
        (apogee_spectrum_model, "lco", "apogee"),
    ]
    
    all_fields = {}
    for spectrum_model, observatory, instrument in struct:

        drp_spectrum_model = drp_spectrum_models[spectrum_model]
        
        models = [Source]
        if drp_spectrum_model != spectrum_model:
            models.append(drp_spectrum_model)
        models.extend([spectrum_model, pipeline_model])
        
        try:
            fields = all_fields[spectrum_model]
        except KeyError:
            fields = all_fields[spectrum_model] = get_fields(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_names=ignore_field_names
            )

        header = get_basic_header(pipeline=pipeline, observatory=observatory, instrument=instrument)

        q = (
            spectrum_model
            .select(*tuple(fields.values()))
        )
        if drp_spectrum_model != spectrum_model:
            q = (
                q
                .join(drp_spectrum_model, on=(spectrum_model.drp_spectrum_pk == drp_spectrum_model.spectrum_pk))
                .switch(spectrum_model)
            )
            
        q = (
            q
            .join(pipeline_model, on=(pipeline_model.spectrum_pk == spectrum_model.spectrum_pk))
            .switch(spectrum_model)
            .join(Source, on=(Source.pk == spectrum_model.source_pk))
            .where(spectrum_model.telescope.startswith(observatory))
        )
        if where: # Need to check, otherwise it requires AND with previous where.
            q = q.where(where)
        
        q = q.limit(limit).dicts()

        hdu = get_binary_table_hdu(
            q,
            header=header,
            models=models,
            fields=fields,
            **kwds
        )
        hdus.append(hdu)

    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path    
