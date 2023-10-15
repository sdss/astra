""""Functions for creating summary products (e.g., mwmTargets, mwmAllStar, mwmAllVisit)."""

from peewee import JOIN
from astropy.io import fits
from astra import __version__
from astra.utils import expand_path
from astra.models import Source, ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar, BossVisitSpectrum
from astra.products.utils import (get_fields, get_basic_header, get_binary_table_hdu, check_path)

get_path = lambda bn: expand_path(f"$MWM_ASTRA/{__version__}/summary/{bn}")

DEFAULT_IGNORE_FIELD_NAMES = ("pk", )

print("mwmTargets needs a more inclusive list of cartons!")
def create_mwm_targets_product(
    where=(
        Source.assigned_to_mapper("mwm")
    |   Source.sdss4_apogee_id.is_null(False)
    ),
    limit=None,
    output_template="mwmTargets-{version}.fits",
    ignore_field_names=DEFAULT_IGNORE_FIELD_NAMES,
    upper=False,
    fill_values=None,
    overwrite=False,
    full_output=False
):
    """
    Create an `mwmTargets` product containing a single HDU with source-level information about all targets,
    excluding any data reduction information (NOT including data reduction results).
    
    :param where: [optional]
        A `where` clause for the `Source.select()` query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows.
        
    :param output_template: [optional]
        The output basename template to use for this product.

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
    path = get_path(output_template.format(version=__version__))
    check_path(path, overwrite)

    fields = get_fields(
        (Source, ),
        name_conflict_strategy=None,
        ignore_field_names=ignore_field_names
    )
    
    q = (
        Source
        .select(*tuple(fields.values()))
    )
    if where is not None:
        q = q.where(where)
    q = (
        q
        .limit(limit)
        .dicts()
    )
    
    hdus = [
        fits.PrimaryHDU(header=get_basic_header()),
        get_binary_table_hdu(
            q,
            header=get_basic_header(),
            models=(Source, ),
            fields=fields,
            upper=upper,
            fill_values=fill_values,
            limit=limit
        )
    ]

    hdu_list = fits.HDUList(hdus)
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path    


def create_mwm_all_star_product(
    where=None,
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeCoaddedSpectrumInApStar,
    output_template="mwmAllStar-{version}.fits",
    ignore_field_names=DEFAULT_IGNORE_FIELD_NAMES,
    name_conflict_strategy=None,
    upper=False,
    fill_values=None,
    overwrite=False,
    full_output=False,
):
    """
    Create an `mwmAllStar` product containing the information about all sources (NOT including pipeline results).

    :param where: [optional]
        A `where` clause for the `Source.select()` query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows.
    
    :param boss_where: [optional]
        A `where` clause for the `boss_spectrum_model` query.
    
    :param apogee_where: [optional]
        A `where` clause for the `apogee_spectrum_model` query.

    :param boss_spectrum_model: [optional]
        The BOSS spectrum model to use when constructing this query.
    
    :param apogee_spectrum_model: [optional]
        The APOGEE spectrum model to use when constructing this query.
        
    :param output_template: [optional]
        The output basename template to use for this product.

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
    path = get_path(output_template.format(version=__version__))
    return _create_summary_product(
        path,
        where=where,
        limit=limit,
        boss_where=boss_where,
        apogee_where=apogee_where,
        boss_spectrum_model=boss_spectrum_model,
        apogee_spectrum_model=apogee_spectrum_model,
        ignore_field_names=ignore_field_names,
        name_conflict_strategy=name_conflict_strategy,
        upper=upper,
        fill_values=fill_values,
        overwrite=overwrite,
        full_output=full_output
    )


def create_mwm_all_visit_product(
    where=None,
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeVisitSpectrum,
    output_template="mwmAllVisit-{version}.fits",
    ignore_field_names=DEFAULT_IGNORE_FIELD_NAMES,
    name_conflict_strategy=None,
    upper=False,
    fill_values=None,
    overwrite=False,
    full_output=False,
):
    """
    Create an `mwmAllVisit` product containing all the visit information about all sources (NOT including pipeline results).
    
    :param where: [optional]
        A `where` clause for the database query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows per HDU.
    
    :param boss_where: [optional]
        A `where` clause for the `boss_spectrum_model` query.
    
    :param apogee_where: [optional]
        A `where` clause for the `apogee_spectrum_model` query.

    :param boss_spectrum_model: [optional]
        The BOSS spectrum model to use when constructing this query.
    
    :param apogee_spectrum_model: [optional]
        The APOGEE spectrum model to use when constructing this query.

    :param output_template: [optional]
        The output basename template to use for this product.

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
    path = get_path(output_template.format(version=__version__))
    return _create_summary_product(
        path,
        where=where,
        limit=limit,
        boss_where=boss_where,
        apogee_where=apogee_where,
        boss_spectrum_model=boss_spectrum_model,
        apogee_spectrum_model=apogee_spectrum_model,
        ignore_field_names=ignore_field_names,
        name_conflict_strategy=name_conflict_strategy,
        upper=upper,
        fill_values=fill_values,
        overwrite=overwrite,
        full_output=full_output
    )

def _create_summary_product(
    path,
    where=None,
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeVisitSpectrum,
    ignore_field_names=None,
    name_conflict_strategy=None,
    upper=False,
    fill_values=None,
    overwrite=False,
    full_output=False,        
):    
    check_path(path, overwrite)

    kwds = dict(upper=upper, fill_values=fill_values, limit=limit)

    hdus = [
        fits.PrimaryHDU(header=get_basic_header(include_hdu_descriptions=True))
    ]

    struct = [
        (boss_spectrum_model, "apo", boss_where),
        (boss_spectrum_model, "lco", boss_where),
        (apogee_spectrum_model, "apo", apogee_where),
        (apogee_spectrum_model, "lco", apogee_where),
    ]
    
    all_fields = {}
    for model, observatory, hdu_where in struct:

        instrument = model.__name__.split("Visit")[0].lower()

        models = (Source, model)
        try:
            fields = all_fields[model]
        except KeyError:
            fields = all_fields[model] = get_fields(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_names=ignore_field_names
            )

        header = get_basic_header(observatory=observatory, instrument=instrument)

        q = (
            model
            .select(*tuple(fields.values()))
            .join(Source, JOIN.LEFT_OUTER, on=(Source.pk == model.source_pk))
            .where(model.telescope.startswith(observatory))
        )        

        if hdu_where is not None:
            q = q.where(hdu_where)

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