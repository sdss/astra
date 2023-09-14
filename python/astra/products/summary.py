""""Functions for creating summary products (e.g., astraAllStar, astraAllVisit)."""

from astropy.io import fits
from astra import __version__
from astra.utils import expand_path
from astra.models import Source, ApogeeVisitSpectrum, BossVisitSpectrum
from astra.products.utils import (
    get_fields, get_basic_header, get_binary_table_hdu, check_path
)

get_path = lambda bn: expand_path(f"$MWM_ASTRA/{__version__}/summary/{bn}")

def create_all_star_product(
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
    Create an `astraAllStar` product containing the information about all sources.

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
    
    path = get_path(f"astraAllStar-{__version__}.fits")
    check_path(path, overwrite)    

    fields = get_fields(
        (Source, ),
        name_conflict_strategy=name_conflict_strategy,
        ignore_field_names=ignore_field_names
    )

    q = (
        Source
        .select(*tuple(fields.values()))
        .where(where)
        .limit(limit)
        .dicts()
    )

    hdu = get_binary_table_hdu(
        q,
        models=[Source],
        fields=fields,
        upper=upper,
        fill_values=fill_values,
        limit=limit,
    )

    hdu_list = fits.HDUList([
        fits.PrimaryHDU(header=get_basic_header()),
        hdu
    ])
    hdu_list.writeto(path, overwrite=overwrite)
    return (path, hdu_list) if full_output else path


def create_all_visit_product(
    run2d=None,
    apred=None,
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
    Create an `astraAllVisit` product containing all the visit information about all sources.

    :param run2d: [optional]
        The version of the BOSS data reduction pipeline to include. If `run2d` is given then `apred` must also be given.
    
    :param apred: [optional]
        The version of the APOGEE data reduction pipeline to include. If `run2d` is given then `apred` must also be given.
    
    :param where: [optional]
        A `where` clause for the database query.
    
    :param limit: [optional]
        Specify an optional limit on the number of rows per HDU.
    
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
    # TODO: Do we want to allow this tom-fuckery?
    if run2d is None and apred is None:
        boss_where = apogee_where = None
        path = get_path(f"astraAllVisit-{__version__}.fits")

    elif run2d is not None and apred is not None:
        boss_where = (BossVisitSpectrum.run2d == run2d)
        apogee_where = (ApogeeVisitSpectrum.apred == apred) 
        path = get_path(f"astraAllVisit-{run2d}-{apred}-{__version__}.fits")   
        
    else:
        raise ValueError(f"Either `apred` and `run2d` must both be None, or both given")

    check_path(path, overwrite)

    kwds = dict(upper=upper, fill_values=fill_values, limit=limit)

    hdus = [
        fits.PrimaryHDU(header=get_basic_header(include_hdu_descriptions=True))
    ]

    struct = [
        (BossVisitSpectrum, "apo", boss_where),
        (BossVisitSpectrum, "lco", boss_where),
        (ApogeeVisitSpectrum, "apo", apogee_where),
        (ApogeeVisitSpectrum, "lco", apogee_where),
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
            .join(Source, on=(Source.pk == model.source_pk))
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