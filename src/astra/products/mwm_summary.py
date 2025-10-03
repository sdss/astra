""""Functions for creating summary products (e.g., mwmTargets, mwmAllStar, mwmAllVisit)."""

import os
from peewee import JOIN
from astropy.io import fits
from astra import __version__
from astra.utils import expand_path
from astra.models.source import Source
from astra.models.apogee import ApogeeVisitSpectrum, ApogeeCoaddedSpectrumInApStar
from astra.models.boss import BossVisitSpectrum
from astra.models.mwm import BossCombinedSpectrum, ApogeeCombinedSpectrum
from astra.products.utils import (get_fields, get_basic_header, get_binary_table_hdu, check_path)

get_path = lambda bn: expand_path(f"$MWM_ASTRA/{__version__}/summary/{bn}")

STELLAR_LIKE_CARTON_NAMES = (
    'bhm_csc_apogee',
    'bhm_csc_boss',
    'bhm_csc_boss-bright',
    'bhm_csc_boss-dark',
    'bhm_csc_boss_bright',
    'bhm_csc_boss_dark',
    'manual_bright_target_offsets_1',
    'manual_bright_target_offsets_1_g13',
    'manual_bright_target_offsets_3',
    'manual_bright_targets_g13',
    'manual_bright_targets_g13_offset_fixed_1',
    'manual_bright_targets_g13_offset_fixed_3',
    'manual_bright_targets_g13_offset_fixed_5',
    'manual_bright_targets_g13_offset_fixed_7',
    'manual_fps_position_stars',
    'manual_fps_position_stars_10',
    'manual_fps_position_stars_apogee_10',
    'manual_offset_mwmhalo_off00',
    'manual_offset_mwmhalo_off05',
    'manual_offset_mwmhalo_off10',
    'manual_offset_mwmhalo_off20',
    'manual_offset_mwmhalo_off30',
    'manual_offset_mwmhalo_offa',
    'manual_offset_mwmhalo_offb',
    'openfibertargets_nov2020_10',
    'openfibertargets_nov2020_1000',
    'openfibertargets_nov2020_1001a',
    'openfibertargets_nov2020_1001b',
    'openfibertargets_nov2020_12',
    'openfibertargets_nov2020_14',
    'openfibertargets_nov2020_15',
    'openfibertargets_nov2020_17',
    'openfibertargets_nov2020_19a',
    'openfibertargets_nov2020_19b',
    'openfibertargets_nov2020_19c',
    'openfibertargets_nov2020_22',
    'openfibertargets_nov2020_24',
    'openfibertargets_nov2020_25',
    'openfibertargets_nov2020_28a',
    'openfibertargets_nov2020_28b',
    'openfibertargets_nov2020_28c',
    'openfibertargets_nov2020_29',
    'openfibertargets_nov2020_3',
    'openfibertargets_nov2020_31',
    'openfibertargets_nov2020_32',
    'openfibertargets_nov2020_34a',
    'openfibertargets_nov2020_34b',
    'openfibertargets_nov2020_35a',
    'openfibertargets_nov2020_35b',
    'openfibertargets_nov2020_35c',
    'openfibertargets_nov2020_46',
    'openfibertargets_nov2020_47a',
    'openfibertargets_nov2020_47b',
    'openfibertargets_nov2020_47c',
    'openfibertargets_nov2020_47d',
    'openfibertargets_nov2020_47e',
    'openfibertargets_nov2020_5',
    'openfibertargets_nov2020_6a',
    'openfibertargets_nov2020_6b',
    'openfibertargets_nov2020_6c',
    'openfibertargets_nov2020_8',
    'openfibertargets_nov2020_9',
    'ops_apogee_stds',
    'ops_std_apogee',
    'ops_std_boss',
    'ops_std_boss-red',
    'ops_std_boss_gdr2',
    'ops_std_boss_lsdr8',
    'ops_std_boss_ps1dr2',
    'ops_std_boss_red',
    'ops_std_boss_tic',
    'ops_std_eboss',
)

DEFAULT_MWM_WHERE = (
    Source.assigned_to_mapper("mwm")
|   Source.sdss4_apogee_id.is_null(False)
|   Source.assigned_to_carton_with_name('bhm_csc_apogee')
|   Source.assigned_to_carton_with_name('bhm_csc_boss')
|   Source.assigned_to_carton_with_name('bhm_csc_boss-bright')
|   Source.assigned_to_carton_with_name('bhm_csc_boss-dark')
|   Source.assigned_to_carton_with_name('bhm_csc_boss_bright')
|   Source.assigned_to_carton_with_name('bhm_csc_boss_dark')
|   Source.assigned_to_carton_with_name('manual_bright_target_offsets_1')
|   Source.assigned_to_carton_with_name('manual_bright_target_offsets_1_g13')
|   Source.assigned_to_carton_with_name('manual_bright_target_offsets_3')
|   Source.assigned_to_carton_with_name('manual_bright_targets_g13')
|   Source.assigned_to_carton_with_name('manual_bright_targets_g13_offset_fixed_1')
|   Source.assigned_to_carton_with_name('manual_bright_targets_g13_offset_fixed_3')
|   Source.assigned_to_carton_with_name('manual_bright_targets_g13_offset_fixed_5')
|   Source.assigned_to_carton_with_name('manual_bright_targets_g13_offset_fixed_7')
|   Source.assigned_to_carton_with_name('manual_fps_position_stars')
|   Source.assigned_to_carton_with_name('manual_fps_position_stars_10')
|   Source.assigned_to_carton_with_name('manual_fps_position_stars_apogee_10')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_off00')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_off05')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_off10')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_off20')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_off30')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_offa')
|   Source.assigned_to_carton_with_name('manual_offset_mwmhalo_offb')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_10')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_1000')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_1001a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_1001b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_12')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_14')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_15')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_17')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_19a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_19b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_19c')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_22')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_24')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_25')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_28a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_28b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_28c')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_29')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_3')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_31')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_32')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_34a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_34b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_35a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_35b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_35c')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_46')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_47a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_47b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_47c')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_47d')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_47e')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_5')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_6a')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_6b')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_6c')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_8')
|   Source.assigned_to_carton_with_name('openfibertargets_nov2020_9')
|   Source.assigned_to_carton_with_name('ops_apogee_stds')
|   Source.assigned_to_carton_with_name('ops_std_apogee')
|   Source.assigned_to_carton_with_name('ops_std_boss')
|   Source.assigned_to_carton_with_name('ops_std_boss-red')
|   Source.assigned_to_carton_with_name('ops_std_boss_gdr2')
|   Source.assigned_to_carton_with_name('ops_std_boss_lsdr8')
|   Source.assigned_to_carton_with_name('ops_std_boss_ps1dr2')
|   Source.assigned_to_carton_with_name('ops_std_boss_red')
|   Source.assigned_to_carton_with_name('ops_std_boss_tic')
|   Source.assigned_to_carton_with_name('ops_std_eboss')
)
def ignore_field_name_callable(field_name):
    return field_name in ("pk", "input_spectrum_pks", )


def create_mwm_targets_product(
    where=(
        DEFAULT_MWM_WHERE
    &   ((Source.n_apogee_visits > 0) | (Source.n_boss_visits > 0))
    ),
    limit=None,
    output_template="mwmTargets-{version_major_minor}.fits",
    ignore_field_name_callable=ignore_field_name_callable,
    upper=False,
    fill_values=None,
    overwrite=False,
    gzip=True,
    full_output=False,
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
        
    :param gzip: [optional]
        Gzip the file.
    
    :param full_output: [optional]
        If `True`, return a two-length tuple containing the path and the HDU list,
        otherwise just return the path.      
    """
    path = get_path(output_template.format(
        version=__version__,
        version_major_minor=".".join(__version__.split(".")[:2])
    ))
    check_path(path, overwrite, gzip)

    fields = get_fields(
        (Source, ),
        name_conflict_strategy=None,
        ignore_field_name_callable=ignore_field_name_callable
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
    if gzip:
        os.system(f"gzip -f {path}")
        path += ".gz"
    os.system(f"chmod 755 {path}")        
    return (path, hdu_list) if full_output else path    


def create_mwm_all_star_product(
    where=(
        DEFAULT_MWM_WHERE
    &   ((Source.n_apogee_visits > 0) | (Source.n_boss_visits > 0))
    ),
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossCombinedSpectrum,
    apogee_spectrum_model=ApogeeCoaddedSpectrumInApStar,
    output_template="mwmAllStar-{version_major_minor}.fits",
    ignore_field_name_callable=ignore_field_name_callable,
    name_conflict_strategy=None,
    upper=False,
    fill_values=None,
    gzip=True,
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
    path = get_path(output_template.format(
        version=__version__,
        version_major_minor=".".join(__version__.split(".")[:2])
    ))

    return _create_summary_product(
        path,
        where=where,
        limit=limit,
        boss_where=boss_where,
        apogee_where=apogee_where,
        boss_spectrum_model=boss_spectrum_model,
        apogee_spectrum_model=apogee_spectrum_model,
        ignore_field_name_callable=ignore_field_name_callable,
        name_conflict_strategy=name_conflict_strategy,
        upper=upper,
        fill_values=fill_values,
        overwrite=overwrite,
        gzip=gzip,
        full_output=full_output
    )


def create_mwm_all_visit_product(
    where=(
        DEFAULT_MWM_WHERE
    &   ((Source.n_apogee_visits > 0) | (Source.n_boss_visits > 0))
    ),
    limit=None,
    boss_where=None,
    apogee_where=None,
    boss_spectrum_model=BossVisitSpectrum,
    apogee_spectrum_model=ApogeeVisitSpectrum,
    output_template="mwmAllVisit-{version_major_minor}.fits",
    ignore_field_name_callable=ignore_field_name_callable,
    name_conflict_strategy=None,
    upper=False,
    fill_values=None,
    gzip=True,
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
    path = get_path(output_template.format(
        version=__version__,
        version_major_minor=".".join(__version__.split(".")[:2])
    ))
    return _create_summary_product(
        path,
        where=where,
        limit=limit,
        boss_where=boss_where,
        apogee_where=apogee_where,
        boss_spectrum_model=boss_spectrum_model,
        apogee_spectrum_model=apogee_spectrum_model,
        ignore_field_name_callable=ignore_field_name_callable,
        name_conflict_strategy=name_conflict_strategy,
        upper=upper,
        fill_values=fill_values,
        gzip=gzip,
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
    ignore_field_name_callable=None,
    name_conflict_strategy=None,
    upper=False,
    fill_values=None,
    gzip=True,
    overwrite=False,
    full_output=False,        
):    
    check_path(path, overwrite, gzip)

    kwds = dict(upper=upper, fill_values=fill_values, limit=limit)

    hdus = [
        fits.PrimaryHDU(header=get_basic_header(include_hdu_descriptions=True))
    ]

    struct = [
    #    (boss_spectrum_model, "apo", boss_where),
    #    (boss_spectrum_model, "lco", boss_where),
    #    (apogee_spectrum_model, "apo", apogee_where),
    #    (apogee_spectrum_model, "lco", apogee_where),
        (boss_spectrum_model, boss_where),
        (apogee_spectrum_model, apogee_where)
    ]
    
    all_fields = {}
    for model, hdu_where in struct:

        instrument = "BOSS" if "boss" in model.__name__.lower() else "APOGEE"
        
        models = (Source, model)
        try:
            fields = all_fields[model]
        except KeyError:
            fields = all_fields[model] = get_fields(
                models,
                name_conflict_strategy=name_conflict_strategy,
                ignore_field_name_callable=ignore_field_name_callable
            )

        header = get_basic_header(
            #observatory=observatory, 
            instrument=instrument
        )

        q = (
            model
            .select(*tuple(fields.values()))
            .join(Source, JOIN.LEFT_OUTER, on=(Source.pk == model.source_pk))
            #.where(model.telescope.startswith(observatory))
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
    if gzip:
        os.system(f"gzip -f {path}")
        path += ".gz"    
    os.system(f"chmod 755 {path}")        
    return (path, hdu_list) if full_output else path