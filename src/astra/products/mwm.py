"""Functions to create mwmVisit and mwmStar products."""

import os
import concurrent.futures
from astropy.io import fits
from astropy.io.fits.card import VerifyWarning
from peewee import JOIN
from tqdm import tqdm
from astra import task, __version__
from astra.utils import log
from astra.models.source import Source
from datetime import datetime
from typing import Iterable, Optional
import warnings
from astra.fields import BasePixelArrayAccessor
from astra.models.mwm import (
    MWMStarMixin, MWMVisitMixin, MWMSpectrumProductStatus,
    BossCombinedSpectrum, BossRestFrameVisitSpectrum,
    ApogeeCombinedSpectrum, ApogeeRestFrameVisitSpectrum,
)
from astra.products.utils import (
    BLANK_CARD,
    create_source_primary_hdu,
    create_source_primary_hdu_cards,
    create_source_primary_hdu_from_cards,
    get_fields_and_pixel_arrays, get_basic_header,
    fits_column_kwargs, get_fill_value, check_path, resolve_model,
    add_category_headers, add_category_comments, dispersion_array
)

DEFAULT_STAR_IGNORE_FIELD_NAMES = ("pk", "sdss5_target_flags", "source", "source_pk", "source_pk_id", "input_spectrum_pks")
DEFAULT_VISIT_IGNORE_FIELD_NAMES = tuple(list(DEFAULT_STAR_IGNORE_FIELD_NAMES) + ["wavelength"])

from astra.products.mwm_summary import DEFAULT_MWM_WHERE, STELLAR_LIKE_CARTON_NAMES
from astra.products.apogee import prepare_apogee_resampled_visit_and_coadd_spectra
from astra.products.boss import prepare_boss_resampled_visit_and_coadd_spectra

@task
def create_mwmVisit_and_mwmStar_products(
    sources: Iterable[Source], 
    apreds: Optional[Iterable[str]] = ("1.4", "dr17"),
    run2ds: Optional[Iterable[str]] = ("v6_2_0", ),
    max_workers: Optional[int] = 128,
    **kwargs
) -> Iterable[MWMSpectrumProductStatus]:
    for source in sources:
        cartons = source.sdss5_cartons
        has_sdss_id = (source.sdss_id is not None)
        is_stellar_like = (
            "mwm" in cartons["mapper"]
            or source.sdss4_apogee_id is not None
            or set(cartons["name"]).intersection(STELLAR_LIKE_CARTON_NAMES)
        )
        if has_sdss_id and is_stellar_like:
            source_pk, flagged_exception, created_visit, created_star = _create_mwmVisit_and_mwmStar_products(source, apreds, run2ds, debug=True, overwrite=True)
        else:
            flagged_exception = created_visit = created_star = False

        yield MWMSpectrumProductStatus(
            source_pk=source.pk, 
            flag_skipped_because_no_sdss_id=not has_sdss_id,
            flag_skipped_because_not_stellar_like=not is_stellar_like,
            flag_attempted_but_exception=flagged_exception,
            flag_created_mwm_visit=created_visit,
            flag_created_mwm_star=created_star
        )
    
    """
    futures = []        
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:        
        for source in sources:
            cartons = source.sdss5_cartons
            has_sdss_id = (source.sdss_id is not None)
            is_stellar_like = (
                "mwm" in cartons["mapper"]
                or source.sdss4_apogee_id is not None
                or set(cartons["name"]).intersection(STELLAR_LIKE_CARTON_NAMES)
            )
            if has_sdss_id and is_stellar_like:
                futures.append(executor.submit(_create_mwmVisit_and_mwmStar_products, source, apreds, run2ds))
            else:
                yield MWMSpectrumProductStatus(
                    source_pk=source.pk, 
                    flag_skipped_because_no_sdss_id=not has_sdss_id,
                    flag_skipped_because_not_stellar_like=not is_stellar_like,
                    flag_attempted_but_exception=False,
                    flag_created_mwm_visit=False,
                    flag_created_mwm_star=False
                )

        for future in concurrent.futures.as_completed(futures):
            source_pk, flag_exception, flag_created_mwm_visit, flag_created_mwm_star = future.result()
            yield MWMSpectrumProductStatus(
                source_pk=source_pk, 
                flag_skipped_because_no_sdss_id=False,
                flag_skipped_because_not_stellar_like=False,
                flag_attempted_but_exception=flag_exception,
                flag_created_mwm_visit=flag_created_mwm_visit,
                flag_created_mwm_star=flag_created_mwm_star
            )
    """


@task
def old_create_all_mwm_products(apreds=("dr17", "1.4"), run2ds=("v6_1_3", ), page=None, limit=None, max_workers=1, **kwargs):
    warnings.simplefilter("ignore") # astropy fits warnings

    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.boss import BossVisitSpectrum
    from peewee import JOIN


    if isinstance(apreds, str):
        apreds = (apreds, )
    if isinstance(run2ds, str):
        run2ds = (run2ds, )

    if apreds is not None and run2ds is not None:
        q_apogee = (
            ApogeeVisitSpectrum
            .select(ApogeeVisitSpectrum.source_pk)
            .distinct(ApogeeVisitSpectrum.source_pk)
            .where(ApogeeVisitSpectrum.apred.in_(apreds))
            .alias("q_apogee")
        )
        q_boss = (
            BossVisitSpectrum
            .select(BossVisitSpectrum.source_pk)
            .distinct(BossVisitSpectrum.source_pk)
            .where(BossVisitSpectrum.run2d.in_(run2ds))
            .alias("q_boss")
        )

        q = (
            Source
            .select()
            .distinct(Source.pk)
            .where(
                Source.sdss_id.is_null(False)
            &   DEFAULT_MWM_WHERE
            )
            .join(q_apogee, JOIN.LEFT_OUTER, on=(q_apogee.c.source_pk == Source.pk))
            .switch(Source)
            .join(q_boss, JOIN.LEFT_OUTER, on=(q_boss.c.source_pk == Source.pk))
            .where(
                (~q_apogee.c.source_pk.is_null())
            |   (~q_boss.c.source_pk.is_null())
            )
        )       

    else:
        raise NotImplementedError

    q = q.order_by(Source.pk.desc())

    if page is not None and limit is not None:
        q = q.paginate(page, limit)
        total = limit
    elif limit is not None:
        q = q.limit(limit)
        total = limit
    else:
        total = None
        
    if max_workers > 1:
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        futures = []
        for source in tqdm(q, total=total, desc="Submitting"):
            futures.append(executor.submit(create_mwmVisit_and_mwmStar_products, source, apreds, run2ds))
        
        completed = []
        with tqdm(total=len(futures)) as pb:
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
                pb.update()

    else:   
        for source in tqdm(q, total=total, desc="Creating"):     
            try:
                create_mwmVisit_and_mwmStar_products(source, apreds, run2ds, **kwargs)
            except:
                log.exception(f"Exception trying to create mwmVisit/mwmStar products for {source}")
                raise 
    yield None


def _create_mwmVisit_and_mwmStar_products(
    source,
    apreds=None,
    run2ds=None,
    star_ignore_field_names=DEFAULT_STAR_IGNORE_FIELD_NAMES,
    visit_ignore_field_names=DEFAULT_VISIT_IGNORE_FIELD_NAMES,
    fill_values=None,
    upper=False,
    overwrite=False,
    debug=False
):
    try:
        # use a fake ApogeeCombinedSpectrum to get the right path
        mwmStar_path = BossCombinedSpectrum(sdss_id=source.sdss_id, v_astra=__version__).absolute_path
        mwmVisit_path = ApogeeRestFrameVisitSpectrum(sdss_id=source.sdss_id, v_astra=__version__).absolute_path

        for model in (BossCombinedSpectrum, ApogeeCombinedSpectrum, BossRestFrameVisitSpectrum, ApogeeRestFrameVisitSpectrum):
            model.delete().where(model.source_pk == source.pk).execute()

        if not overwrite and (os.path.exists(mwmStar_path) or os.path.exists(mwmVisit_path)):
            return (source.pk, True, os.path.exists(mwmVisit_path), os.path.exists(mwmStar_path))
        

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=VerifyWarning)

            cards, original_names = create_source_primary_hdu_cards(source, context="spectra")    
            mwmVisit_hdus = [create_source_primary_hdu_from_cards(source, cards, original_names), None, None, None, None]
            mwmStar_hdus = [create_source_primary_hdu_from_cards(source, cards, original_names), None, None, None, None]
        
            star_kwds = dict(
                include_dispersion_cards=True,
                ignore_field_names=star_ignore_field_names,
                fill_values=fill_values,
                upper=upper,
            )
            visit_kwds = dict(
                include_dispersion_cards=True,
                ignore_field_names=visit_ignore_field_names,
                fill_values=fill_values,
                upper=upper,
            )            
            
            any_coadd, any_visit = (False, False)
            for i, telescope in enumerate(("apo25m", "lco25m")):
                observatory = telescope[:3].upper()
                
                try:
                    boss_coadd, boss_visit = prepare_boss_resampled_visit_and_coadd_spectra(source, telescope, run2ds)    
                except FileNotFoundError:
                    boss_coadd = boss_visit = None
                    log.exception(f"Exception preparing BOSS spectra for {source} / {telescope} / {run2ds}")
                    if debug:
                        raise
                    
                coadd_boss_hdu = _create_single_model_hdu([boss_coadd], BossCombinedSpectrum, observatory, "BOSS", **star_kwds)

                visit_boss_hdu = _create_single_model_hdu(boss_visit, BossRestFrameVisitSpectrum, observatory, "BOSS", **visit_kwds)        
                
                # Here we are going to consider things just from the OBSERVATORY so that we include apo1m spectra with apo25m spectra
                try:
                    apogee_coadd, apogee_visit = prepare_apogee_resampled_visit_and_coadd_spectra(source, observatory, apreds)
                except FileNotFoundError:
                    apogee_coadd = apogee_visit = None
                    log.exception(f"Exception preparing APOGEE spectra for {source} / {observatory} / {apreds}")
                    if debug:
                        raise
                
                coadd_apogee_hdu = _create_single_model_hdu([apogee_coadd], ApogeeCombinedSpectrum, observatory, "APOGEE", **star_kwds)        
                
                visit_apogee_hdu = _create_single_model_hdu(apogee_visit, ApogeeRestFrameVisitSpectrum, observatory, "APOGEE", **visit_kwds)
                            
                # Order is: BOSS APO, BOSS LCO, APOGEE APO, APOGEE LCO
                mwmVisit_hdus[1 + i] = visit_boss_hdu
                mwmVisit_hdus[3 + i] = visit_apogee_hdu
                
                mwmStar_hdus[1 + i] = coadd_boss_hdu
                mwmStar_hdus[3 + i] = coadd_apogee_hdu
                
                if boss_coadd is not None or apogee_coadd is not None:
                    any_coadd = True
                if boss_visit is not None or apogee_visit is not None:
                    any_visit = True

            
            for path in (mwmStar_path, mwmVisit_path):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            if any_coadd:
                mwmStar = fits.HDUList(mwmStar_hdus)
                mwmStar.writeto(mwmStar_path, overwrite=True)
                #log.info(f"Created {mwmStar_path}")
                
            #else:
            #    log.info(f"No mwmStar created for {mwmStar_path}")

            if any_visit:        
                mwmVisit = fits.HDUList(mwmVisit_hdus)
                mwmVisit.writeto(mwmVisit_path, overwrite=True)        
                #log.info(f"Created {mwmVisit_path}")
            #else:
            #    log.info(f"No mwmVisit created for {mwmVisit_path}")
    except:
        log.exception(f"Exception on source {source}")
        if debug:
            raise
        return (source.pk, True, False, False)
    else:
        #source.updated_mwm_visit_mwm_star_products = datetime.now()
        #source.save()
        return (source.pk, False, any_coadd, any_visit)


def _create_single_model_hdu(
    results,
    model,
    observatory=None,
    instrument=None,
    include_dispersion_cards=False,
    ignore_field_names=DEFAULT_STAR_IGNORE_FIELD_NAMES,
    fill_values=None,
    upper=False
):
    if results is None:
        results = []
    fields = get_fields_and_pixel_arrays(
        (model, ),
        name_conflict_strategy=None,
        ignore_field_names=ignore_field_names,
    )
    data = { name: [] for name in fields.keys() }
    for result in results:
        if result is None:
            continue
        for name, field in fields.items():
            value = getattr(result, name)
            if value is None:
                value = get_fill_value(field, fill_values)
            data[name].append(value)    
            
    original_names, columns = ({}, [])
    for name, field in fields.items():
        kwds = fits_column_kwargs(field, data[name], upper=upper)
        # Keep track of field-to-HDU names so that we can add help text.
        original_names[kwds['name']] = name
        columns.append(fits.Column(**kwds))

    hdu = fits.BinTableHDU.from_columns(
        columns, 
        header=get_basic_header(
            observatory=observatory,
            instrument=instrument,
            include_dispersion_cards=include_dispersion_cards,
            upper=upper,
        )
    )
    for i, name in enumerate(hdu.data.dtype.names, start=1):
        field = fields[original_names[name]]
        hdu.header.comments[f"TTYPE{i}"] = field.help_text

    # Add category groupings.
    add_category_headers(hdu, (model, ), original_names, upper, suppress_warnings=True)
    add_category_comments(hdu, (model, ), original_names, upper)

    # Add checksums.
    hdu.add_checksum()
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.add_checksum()
    
    return hdu

