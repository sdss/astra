
from astropy.table import Table
from collections import OrderedDict
from glob import glob
from peewee import chunked
import numpy as np
from tqdm import tqdm
import h5py as h5


from astra.utils import dict_to_iterable

from astra.models import Source
from astra.models.arjl import (
    ARJLTHVisitSpectrum, ARJLDDVisitSpectrum,
    ARJLTHRestFrameVisitSpectrum, ARJLDDRestFrameVisitSpectrum,
    ARJLTHStarLinesVisitSpectrum, ARJLDDStarLinesVisitSpectrum,
    ARJLTHStarLinesRestFrameVisitSpectrum, ARJLDDStarLinesRestFrameVisitSpectrum,
    ARJLTHCoaddedSpectrum, ARJLDDCoaddedSpectrum,
    ARJLTHStarLinesCoaddedSpectrum, ARJLDDStarLinesCoaddedSpectrum,
    prefetch_pixel_arrays,
)
from astra.migrations.utils import enumerate_new_spectrum_pks




def shift(array, pixels, fill_value=0):
    pixels = np.asarray(pixels).astype(int)

    if pixels.ndim == 0:
        pixels = int(pixels)
        pad = fill_value * np.ones(abs(pixels))
        if pixels >= 0:
            return np.hstack([array[pixels:], pad])
        else:
            return np.hstack([pad, array[:pixels]])

    N, P = array.shape
    col_idx = np.arange(P)
    src_idx = col_idx[np.newaxis, :] + pixels[:, np.newaxis]  # (N, P)

    valid = (src_idx >= 0) & (src_idx < P)
    row_idx = np.arange(N)[:, np.newaxis] * np.ones((1, P), dtype=int)

    out = np.where(valid, array[row_idx, np.clip(src_idx, 0, P - 1)], fill_value)
    return out



def ingest_arjl_dr17_spectra(
    base_dir="/uufs/chpc.utah.edu/common/home/sdss50/dr19/vac/mwm/apMADGICS/v2024_03_16",
    batch_size=1000,
    limit=None
):
    from astra.models.base import database

    # Lookup all sdss_id vs source_pk first
    q = (
        Source
        .select(Source.pk, Source.sdss_id)
        .tuples()
    )
    source_pk_to_sdss_id = { sdss_id: pk for pk, sdss_id in q }

    # Ingest theory sepctra
    kinds = [
        ("outdir_wu_th/", (
            ARJLTHVisitSpectrum, ARJLTHRestFrameVisitSpectrum,
            ARJLTHStarLinesVisitSpectrum, ARJLTHStarLinesRestFrameVisitSpectrum,
        )),
        ("outdir_wu_dd/", (
            ARJLDDVisitSpectrum, ARJLDDRestFrameVisitSpectrum,
            ARJLDDStarLinesVisitSpectrum, ARJLDDStarLinesRestFrameVisitSpectrum,
        )),
    ]

    flatten = lambda x: x.flatten()[0]

    for subdir, models in kinds:

        with database.atomic():
            database.create_tables(models)

        visits = Table.read(glob(f"{base_dir}/{subdir}/allVisit_*.fits")[0])
        if limit is not None:
            visits = visits[:limit]

        n_unmatched, n_visits = (0, len(visits))

        rows_as_dict = dict(
            sdss_id=visits["SDSS_ID"].flatten(),
            release=["dr19"] * n_visits,
            v_arjl=["v2024_03_16"] * n_visits,
            source_pk=[ None ] * n_visits,
            spectrum_pk=[ None ] * n_visits,
            component_dir=[f"{base_dir}/{subdir}"] * n_visits,
            #row_index=np.arange(n_visits),
            row_index=visits["map2madgics"].flatten() - 1,  # 1-indexed in the file, convert to 0-indexed
            mjd=visits["MJD"],
            plate=visits["PLATE"],
            field=visits["FIELD"],
            obj=visits["APOGEE_ID"],
            fiber=visits["FIBERID"],
            telescope=visits["TELESCOPE"],
            adjfiberindx=visits["adjfiberindx"],
            v_rad=visits["RV_bary"],
            v_rad_flags=visits["RV_flag"],
            v_rad_minchi2_final=visits["RV_minchi2_final"],
            v_rad_pix_var=visits["RV_pix_var"],
            v_rad_pixoff_disc_final=visits["RV_pixoff_disc_final"],
            v_rad_pixoff_final=visits["RV_pixoff_final"],
            v_rel=visits["RV_vel"],
            v_rad_chi2_residuals=visits["RVchi2_residuals"],
            drp_snr=visits["DRP_SNR"],
            drp_starflag=visits["DRP_STARFLAG"],
            drp_vhelio=visits["DRP_VHELIO"],
            drp_vrel=visits["DRP_VREL"],
            drp_vrelerr=visits["DRP_VRELERR"],
            dr17_teff=visits["DR17_TEFF"],
            dr17_logg=visits["DR17_LOGG"],
            dr17_x_h=visits["DR17_X_H"],
            dr17_vsini=visits["DR17_VSINI"],
        )

        # iterate and only keep what has a source linking
        # DELETE THIS ONCE HAVE PROPER MIGRATION!!!!!
        keep = []
        for i, (spectrum_pk, sdss_id) in enumerate(enumerate_new_spectrum_pks(tqdm(rows_as_dict["sdss_id"]))):
            source_pk = source_pk_to_sdss_id.get(sdss_id)

            if source_pk is None:
                n_unmatched += 1
                continue

            rows_as_dict["spectrum_pk"][i] = spectrum_pk
            rows_as_dict["source_pk"][i] = source_pk
            keep.append(i)

        for key in ("plate", "field", "obj"):
            rows_as_dict[key] = list(map(str.strip, rows_as_dict[key]))

        rows_as_dict = {
            key: np.asarray(value)[keep]
            for key, value in rows_as_dict.items()
        }

        # for i, (spectrum_pk, sdss_id) in enumerate(enumerate_new_spectrum_pks(tqdm(rows_as_dict["sdss_id"]))):
        #     source_pk = source_pk_to_sdss_id.get(sdss_id, None)
        #     rows_as_dict["spectrum_pk"][i] = spectrum_pk
        #     rows_as_dict["source_pk"][i] = source_pk
        #     if source_pk is None:
        #         n_unmatched += 1

        print(f"Found {n_unmatched} unmatched sources in {subdir}")

        for model in models:
            n = 0
            with database.atomic():
                with tqdm(total=n_visits, desc=f"Ingesting {model.__name__}") as pb:
                    for chunk in chunked(dict_to_iterable(rows_as_dict), batch_size):
                        (
                            model
                            .insert_many(chunk)
                            .execute()
                        )
                        n += len(chunk)
                        pb.update(len(chunk))

            # For now, delete the sources with source_pk as None (unmatched) since they will cause issues downstream
            # and I don't want to write the mgiration code yet.
            n_deleted = (
                model
                .delete()
                .where(model.source_pk.is_null())
                .execute()
            )

    # now add in the coadds
    model_pairs = [(ARJLTHRestFrameVisitSpectrum, ARJLTHCoaddedSpectrum),
                   (ARJLDDRestFrameVisitSpectrum, ARJLDDCoaddedSpectrum),
                   (ARJLTHStarLinesRestFrameVisitSpectrum, ARJLTHStarLinesCoaddedSpectrum),
                   (ARJLDDStarLinesRestFrameVisitSpectrum, ARJLDDStarLinesCoaddedSpectrum)]
    for visit_model, coadd_model in model_pairs:
        _ = build_arjl_coadds(visit_model, coadd_model)


def weighted_average(x, e_x):
    """Inverse-variance weighted mean of `x`, given per-element uncertainties `e_x`."""
    ivar = np.array(e_x) ** -2
    sum_var = 1 / np.sum(ivar)
    return (np.sum(x * ivar) * sum_var, np.sqrt(sum_var))


def build_arjl_coadds(visit_model, coadd_model, output_dir=None, batch_size=1000, n_workers=8):
    """
    Build per-source, per-telescope coadds from ARJL rest-frame visit spectra.

    DRAFT: see the open questions documented on `astra.models.arjl.ARJLCoaddedSpectrum`.

    Unlike BOSS/APOGEE, no resampling is needed here: the *RestFrame* visit models already
    have `flux`/`ivar`/`pixel_flags` shifted onto the same integer-pixel log-lambda grid
    (see `transform_to_rest` in `astra.models.arjl`), so combining visits for a source is
    just a direct call to `pixel_weighted_spectrum`.

    Memory use is bounded by `batch_size`, not by the total number of visits: visits are
    streamed from the DB one (source_pk, telescope) group at a time (the query is ordered
    so a group's visits are contiguous), and once `batch_size` groups have accumulated,
    that batch's pixel arrays are prefetched, coadded, appended to the output HDF5 file
    (which uses resizable datasets for exactly this reason), and inserted into `coadd_model`
    -- then dropped before the next batch starts. At full survey scale (millions of visits),
    loading everything up front (the previous approach) would need on the order of a
    terabyte of RAM; this instead holds only `batch_size` groups' worth of pixel data (a
    few GB at the default) at any one time.

    :param visit_model:
        One of the ARJL *RestFrame* visit spectrum models (e.g. `ARJLTHRestFrameVisitSpectrum`).

    :param coadd_model:
        The corresponding coadd model to populate (e.g. `ARJLTHCoaddedSpectrum`). Its
        `coadd_basename` class attribute determines the HDF5 filename this run writes to,
        so different `coadd_model`s sharing the same `output_dir` don't collide.

    :param output_dir: [optional]
        Directory to write the new `coadd_model.coadd_basename` file (flux/ivar/pixel_flags,
        one row per source/telescope) that the coadd model's pixel arrays will be read back
        from. Defaults to `$MWM_ASTRA/{v_astra}/spectra/star`, the same top-level directory
        that `mwmStar` files live under (but without the per-source `sdss_id_groups`
        splitting, since this is one file for the whole batch, not one per source).

    :param batch_size: [optional]
        Number of (source_pk, telescope) groups to process per batch -- see above. Also the
        number of coadd rows inserted into the database per `insert_many` call (default:
        1000).

    :param n_workers: [optional]
        Number of worker processes used to read visit pixel arrays concurrently (see
        `astra.models.arjl.prefetch_pixel_arrays`; default: 8). This is a shared filesystem
        -- keep this modest rather than maxing out cores.

    :returns:
        The total number of coadd rows written (not the row dicts themselves -- keeping
        every batch's row metadata around for the whole run isn't necessary for anything
        currently calling this, and would undercut the memory-bounded design above).
    """
    import os
    from astropy.constants import c
    from astropy import units as u
    from astra import __version__
    from astra.utils import expand_path
    from astra.models.base import database
    from astra.specutils.resampling import pixel_weighted_spectrum

    if output_dir is None:
        output_dir = expand_path(f"$MWM_ASTRA/{__version__}/spectra/star")

    # v_rad_pix_var is "stellar RV uncertainty expressed as a variance in the pixel offset"
    # (see the APMADGICS allVisit_MADGICS datamodel). Convert pixels -> km/s using the
    # log-lambda grid's pixel scale (crval=4.179, cdelt=6e-6) to get a proper e_v_rad.
    cdelt = 6e-6
    km_s_per_pixel = c.to(u.km / u.s).value * np.log(10) * cdelt

    n_pixels = 8575

    def coadd_batch(batch_groups, row_index_start):
        """Prefetch, coadd, and build DB row dicts for one batch of groups."""
        all_visits = [visit for visits in batch_groups.values() for visit in visits]
        prefetch_pixel_arrays(all_visits, field_names=("flux", "ivar", "pixel_flags"), n_workers=n_workers)

        n = len(batch_groups)
        flux = np.zeros((n, n_pixels))
        ivar = np.zeros((n, n_pixels))
        pixel_flags = np.zeros((n, n_pixels), dtype=np.int64)
        rows = []

        for i, ((source_pk, telescope), visits) in enumerate(batch_groups.items()):
            v_flux = np.array([v.flux for v in visits])
            v_ivar = np.array([v.ivar for v in visits])
            v_pixel_flags = np.array([v.pixel_flags for v in visits])

            stacked_flux, stacked_ivar, stacked_pixel_flags, *_ = pixel_weighted_spectrum(
                v_flux, v_ivar, v_pixel_flags
            )
            flux[i] = stacked_flux
            ivar[i] = stacked_ivar
            pixel_flags[i] = stacked_pixel_flags

            with np.errstate(invalid="ignore"):
                snr = stacked_flux * np.sqrt(stacked_ivar)
            finite = np.isfinite(snr) & (stacked_ivar > 0)

            v_rads = np.array([v.v_rad for v in visits])
            e_v_rads = np.sqrt(np.array([v.v_rad_pix_var for v in visits])) * km_s_per_pixel
            fibers = np.array([v.fiber for v in visits])

            finite_rv = np.isfinite(v_rads) & np.isfinite(e_v_rads) & (e_v_rads > 0)
            if np.any(finite_rv):
                v_rad, e_v_rad = weighted_average(v_rads[finite_rv], e_v_rads[finite_rv])
            else:
                v_rad, e_v_rad = (np.nan, np.nan)

            rows.append(dict(
                source_pk=source_pk,
                sdss_id=visits[0].sdss_id,
                catalogid=visits[0].catalogid,
                release=visits[0].release,
                v_arjl=visits[0].v_arjl,
                telescope=telescope,
                # A per-star crossmatch value, not per-visit, so any visit's copy will do.
                # Needed by ASPCAP's get_initial_arjl_guesses() to seed its initial guess.
                dr17_teff=visits[0].dr17_teff,
                dr17_logg=visits[0].dr17_logg,
                dr17_x_h=visits[0].dr17_x_h,
                dr17_vsini=visits[0].dr17_vsini,
                row_index=row_index_start + i,
                component_dir=output_dir,
                min_mjd=min(v.mjd for v in visits),
                max_mjd=max(v.mjd for v in visits),
                n_visits=len(visits),
                n_good_visits=len(visits),  # TODO: define 'good' (see open questions)
                snr=np.mean(snr[finite]) if np.any(finite) else np.nan,
                mean_fiber=np.mean(fibers),
                std_fiber=np.std(fibers),
                v_rad=v_rad,
                e_v_rad=e_v_rad,
                std_v_rad=np.std(v_rads),
                v_rad_flags=int(np.bitwise_or.reduce([int(v.v_rad_flags) for v in visits])),
                drp_starflag=int(np.bitwise_or.reduce([int(v.drp_starflag) for v in visits])),
            ))

        return flux, ivar, pixel_flags, rows

    def append_batch(fp, flux, ivar, pixel_flags, rows):
        """Extend the resizable output datasets with one batch's arrays and metadata."""
        old_size = fp["flux"].shape[0]
        new_size = old_size + len(rows)
        for name, data in (("flux", flux), ("ivar", ivar), ("pixel_flags", pixel_flags)):
            fp[name].resize(new_size, axis=0)
            fp[name][old_size:new_size] = data

        # Identifying metadata in the same row order as the arrays above, so the file is
        # self-describing: you can open it standalone and see which row is which star,
        # rather than that mapping only existing implicitly via `row_index` in the DB.
        fp["source_pk"].resize(new_size, axis=0)
        fp["source_pk"][old_size:new_size] = [row["source_pk"] for row in rows]
        fp["sdss_id"].resize(new_size, axis=0)
        fp["sdss_id"][old_size:new_size] = [row["sdss_id"] for row in rows]
        fp["telescope"].resize(new_size, axis=0)
        fp["telescope"][old_size:new_size] = [row["telescope"] for row in rows]

    os.makedirs(output_dir, exist_ok=True)
    database.create_tables([coadd_model])

    q = (
        visit_model
        .select()
        .where(visit_model.source_pk.is_null(False))
        .order_by(visit_model.source_pk.asc(), visit_model.telescope.asc(), visit_model.mjd.asc())
    )

    # Resizable (chunked) datasets, since the total row count isn't known up front.
    chunk_rows = min(batch_size, 512)
    with h5.File(f"{output_dir}/{coadd_model.coadd_basename}", "w") as fp:
        fp.create_dataset("flux", shape=(0, n_pixels), maxshape=(None, n_pixels), chunks=(chunk_rows, n_pixels), dtype="f8")
        fp.create_dataset("ivar", shape=(0, n_pixels), maxshape=(None, n_pixels), chunks=(chunk_rows, n_pixels), dtype="f8")
        fp.create_dataset("pixel_flags", shape=(0, n_pixels), maxshape=(None, n_pixels), chunks=(chunk_rows, n_pixels), dtype="i8")
        fp.create_dataset("source_pk", shape=(0,), maxshape=(None,), dtype="i8")
        fp.create_dataset("sdss_id", shape=(0,), maxshape=(None,), dtype="i8")
        fp.create_dataset("telescope", shape=(0,), maxshape=(None,), dtype=h5.string_dtype(encoding="utf-8"))

        row_index = 0
        current_key, current_visits = None, None
        batch_groups = OrderedDict()
        pb_groups = tqdm(desc="Coadding groups", unit="group")

        def flush(batch_groups):
            nonlocal row_index
            flux, ivar, pixel_flags, rows = coadd_batch(batch_groups, row_index)
            append_batch(fp, flux, ivar, pixel_flags, rows)
            for spectrum_pk, row in enumerate_new_spectrum_pks(rows):
                row["spectrum_pk"] = spectrum_pk
            with database.atomic():
                coadd_model.insert_many(rows).execute()
            row_index += len(rows)
            pb_groups.update(len(rows))

        for visit in tqdm(q.iterator(), desc=f"Loading {visit_model.__name__}"):
            key = (visit.source_pk, visit.telescope)
            if key != current_key:
                if current_key is not None:
                    batch_groups[current_key] = current_visits
                    if len(batch_groups) >= batch_size:
                        flush(batch_groups)
                        batch_groups = OrderedDict()
                current_key, current_visits = key, []
            current_visits.append(visit)

        if current_key is not None:
            batch_groups[current_key] = current_visits
        if batch_groups:
            flush(batch_groups)

        pb_groups.close()

    return row_index
