import os
from typing import OrderedDict, Optional
from astropy.io import fits
from airflow.models.baseoperator import BaseOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from sdss_access import SDSSPath
from astra.database.apogee_drpdb import Star, Visit
from astra.database.astradb import database, DataProduct, Source, SourceDataProduct
from astra import log
from astra.utils import flatten, expand_path
from astropy.time import Time
from tqdm import tqdm
from time import time
import numpy as np

from astropy.table import Table

from functools import lru_cache

from peewee import fn, JOIN


@lru_cache
def path_instance(release):
    return SDSSPath(release=release)


@lru_cache
def lookup_keys(release, filetype):
    return path_instance(release).lookup_keys(filetype)


def get_or_create_data_product_from_apogee_drpdb(
    origin,
    release=None,
    filetype=None,
):
    """
    Get or create a data product entry in the astra database from an apogee_drpdb origin.

    :param release: [optional]
        Supply a release. Otherwise this will be read from `origin.release`.

    :param filetype: [optional]
        The filetype of the data product. If `None`, this will be read from `origin.filetype`.
    """
    release = release or origin.release
    filetype = filetype or origin.filetype

    kwds = {k: getattr(origin, k) for k in lookup_keys(release, filetype)}
    if "field" in kwds:
        kwds["field"] = kwds["field"].strip()
    path = path_instance(release).full(filetype, **kwds)
    if not os.path.exists(path):
        error = {"detail": path, "origin": origin, "reason": "File does not exist"}
        return (False, error)

    # TODO: If we the data product already exists, check that the size matches
    # with database.atomic() as txn:
    if True:
        data_product, data_product_created = DataProduct.get_or_create(
            release=release,
            filetype=filetype,
            kwargs=kwds,
        )
        # if not data_product_created:
        #    # Update the size
        #    if data_product.size != size:
        #        log.info(f"Updating size of data product {data_product} from {data_product.size} to {size}")
        #        data_product.size = size
        #        data_product.save()

        # Have to make sure the source exists before we link SourceDataProduct..
        source, _ = Source.get_or_create(catalogid=origin.catalogid)
        SourceDataProduct.get_or_create(
            source_id=origin.catalogid, data_product=data_product
        )

    result = dict(data_product_id=data_product.id, source_id=origin.catalogid)
    return (True, result)


def get_apvisit_metadata(apstar_data_product):
    # It's stupid to open the file for this, but I can't yet seem to find the magic to
    # match visits to apstar files in apogee_drpdb because visits are excluded from the
    # stack for a variety of reasons.
    apogee_id = apstar_data_product.kwargs["obj"]
    meta = []
    with fits.open(apstar_data_product.path) as image:
        N, P = image[1].data.shape
        if N > 2:
            meta.extend([{"note": "stacked_0"}, {"note": "stacked_1"}])
            indices = range(1, N - 1)
        else:
            indices = [1]

        for i in indices:
            # get visit pk
            visit = (
                Visit.select()
                .where(
                    (Visit.apogee_id == apogee_id)
                    & (Visit.dateobs == image[0].header[f"DATE{i:.0f}"])
                )
                .first()
            )

            visit_pk = None if visit is None else visit.pk
            meta.append(
                {
                    "visit_pk": visit_pk,
                    "fiber": image[0].header[f"FIBER{i:.0f}"],
                    "date_obs": image[0].header[f"DATE{i:.0f}"],
                    "jd": image[0].header[f"JD{i:.0f}"],
                    "bc": image[0].header[f"BC{i:.0f}"],
                    "vrad": image[0].header[f"VRAD{i:.0f}"],
                    "vhbary": image[0].header[f"VHBARY{i:.0f}"],
                    # 2022-04-28: STARFLAG is not present in all ApStar files
                    # "starflag": image[0].header[f"STARFLAG{i:.0f}"]
                }
            )

    return meta


class BossSpectrumOperator(BaseOperator):
    """
    A base operator for working with SDSS-V BOSS spectrum data products.

    This operator will generate task instances based on BOSS reduced data products it finds that were
    *observed* in the operator execution period.
    """

    ui_color = "#A0B9D9"

    def __init__(
        self,
        *,
        release: Optional[str] = None,
        filetype: Optional[str] = "specFull",
        run2d: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.release = release
        self.filetype = filetype
        self.run2d = run2d
        return None

    def execute(self, context):

        data = Table.read(
            expand_path(f"$BOSS_SPECTRO_REDUX/{self.run2d}/spAll-{self.run2d}.fits")
        )

        prev_ds, ds = (context["prev_ds"], context["ds"])
        if prev_ds is None:
            # Schedule once only
            mask = np.ones(len(data), dtype=bool)
        else:
            mjd_start, mjd_end = list(map(lambda x: Time(x).mjd, (prev_ds, ds)))
            mask = (mjd_end > data["MJD"]) & (data["MJD"] >= mjd_start)

        N = sum(mask)
        if N == 0:
            raise AirflowSkipException(
                f"No products for {self} observed between {prev_ds} and {ds}"
            )

        errors = []
        ids = []
        for row in data[mask]:
            catalogid = row["CATALOGID"]
            # From Sean Morrison:
            # Note if you have looked at the v604 outputs we used a p at the end of the plate ID like 15000p,
            # this was to uncertainty about if the plateids were going to be unique to the fps, but in v609
            # this will change since the ids are unique fieldid<16000 is plates, > is fps.
            isplate = "p" if row["FIELD"] < 16_000 else ""
            kwds = dict(
                fieldid=f"{row['FIELD']:0>6.0f}",
                mjd=int(row["MJD"]),
                catalogid=int(catalogid),
                run2d=row["RUN2D"],
                isplate=isplate,
            )
            path = path_instance(self.release).full(self.filetype, **kwds)
            if not os.path.exists(path):
                error = {"detail": path, "origin": row, "reason": "File does not exist"}
                errors.append(error)
                continue

            # We need the ZWARNING metadata downstream.
            with database.atomic() as txn:
                data_product, _ = DataProduct.get_or_create(
                    release=self.release,
                    filetype=self.filetype,
                    kwargs=kwds,
                )
                # Update the ZWARNING metadata, even if this data product already exists.
                data_product.metadata = {"ZWARNING": row["ZWARNING"]}
                data_product.save()

                source, _ = Source.get_or_create(catalogid=catalogid)
                SourceDataProduct.get_or_create(
                    source=source, data_product=data_product
                )

            log.info(f"Data product {data_product} matched to {source}")
            ids.append(data_product.id)

        N = sum(mask)
        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        if len(errors):
            log.warning(f"Encountered {len(errors)} errors:")
            for error in errors:
                log.warning(f"{error['reason']}: {error['detail']}")

        if len(errors) == N:
            raise AirflowSkipException(f"{N}/{N} data products had errors")
        return ids


class ApVisitOperator(BaseOperator):
    def __init__(
        self,
        apred: Optional[str] = "daily",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.apred = apred
        return None

    def execute(self, context):

        prev_ds, ds = (context["prev_ds"], context["ds"])

        if prev_ds is None:
            expression = Visit.created > ds
        else:
            expression = Visit.created.between(prev_ds, ds)

        # catalogid < 0 apparently implies unassigned fibre
        # TODO: check if they are really unassigned ornot .
        # apogee-pipeline 470] off target robots in reductions

        # Through inspection I found that all the visit (Database) records with missing paths
        # tended to be those without any sdssv_apogee_target0 value.

        log.info(f"{self} is looking for products created between {prev_ds} and {ds}")
        q = (
            Visit.select()
            .where(
                expression & (Visit.apred_vers == self.apred) & (Visit.catalogid > 0)
            )
            .order_by(Visit.pk.asc())
        )

        N = q.count()
        if N == 0:
            raise AirflowSkipException(
                f"No products for {self} created between {prev_ds} and {ds}"
            )

        log.info(f"Found {N} rows.")

        ids = []
        errors = []
        for origin in q:
            success, result = get_or_create_data_product_from_apogee_drpdb(origin)
            if success:
                log.debug(
                    "Data product {data_product_id} matched to {source_id}".format(
                        **result
                    )
                )
                ids.append(result["data_product_id"])
            else:
                log.warning("{reason} ({detail}) from {origin}".format(**result))
                errors.append(result)

        log.info(f"Created {len(ids)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowSkipException(f"{N}/{N} data products had errors")
        return ids


class CartonOperator(BaseOperator):
    """
    Filter upstream data model products and only keep those that are matched to a specific SDSS-V carton.

    This operator requires a data model operator directly preceeding it in a DAG (e.g., an ApStarOperator).
    """

    ui_color = "#FEA83A"

    def __init__(
        self,
        *,
        cartons=None,
        programs=None,
        mappers=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cartons = cartons
        self.programs = programs
        self.mappers = mappers

    def execute(
        self,
        context,
    ):
        # It's bad practice to import here, but we don't want to depend on catalogdb to be accessible from
        # the outer scope of this operator.
        from astra.sdss.catalog import filter_sources

        ti, task = (context["ti"], context["task"])
        data_product_ids = tuple(
            set(flatten(ti.xcom_pull(task_ids=task.upstream_task_ids)))
        )

        log.info(f"Data product IDs ({len(data_product_ids)}): {data_product_ids}")

        log.info(f"Matching on:")
        log.info(f"     Cartons: {self.cartons}")
        log.info(f"     Programs: {self.programs}")
        log.info(f"     Mappers: {self.mappers}")

        # Retrieve the source catalog identifiers for these data products.
        q = (
            Source.select(Source.catalogid, DataProduct.id)
            .join(SourceDataProduct)
            .join(DataProduct)
            .where(DataProduct.id.in_(data_product_ids))
            .tuples()
        )
        lookup = {c_id: dp_id for c_id, dp_id in q}
        log.info(f"Lookup table contains {len(lookup)} matched entries.")

        # Only return the data product ids that match.
        keep = filter_sources(
            tuple(lookup.keys()),
            cartons=self.cartons,
            programs=self.programs,
            mappers=self.mappers,
        )
        log.info(f"Keeping {len(keep)} matched.")

        keep_data_product_ids = [lookup[c_id] for c_id in keep]
        if len(keep_data_product_ids) == 0:
            raise AirflowSkipException(
                f"None of the sources of upstream data products matched."
            )

        log.info(f"{keep_data_product_ids}")
        # Return the associated data product identifiers from the lookup.
        return keep_data_product_ids


class MilkyWayMapperOperator(BaseOperator):

    """
    Filter upstream data products and only keep those that are relevant to Milky Way Mapper.

    See code (here) and https://wiki.sdss.org/display/OPS/Open+Fiber+Implementation for details.
    """

    # From the SDSS database:
    # sdss5db=> set search_path to targetdb;
    # sdss5db=> select distinct(carton), program from carton order by carton asc;
    cartons = (
        # "bhm_aqmes_bonus_bright",                     # bhm_filler
        # "bhm_aqmes_bonus-bright",                     # bhm_filler
        # "bhm_aqmes_bonus_core",                       # bhm_filler
        # "bhm_aqmes_bonus-dark",                       # bhm_filler
        # "bhm_aqmes_bonus_faint",                      # bhm_filler
        # "bhm_aqmes_med",                              # bhm_aqmes
        # "bhm_aqmes_med_faint",                        # bhm_filler
        # "bhm_aqmes_med-faint",                        # bhm_filler
        # "bhm_aqmes_wide2",                            # bhm_aqmes
        # "bhm_aqmes_wide2_faint",                      # bhm_filler
        # "bhm_aqmes_wide2-faint",                      # bhm_filler
        # "bhm_aqmes_wide3",                            # bhm_aqmes
        # "bhm_aqmes_wide3-faint",                      # bhm_filler
        # "bhm_colr_galaxies_lsdr8",                    # bhm_filler
        # "bhm_csc_apogee",                             # bhm_csc
        # "bhm_csc_boss",                               # bhm_csc
        # "bhm_csc_boss_bright",                        # bhm_csc
        # "bhm_csc_boss-bright",                        # bhm_csc
        # "bhm_csc_boss_dark",                          # bhm_csc
        # "bhm_csc_boss-dark",                          # bhm_csc
        # "bhm_gua_bright",                             # bhm_filler
        # "bhm_gua_dark",                               # bhm_filler
        # "bhm_rm_ancillary",                           # bhm_rm
        # "bhm_rm_core",                                # bhm_rm
        # "bhm_rm_known_spec",                          # bhm_rm
        # "bhm_rm_known-spec",                          # bhm_rm
        # "bhm_rm_var",                                 # bhm_rm
        # "bhm_spiders_agn-efeds",                      # bhm_spiders
        # "bhm_spiders_agn_efeds_stragglers",           # bhm_spiders
        # "bhm_spiders_agn_gaiadr2",                    # bhm_spiders
        # "bhm_spiders_agn_lsdr8",                      # bhm_spiders
        # "bhm_spiders_agn_ps1dr2",                     # bhm_spiders
        # "bhm_spiders_agn_sep",                        # bhm_spiders
        # "bhm_spiders_agn_skymapperdr2",               # bhm_spiders
        # "bhm_spiders_agn_supercosmos",                # bhm_spiders
        # "bhm_spiders_clusters-efeds-erosita",         # bhm_spiders
        # "bhm_spiders_clusters-efeds-hsc-redmapper",   # bhm_spiders
        # "bhm_spiders_clusters-efeds-ls-redmapper",    # bhm_spiders
        # "bhm_spiders_clusters-efeds-sdss-redmapper",  # bhm_spiders
        # "bhm_spiders_clusters_efeds_stragglers",      # bhm_spiders
        # "bhm_spiders_clusters_lsdr8",                 # bhm_spiders
        # "bhm_spiders_clusters_ps1dr2",                # bhm_spiders
        "comm_pleiades",  # commissioning
        "comm_spectrophoto",  # open_fiber
        # "manual_bhm_spiders_comm",                    # commissioning
        "manual_bright_target",  # mwm_ob
        "manual_bright_target_offsets_1",  # commissioning
        "manual_bright_target_offsets_1_g13",  # commissioning
        "manual_bright_target_offsets_2",  # commissioning
        "manual_bright_target_offsets_2_g13",  # commissioning
        "manual_bright_target_offsets_3",  # mwm_ob
        "manual_bright_targets",  # commissioning
        "manual_bright_targets_g13",  # commissioning
        "manual_bright_targets_g13_offset_fixed_1",  # commissioning
        "manual_bright_targets_g13_offset_fixed_2",  # commissioning
        "manual_bright_targets_g13_offset_fixed_3",  # commissioning
        "manual_bright_targets_g13_offset_fixed_4",  # commissioning
        "manual_bright_targets_g13_offset_fixed_5",  # commissioning
        "manual_bright_targets_g13_offset_fixed_6",  # commissioning
        "manual_bright_targets_g13_offset_fixed_7",  # commissioning
        "manual_fps_position_stars",  # commissioning
        "manual_fps_position_stars_10",  # commissioning
        "manual_fps_position_stars_apogee_10",  # commissioning
        "manual_mwm_tess_ob",  # mwm_tessob
        "manual_nsbh_apogee",  # mwm_cb
        "manual_nsbh_boss",  # mwm_cb
        "manual_offset_mwmhalo_off00",  # commissioning
        "manual_offset_mwmhalo_off05",  # commissioning
        "manual_offset_mwmhalo_off10",  # commissioning
        "manual_offset_mwmhalo_off20",  # commissioning
        "manual_offset_mwmhalo_off30",  # commissioning
        "manual_offset_mwmhalo_offa",  # commissioning
        "manual_offset_mwmhalo_offb",  # commissioning
        "manual_planet_ca_legacy_v0",  # mwm_planet
        "manual_planet_gaia_astrometry_v0",  # mwm_planet
        "manual_planet_gpi_v0",  # mwm_planet
        "manual_planet_harps_v0",  # mwm_planet
        "manual_planet_known_v0",  # mwm_planet
        "manual_planet_sophie_v0",  # mwm_planet
        "manual_planet_sphere_v0",  # mwm_planet
        "manual_planet_tess_eb_v0",  # mwm_planet
        "manual_planet_tess_pc_v0",  # mwm_planet
        "manual_planet_transiting_bd_v0",  # mwm_planet
        "manual_validation_apogee",  # mwm_validation
        "manual_validation_boss",  # mwm_validation
        "manual_validation_rv",  # mwm_validation
        "mwm_cb_300pc",  # mwm_cb
        "mwm_cb_300pc_apogee",  # mwm_cb
        "mwm_cb_300pc_boss",  # mwm_cb
        "mwm_cb_cvcandidates",  # mwm_cb
        "mwm_cb_cvcandidates_apogee",  # mwm_cb
        "mwm_cb_cvcandidates_boss",  # mwm_cb
        "mwm_cb_gaiagalex",  # mwm_cb
        "mwm_cb_gaiagalex_apogee",  # mwm_cb
        "mwm_cb_gaiagalex_boss",  # mwm_cb
        "mwm_cb_uvex1",  # mwm_cb
        "mwm_cb_uvex2",  # mwm_cb
        "mwm_cb_uvex3",  # mwm_cb
        "mwm_cb_uvex4",  # mwm_cb
        "mwm_cb_uvex5",  # mwm_cb
        "mwm_dust_core",  # mwm_dust
        "mwm_erosita_compact_gen",  # mwm_erosita
        "mwm_erosita_compact_var",  # mwm_erosita
        "mwm_erosita_stars",  # mwm_erosita
        "mwm_galactic_core",  # mwm_galactic
        "mwm_gg_core",  # mwm_gg
        "mwm_halo_bb",  # mwm_halo
        "mwm_halo_bb_apogee",  # mwm_filler
        "mwm_halo_bb_boss",  # mwm_filler
        "mwm_halo_sm",  # mwm_halo
        "mwm_halo_sm_apogee",  # mwm_filler
        "mwm_halo_sm_boss",  # mwm_filler
        "mwm_legacy_ir2opt",  # mwm_legacy
        "mwm_ob_cepheids",  # mwm_ob
        "mwm_ob_core",  # mwm_ob
        "mwm_planet_tess",  # mwm_planet
        "mwm_rv_long_bplates",  # mwm_rv
        "mwm_rv_long-bplates",  # mwm_rv
        "mwm_rv_long_fps",  # mwm_rv
        "mwm_rv_long-fps",  # mwm_rv
        "mwm_rv_long_rm",  # mwm_rv
        "mwm_rv_long-rm",  # mwm_rv
        "mwm_rv_short_bplates",  # mwm_rv
        "mwm_rv_short-bplates",  # mwm_rv
        "mwm_rv_short_fps",  # mwm_rv
        "mwm_rv_short-fps",  # mwm_rv
        "mwm_rv_short_rm",  # mwm_rv
        "mwm_rv_short-rm",  # mwm_rv
        "mwm_snc_100pc",  # mwm_snc
        "mwm_snc_100pc_apogee",  # mwm_snc
        "mwm_snc_100pc_boss",  # mwm_snc
        "mwm_snc_250pc",  # mwm_snc
        "mwm_snc_250pc_apogee",  # mwm_snc
        "mwm_snc_250pc_boss",  # mwm_snc
        "mwm_tess_ob",  # mwm_tess_ob
        "mwm_tess_ob",  # mwm_tessob
        "mwm_tess_planet",  # mwm_planet
        "mwm_tessrgb_core",  # mwm_tessrgb
        "mwm_wd_core",  # mwm_wd
        "mwm_yso_cluster",  # mwm_yso
        "mwm_yso_cluster_apogee",  # mwm_yso
        "mwm_yso_cluster_boss",  # mwm_yso
        "mwm_yso_cmz",  # mwm_yso
        "mwm_yso_cmz_apogee",  # mwm_yso
        "mwm_yso_disk_apogee",  # mwm_yso
        "mwm_yso_disk_boss",  # mwm_yso
        "mwm_yso_embedded_apogee",  # mwm_yso
        "mwm_yso_nebula_apogee",  # mwm_yso
        "mwm_yso_ob",  # mwm_yso
        "mwm_yso_ob_apogee",  # mwm_yso
        "mwm_yso_ob_boss",  # mwm_yso
        "mwm_yso_pms_apogee",  # mwm_yso
        "mwm_yso_pms_boss",  # mwm_yso
        "mwm_yso_s1",  # mwm_yso
        "mwm_yso_s2",  # mwm_yso
        "mwm_yso_s2-5",  # mwm_yso
        "mwm_yso_s3",  # mwm_yso
        "mwm_yso_variable_apogee",  # mwm_yso
        "mwm_yso_variable_boss",  # mwm_yso
        "openfiberstargets_test",  # open_fiber
        "openfibertargets_nov2020_10",  # open_fiber
        "openfibertargets_nov2020_1000",  # open_fiber
        "openfibertargets_nov2020_1001a",  # open_fiber
        "openfibertargets_nov2020_1001b",  # open_fiber
        # "openfibertargets_nov2020_11",                # open_fiber --> QSO/AGN
        "openfibertargets_nov2020_12",  # open_fiber
        "openfibertargets_nov2020_14",  # open_fiber
        "openfibertargets_nov2020_15",  # open_fiber
        "openfibertargets_nov2020_17",  # open_fiber
        # "openfibertargets_nov2020_18",                # open_fiber --> QSO/AGN
        "openfibertargets_nov2020_19a",  # open_fiber
        "openfibertargets_nov2020_19b",  # open_fiber
        "openfibertargets_nov2020_19c",  # open_fiber
        # "openfibertargets_nov2020_20",               # open_fiber --> SN host
        "openfibertargets_nov2020_22",  # open_fiber
        "openfibertargets_nov2020_24",  # open_fiber
        "openfibertargets_nov2020_25",  # open_fiber
        # "openfibertargets_nov2020_26",                # open_fiber --> QSO/AGN
        # "openfibertargets_nov2020_27",                # open_fiber --> QSO/AGN
        "openfibertargets_nov2020_28a",  # open_fiber
        "openfibertargets_nov2020_28b",  # open_fiber
        "openfibertargets_nov2020_28c",  # open_fiber
        "openfibertargets_nov2020_29",  # open_fiber
        "openfibertargets_nov2020_3",  # open_fiber
        # "openfibertargets_nov2020_30",                # open_fiber --> QSO/AGN
        "openfibertargets_nov2020_31",  # open_fiber
        "openfibertargets_nov2020_32",  # open_fiber
        # "openfibertargets_nov2020_33",                # open_fiber --> QSO/AGN
        "openfibertargets_nov2020_34a",  # open_fiber
        "openfibertargets_nov2020_34b",  # open_fiber
        "openfibertargets_nov2020_35a",  # open_fiber
        "openfibertargets_nov2020_35b",  # open_fiber
        "openfibertargets_nov2020_35c",  # open_fiber
        "openfibertargets_nov2020_46",  # open_fiber
        "openfibertargets_nov2020_47a",  # open_fiber
        "openfibertargets_nov2020_47b",  # open_fiber
        "openfibertargets_nov2020_47c",  # open_fiber
        "openfibertargets_nov2020_47d",  # open_fiber
        "openfibertargets_nov2020_47e",  # open_fiber
        "openfibertargets_nov2020_5",  # open_fiber
        "openfibertargets_nov2020_6a",  # open_fiber
        "openfibertargets_nov2020_6b",  # open_fiber
        "openfibertargets_nov2020_6c",  # open_fiber
        "openfibertargets_nov2020_8",  # open_fiber
        "openfibertargets_nov2020_9",  # open_fiber
        "ops_2mass_psc_brightneighbors",  # ops
        "ops_apogee_stds",  # ops_std
        "ops_gaia_brightneighbors",  # ops
        "ops_std_apogee",  # ops_std
        "ops_std_boss",  # ops_std
        "ops_std_boss_gdr2",  # ops_std
        "ops_std_boss_lsdr8",  # ops_std
        "ops_std_boss_ps1dr2",  # ops_std
        "ops_std_boss_red",  # ops_std
        "ops_std_boss-red",  # ops_std
        "ops_std_boss_tic",  # ops_std
        "ops_std_eboss",  # ops_std
        "ops_tycho2_brightneighbors",  # ops
    )

    def __init__(self, *, return_id_type="data_product", **kwargs) -> None:
        super().__init__(**kwargs)
        available_id_types = ("data_product", "source")
        if return_id_type not in available_id_types:
            raise ValueError(
                f"Return identifier type must be one of {available_id_types} (not {return_id_type})"
            )
        self.return_id_type = return_id_type
        return None

    def execute(self, context):

        ti, task = (context["ti"], context["task"])
        data_product_ids = tuple(
            set(flatten(ti.xcom_pull(task_ids=task.upstream_task_ids)))
        )

        log.info(f"Data product IDs ({len(data_product_ids)}): {data_product_ids}")

        # Retrieve the source catalog identifiers for these data products.
        q = (
            Source.select(Source.catalogid, DataProduct.id)
            .join(SourceDataProduct)
            .join(DataProduct)
            .where(DataProduct.id.in_(data_product_ids))
            .tuples()
        )
        lookup = {}
        for catalogid, data_product_id in q:
            lookup.setdefault(catalogid, [])
            lookup[catalogid].append(data_product_id)

        catalogids = tuple(lookup.keys())

        log.info(f"Lookup table contains {len(catalogids)} sources.")

        # TODO: This should be re-organised with other SDSS stuff, away from just "operators"
        from peewee import JOIN
        from sdssdb.peewee.sdss5db import database as sdss5_database

        sdss5_database.set_profile(
            "operations"
        )  # TODO: HOW CAN WE SET THIS AS DEFAULT!?!

        from sdssdb.peewee.sdss5db.targetdb import Target, CartonToTarget, Carton

        q = (
            Target.select(Target.catalogid)
            .distinct()
            .join(CartonToTarget)
            .join(Carton)
            .where(Target.catalogid.in_(catalogids) & Carton.carton.in_(self.cartons))
        )

        keep_catalogids = flatten(q.tuples())
        log.info(f"Keeping {len(keep_catalogids)} sources matched.")

        if len(keep_catalogids) == 0:
            raise AirflowSkipException(
                f"None of the sources of upstream data products matched."
            )

        if self.return_id_type == "data_product":
            # Return the associated data product identifiers from the lookup.
            return flatten([lookup[c_id] for c_id in keep_catalogids])

        elif self.return_id_type == "source":
            # Return source identifiers
            return keep_catalogids


class ApStarOperator(BaseOperator):
    """
    Generate data model products in the Astra database for all new ApStar
    files produced since the operator was last executed.

    :param releases: [optional]
        The relevant SDSS data releases. If `None` is given then this will be inferred based on
        the execution date.
    """

    ui_color = "#ffb09c"

    def execute(
        self,
        context,
        where=(Star.ngoodvisits > 0) & (Star.catalogid > 0),
        latest_only=True,
    ):
        prev_ds, ds = (context["prev_ds"], context["ds"])

        log.info(
            f"{self} is looking for {'only latest' if latest_only else 'any'} ApStar products created between {prev_ds} and {ds}"
        )

        if latest_only:
            StarAlias = Star.alias()
            sq = (
                StarAlias.select(
                    StarAlias.catalogid, fn.MAX(StarAlias.created).alias("max_created")
                )
                .group_by(StarAlias.catalogid)
                .alias("sq")
            )
            q = (
                Star.select()
                .where(sq.c.max_created.between(prev_ds, ds) & where)
                .join(
                    sq,
                    on=(Star.created == sq.c.max_created)
                    & (Star.catalogid == sq.c.catalogid),
                )
            )
        else:
            q = Star.select().where(Star.created.between(prev_ds, ds) & where)

        N = q.count()
        if N == 0:
            raise AirflowSkipException(
                f"No products for {self} created between {prev_ds} and {ds}"
            )

        ids = []
        errors = []
        for origin in q:
            success, result = get_or_create_data_product_from_apogee_drpdb(origin)
            if success:
                log.info(
                    "Data product {data_product_id} matched to {source_id}".format(
                        **result
                    )
                )
                ids.append(result["data_product_id"])
            else:
                log.warning("{reason} ({detail}) from {origin}".format(**result))
                errors.append(result)

        log.info(f"Found {N} rows.")
        log.info(f"Created {len(ids)} data products.")
        log.info(f"Encountered {len(errors)} errors.")
        if len(errors) == N:
            raise AirflowFailException(f"{N}/{N} data products had errors")
        return ids
