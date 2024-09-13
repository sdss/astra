
from astra.models import AstroNN, ApogeeNet, Slam, Source, SnowWhite, ASPCAP, BossNet, Spectrum, ApogeeCoaddedSpectrumInApStar
from astra.models import BossVisitSpectrum, ApogeeVisitSpectrum
from astra.models.mwm import (ApogeeCombinedSpectrum, BossCombinedSpectrum)
from astra.models.best import MWMBest
from astra.products.mwm import DEFAULT_MWM_WHERE
from astra import task, __version__, log

from typing import Iterable, Sequence
from peewee import fn, Value

# Q: Include MDwarfType classifications? for all things?
#    https://sdss-wiki.atlassian.net/wiki/spaces/SCI/pages/177111227/M-dwarf+IPL-3+Parameter+Flags
#    Nothing K-type, K7.5 probably OK, but on the edge.
#    Require M_G < 5.553  



# Q: Unique results by SOURCE, or by SOURCE + TELECOPE?

@task
def best(apred=("dr17", "1.3"), run2d="v6_1_3") -> Iterable[MWMBest]:

    if isinstance(apred, str):
        apred = [apred]
    if isinstance(run2d, str):
        run2d = [run2d]

    query_order = [
        # Query Snow White, since we only ran that on white dwarf cartons
        # Q: Do we want to place any constraints on the Snow White model? Eg only 'good' results?
        # Q: Do we want to include any CORV stellar parameters instead of Snow White?
        # Q: Do we want to include any CORV radial velocities instead of Snow White?
        (
            "Query Snow White",
            _query_snow_white_coadd, {
                "where": BossCombinedSpectrum.run2d.in_(run2d)
            }
        ),
            
        # Query BOSSNet for HOT stars only, where results are deemed good.
        (
            "Query BOSSNet for hot stars",
            _query_boss_net_coadd, {
                "where": (
                    (Source.assigned_to_program("mwm_ob"))
                &   (BossNet.result_flags == 0)
                &   BossCombinedSpectrum.run2d.in_(run2d)
                )                
        }),

        # Q: Query HotPayne for remaining HOT stars?

        # Query any ASPCAP for remaining HOT stars.
        # Q: Any cuts to make on ASPCAP results here?
        (
            "Query ASPCAP for hot stars",
            _query_aspcap_coadd, {
                "where": (
                    (Source.assigned_to_program("mwm_ob"))
                &   ApogeeCoaddedSpectrumInApStar.apred.in_(apred)
                )
            }
        ),

        # Query sensible APOGEENet results from the YSOs        
        (
            "Query APOGEENet for YSOs",
            _query_apogee_net_coadd, {
                "where": (
                    Source.assigned_to_program("mwm_yso")
                &   ApogeeCoaddedSpectrumInApStar.apred.in_(apred)
                &   (ApogeeNet.result_flags == 0)
                )
            }
        ),

        # Query sensible BOSSNet results from the YSOs.
        (
            "Query BOSSNet for YSOs",
            _query_boss_net_coadd, {
                "where": (
                    Source.assigned_to_program("mwm_yso")
                &   BossCombinedSpectrum.run2d.in_(run2d)
                &   (BossNet.result_flags == 0)
                )
            }
        ),

        # Query SLAM for M-dwarfs.
        (
            "Query SLAM for M-dwarfs",
            _query_slam_coadd, {
                "where": (
                    BossCombinedSpectrum.run2d.in_(run2d)
                &   (Slam.result_flags == 0)
                )
            }
        ),
        
        # Query sensible ASPCAP results for remaining stars.
        (
            "Query sensible ASPCAP results for remaining stars",
            _query_aspcap_coadd, {
                "where": (
                    ~ASPCAP.flag_bad
                &   ApogeeCoaddedSpectrumInApStar.apred.in_(apred)
                )
            }
        ),

        # Query sensible AstroNN results for remaining stars.
        (
            "Query sensible AstroNN results for remaining stars",
            _query_astro_nn_coadd, {
                "where": (
                    ApogeeCoaddedSpectrumInApStar.apred.in_(apred)
                &   (AstroNN.result_flags == 0)
                )
            }
        ),
        
        # Query APOGEENet results for remaining stars.
        (
            "Query APOGEENet for remaining stars",
            _query_apogee_net_coadd, {
                "where": (
                    ApogeeCoaddedSpectrumInApStar.apred.in_(apred)
                &   (ApogeeNet.result_flags == 0)
                )
            }
        ),

        # Query BOSSNet results for remaining stars.
        (
            "Query BOSSNet for remaining stars",
            _query_boss_net_coadd, {
                "where": (
                    BossCombinedSpectrum.run2d.in_(run2d)
                &   (BossNet.result_flags == 0)
                )
            }
        ),        

        # Query APOGEENet for anything else.
        (
            "Query APOGEENet for remaining stars",
            _query_apogee_net_coadd, {
                "where": (
                    ApogeeCoaddedSpectrumInApStar.apred.in_(apred)
                )
            }
        ),
        
        # Query BOSSNet for anything else
        (
            "Query BOSSNet for remaining stars",
            _query_boss_net_coadd, {
                "where": (
                    BossCombinedSpectrum.run2d.in_(run2d)
                )
            }
        ),                

        # Now to include auxillary information:

        # - Include MDwarfType classifications for things that are classified as M and have M_G < 5.55?

        # - Include LineForest information 

        # - Include CORV information?

        # Q: Get all remaining sources with any spectra meeting the apred/run2d.

    ]

    total, rows, source_pks = (0, [], set())
    for desc, f, kwds in query_order:
        log.info(f"{desc} ({f.__name__})")

        q = f(exclude_source_pks=source_pks, **kwds)
        for n, row in enumerate(q, start=1):
            source_pks.add(row["source_pk"])
            yield MWMBest(**row)
            rows.append(row)

        total += n
        log.info(f" -> Found {n:,} rows from {f.__name__} ({total:,} total so far)")

    # Get all the source primary keys that we might have anything for.
    queries = [
        (
            BossVisitSpectrum
            .select(
                BossVisitSpectrum.source_pk
            )
            .distinct(BossVisitSpectrum.source_pk)
            .join(Source)
            .where(
                DEFAULT_MWM_WHERE
            &   BossVisitSpectrum.run2d.in_(run2d)
            )
            .tuples()
        ),
        (
            ApogeeVisitSpectrum
            .select(
                ApogeeVisitSpectrum.source_pk
            )
            .distinct(ApogeeVisitSpectrum.source_pk)
            .where(
                ApogeeVisitSpectrum.apred.in_(apred)
            )
            .tuples()
        )
    ]
    expected_source_pks = set()
    for q in queries:
        for r in q:
            expected_source_pks.add(r[0])
    
    log.info(f"Found {len(expected_source_pks):,} expected sources")

    source_pks_with_no_results = expected_source_pks.difference(source_pks)
    log.info(f"Found {len(source_pks_with_no_results):,} sources with no results")
    

    raise a


SPECTRUM_FIELDS = {
    ApogeeCoaddedSpectrumInApStar: (
        ApogeeCoaddedSpectrumInApStar.release,
        ApogeeCoaddedSpectrumInApStar.filetype,
        ApogeeCoaddedSpectrumInApStar.apred,
        ApogeeCoaddedSpectrumInApStar.apstar,
        ApogeeCoaddedSpectrumInApStar.obj,
        ApogeeCoaddedSpectrumInApStar.telescope,
        ApogeeCoaddedSpectrumInApStar.field,
        ApogeeCoaddedSpectrumInApStar.prefix,

        ApogeeCoaddedSpectrumInApStar.min_mjd,
        ApogeeCoaddedSpectrumInApStar.max_mjd,
        ApogeeCoaddedSpectrumInApStar.n_visits,
        ApogeeCoaddedSpectrumInApStar.n_good_visits,
        ApogeeCoaddedSpectrumInApStar.n_good_rvs,

        ApogeeCoaddedSpectrumInApStar.snr,
        ApogeeCoaddedSpectrumInApStar.mean_fiber,
        ApogeeCoaddedSpectrumInApStar.std_fiber,
        ApogeeCoaddedSpectrumInApStar.spectrum_flags,

        ApogeeCoaddedSpectrumInApStar.v_rad,
        ApogeeCoaddedSpectrumInApStar.e_v_rad,
        ApogeeCoaddedSpectrumInApStar.std_v_rad,
        ApogeeCoaddedSpectrumInApStar.median_e_v_rad,

        ApogeeCoaddedSpectrumInApStar.doppler_teff,
        ApogeeCoaddedSpectrumInApStar.doppler_e_teff,
        ApogeeCoaddedSpectrumInApStar.doppler_logg,
        ApogeeCoaddedSpectrumInApStar.doppler_e_logg,
        ApogeeCoaddedSpectrumInApStar.doppler_fe_h,
        ApogeeCoaddedSpectrumInApStar.doppler_e_fe_h,
        ApogeeCoaddedSpectrumInApStar.doppler_rchi2,
        ApogeeCoaddedSpectrumInApStar.doppler_flags,

        #> Radial Velocity (X-Correlation)
        ApogeeCoaddedSpectrumInApStar.xcorr_v_rad,
        ApogeeCoaddedSpectrumInApStar.xcorr_v_rel,
        ApogeeCoaddedSpectrumInApStar.xcorr_e_v_rel,
        ApogeeCoaddedSpectrumInApStar.ccfwhm,
        ApogeeCoaddedSpectrumInApStar.autofwhm,
        ApogeeCoaddedSpectrumInApStar.n_components,      
    ),
    BossCombinedSpectrum: (
        BossCombinedSpectrum.release,
        BossCombinedSpectrum.filetype,
        BossCombinedSpectrum.run2d,
        BossCombinedSpectrum.telescope,

        BossCombinedSpectrum.min_mjd,
        BossCombinedSpectrum.max_mjd,
        BossCombinedSpectrum.n_visits,
        BossCombinedSpectrum.n_good_visits,
        BossCombinedSpectrum.n_good_rvs,

        BossCombinedSpectrum.v_rad,
        BossCombinedSpectrum.e_v_rad,
        BossCombinedSpectrum.std_v_rad,
        BossCombinedSpectrum.median_e_v_rad,
        BossCombinedSpectrum.xcsao_teff,
        BossCombinedSpectrum.xcsao_e_teff,
        BossCombinedSpectrum.xcsao_logg,
        BossCombinedSpectrum.xcsao_e_logg,
        BossCombinedSpectrum.xcsao_fe_h,
        BossCombinedSpectrum.xcsao_e_fe_h,
        BossCombinedSpectrum.xcsao_meanrxc,

        # Metadata
        BossCombinedSpectrum.snr,
        BossCombinedSpectrum.gri_gaia_transform_flags,
        BossCombinedSpectrum.zwarning_flags,        
    )
}

def _query_astro_nn_coadd(exclude_source_pks=None, where=None):
    q = (
        AstroNN
        .select(
            Value('astro_nn').alias('pipeline'),

            AstroNN.source_pk,
            AstroNN.spectrum_pk,
            AstroNN.task_pk,

            AstroNN.teff,
            AstroNN.e_teff,
            AstroNN.logg,
            AstroNN.e_logg,
            AstroNN.fe_h,
            AstroNN.e_fe_h,

            AstroNN.c_h,
            AstroNN.e_c_h,
            AstroNN.c_1_h,
            AstroNN.e_c_1_h,
            AstroNN.n_h,
            AstroNN.e_n_h,
            AstroNN.o_h,
            AstroNN.e_o_h,
            AstroNN.na_h,
            AstroNN.e_na_h,
            AstroNN.mg_h,
            AstroNN.e_mg_h,
            AstroNN.al_h,
            AstroNN.e_al_h,
            AstroNN.si_h,
            AstroNN.e_si_h,
            AstroNN.p_h,
            AstroNN.e_p_h,
            AstroNN.s_h,
            AstroNN.e_s_h,
            AstroNN.k_h,
            AstroNN.e_k_h,
            AstroNN.ca_h,
            AstroNN.e_ca_h,
            AstroNN.ti_h,
            AstroNN.e_ti_h,
            AstroNN.ti_2_h,
            AstroNN.e_ti_2_h,
            AstroNN.v_h,
            AstroNN.e_v_h,
            AstroNN.cr_h,
            AstroNN.e_cr_h,
            AstroNN.mn_h,
            AstroNN.e_mn_h,
            AstroNN.co_h,
            AstroNN.e_co_h,
            AstroNN.ni_h,
            AstroNN.e_ni_h,

            AstroNN.result_flags,
            *SPECTRUM_FIELDS[ApogeeCoaddedSpectrumInApStar],
        )
        .distinct(AstroNN.source_pk)
        .join(Spectrum)
        .join(ApogeeCoaddedSpectrumInApStar)
        .switch(AstroNN)
        .join(Source)
        .where(
            (AstroNN.v_astra == __version__)
        )        
    )
    if exclude_source_pks is not None:
        q = q.where(~AstroNN.source_pk.in_(exclude_source_pks))

    if where:
        q = q.where(where)
    
    return q.dicts()


def _query_slam_coadd(exclude_source_pks=None, where=None):
    q = (
        Slam
        .select(
            Value('slam').alias('pipeline'),

            Slam.source_pk,
            Slam.spectrum_pk,
            Slam.task_pk,

            Slam.teff,
            Slam.e_teff,
            Slam.logg,
            Slam.e_logg,
            Slam.fe_h,
            Slam.e_fe_h,
            Slam.alpha_fe.alias("alpha_m_h_atm"),
            Slam.e_alpha_fe.alias("e_alpha_m_h_atm"),
            Slam.rchi2,
            Slam.result_flags,

            *SPECTRUM_FIELDS[BossCombinedSpectrum]
        )
        .distinct(Slam.source_pk)
        .join(Spectrum)
        .join(BossCombinedSpectrum)
        .switch(Slam)
        .join(Source)
        .where(
            (Slam.v_astra == __version__)
        &   (BossCombinedSpectrum.v_astra == __version__)
        )
    )
    if exclude_source_pks is not None:
        q = q.where(~Slam.source_pk.in_(exclude_source_pks))
    if where:
        q = q.where(where)
    
    return q.dicts()


def _query_apogee_net_coadd(exclude_source_pks=None, where=None):
    q = (
        ApogeeNet
        .select(
            Value('apogee_net').alias('pipeline'),

            ApogeeNet.source_pk,
            ApogeeNet.spectrum_pk,
            ApogeeNet.task_pk,

            ApogeeNet.teff,
            ApogeeNet.e_teff,
            ApogeeNet.logg,
            ApogeeNet.e_logg,
            ApogeeNet.fe_h,
            ApogeeNet.e_fe_h,
            ApogeeNet.result_flags,

            *SPECTRUM_FIELDS[ApogeeCoaddedSpectrumInApStar],
        )
        .distinct(ApogeeNet.source_pk)
        .join(Spectrum)
        .join(ApogeeCoaddedSpectrumInApStar)
        .switch(ApogeeNet)
        .join(Source)
        .where(ApogeeNet.v_astra == __version__)
    )
    if exclude_source_pks is not None:
        q = q.where(~ApogeeNet.source_pk.in_(exclude_source_pks))
    if where:
        q = q.where(where)        
    
    return q.dicts()        




def _query_aspcap_coadd(exclude_source_pks=None, where=None):
    q = (
        ASPCAP
        .select(
            Value('aspcap').alias('pipeline'),

            ASPCAP.source_pk,
            ASPCAP.spectrum_pk,
            ASPCAP.task_pk,

            ASPCAP.teff,
            ASPCAP.e_teff,
            ASPCAP.logg,
            ASPCAP.e_logg,
            ASPCAP.v_micro,
            ASPCAP.e_v_micro,
            ASPCAP.v_sini,
            ASPCAP.e_v_sini,
            ASPCAP.m_h_atm,
            ASPCAP.e_m_h_atm,
            ASPCAP.alpha_m_atm,
            ASPCAP.e_alpha_m_atm,
            ASPCAP.c_m_atm,
            ASPCAP.e_c_m_atm,
            ASPCAP.n_m_atm,
            ASPCAP.e_n_m_atm,

            #> Chemical Abundances
            ASPCAP.al_h,
            ASPCAP.e_al_h,
            ASPCAP.al_h_flags,
            ASPCAP.al_h_rchi2,

            ASPCAP.c_12_13,
            ASPCAP.e_c_12_13,
            ASPCAP.c_12_13_flags,
            ASPCAP.c_12_13_rchi2,

            ASPCAP.ca_h,
            ASPCAP.e_ca_h,
            ASPCAP.ca_h_flags,
            ASPCAP.ca_h_rchi2,
    
            ASPCAP.ce_h,
            ASPCAP.e_ce_h,
            ASPCAP.ce_h_flags,
            ASPCAP.ce_h_rchi2,
    
            ASPCAP.c_1_h,
            ASPCAP.e_c_1_h,
            ASPCAP.c_1_h_flags,
            ASPCAP.c_1_h_rchi2,
    
            ASPCAP.c_h,
            ASPCAP.e_c_h,
            ASPCAP.c_h_flags,
            ASPCAP.c_h_rchi2,
    
            ASPCAP.co_h,
            ASPCAP.e_co_h,
            ASPCAP.co_h_flags,
            ASPCAP.co_h_rchi2,
    
            ASPCAP.cr_h,
            ASPCAP.e_cr_h,
            ASPCAP.cr_h_flags,
            ASPCAP.cr_h_rchi2,
    
            ASPCAP.cu_h,
            ASPCAP.e_cu_h,
            ASPCAP.cu_h_flags,
            ASPCAP.cu_h_rchi2,
    
            ASPCAP.fe_h,
            ASPCAP.e_fe_h,
            ASPCAP.fe_h_flags,
            ASPCAP.fe_h_rchi2,

            ASPCAP.k_h,
            ASPCAP.e_k_h,
            ASPCAP.k_h_flags,
            ASPCAP.k_h_rchi2,

            ASPCAP.mg_h,
            ASPCAP.e_mg_h,
            ASPCAP.mg_h_flags,
            ASPCAP.mg_h_rchi2,

            ASPCAP.mn_h,
            ASPCAP.e_mn_h,
            ASPCAP.mn_h_flags,
            ASPCAP.mn_h_rchi2,

            ASPCAP.na_h,
            ASPCAP.e_na_h,
            ASPCAP.na_h_flags,
            ASPCAP.na_h_rchi2,

            ASPCAP.nd_h,
            ASPCAP.e_nd_h,
            ASPCAP.nd_h_flags,
            ASPCAP.nd_h_rchi2,

            ASPCAP.ni_h,
            ASPCAP.e_ni_h,
            ASPCAP.ni_h_flags,
            ASPCAP.ni_h_rchi2,

            ASPCAP.n_h,
            ASPCAP.e_n_h,
            ASPCAP.n_h_flags,
            ASPCAP.n_h_rchi2,

            ASPCAP.o_h,
            ASPCAP.e_o_h,
            ASPCAP.o_h_flags,
            ASPCAP.o_h_rchi2,

            ASPCAP.p_h,
            ASPCAP.e_p_h,
            ASPCAP.p_h_flags,
            ASPCAP.p_h_rchi2,

            ASPCAP.si_h,
            ASPCAP.e_si_h,
            ASPCAP.si_h_flags,
            ASPCAP.si_h_rchi2,

            ASPCAP.s_h,
            ASPCAP.e_s_h,
            ASPCAP.s_h_flags,
            ASPCAP.s_h_rchi2,

            ASPCAP.ti_h,
            ASPCAP.e_ti_h,
            ASPCAP.ti_h_flags,
            ASPCAP.ti_h_rchi2,

            ASPCAP.ti_2_h,
            ASPCAP.e_ti_2_h,
            ASPCAP.ti_2_h_flags,
            ASPCAP.ti_2_h_rchi2,

            ASPCAP.v_h,
            ASPCAP.e_v_h,
            ASPCAP.v_h_flags,
            ASPCAP.v_h_rchi2,

            *SPECTRUM_FIELDS[ApogeeCoaddedSpectrumInApStar],       
        )
        .distinct(ASPCAP.source_pk)
        .join(Spectrum)
        .join(ApogeeCoaddedSpectrumInApStar)
        .switch(ASPCAP)
        .join(Source)
        .where(
            (ASPCAP.v_astra == __version__)
        )
    )
    if exclude_source_pks is not None:
        q = q.where(~ASPCAP.source_pk.in_(exclude_source_pks))
    if where:
        q = q.where(where)        
    
    return q.dicts()


def _query_boss_net_coadd(exclude_source_pks=None, where=None):
    q = (
        BossNet
        .select(
            Value('boss_net').alias('pipeline'),
            BossNet.source_pk,
            BossNet.spectrum_pk,
            BossNet.task_pk,

            BossNet.teff,
            BossNet.e_teff,
            BossNet.logg,
            BossNet.e_logg,
            BossNet.fe_h,
            BossNet.e_fe_h,
            BossNet.result_flags,
            BossNet.v_rad.alias("boss_net_v_rad"),
            BossNet.e_v_rad.alias("boss_net_e_v_rad"),

            # healpix, sdss_id will come from Source-level
            *SPECTRUM_FIELDS[BossCombinedSpectrum],
        )
        .distinct(BossNet.source_pk)
        .join(Spectrum)
        .join(BossCombinedSpectrum)
        .switch(BossNet)
        .join(Source)
        .where(
            (BossNet.v_astra == __version__)
        &   (BossCombinedSpectrum.v_astra == __version__)
        )
    )
    if exclude_source_pks is not None:
        q = q.where(~BossNet.source_pk.in_(exclude_source_pks))
    if where:
        q = q.where(where)
    
    return q.dicts()



def _query_snow_white_coadd(exclude_source_pks=None, where=None):
    q = (
        SnowWhite
        .select(
            # Add a fake field called 'pipeline' that just always returns 'snow_white'
            Value('snow_white').alias('pipeline'),
            SnowWhite.source_pk,
            SnowWhite.spectrum_pk,

            SnowWhite.task_pk,
            SnowWhite.classification,
            SnowWhite.p_cv,
            SnowWhite.p_da,
            SnowWhite.p_dab,
            SnowWhite.p_dabz,
            SnowWhite.p_dah,
            SnowWhite.p_dahe,
            SnowWhite.p_dao,
            SnowWhite.p_daz,
            SnowWhite.p_da_ms,
            SnowWhite.p_db,
            SnowWhite.p_dba,
            SnowWhite.p_dbaz,
            SnowWhite.p_dbh,
            SnowWhite.p_dbz,
            SnowWhite.p_db_ms,
            SnowWhite.p_dc,
            SnowWhite.p_dc_ms,
            SnowWhite.p_do,
            SnowWhite.p_dq,
            SnowWhite.p_dqz,
            SnowWhite.p_dqpec,
            SnowWhite.p_dz,
            SnowWhite.p_dza,
            SnowWhite.p_dzb,
            SnowWhite.p_dzba,
            SnowWhite.p_mwd,
            SnowWhite.p_hotdq,

            #> Snow White Stellar Parameters
            SnowWhite.teff,
            SnowWhite.e_teff,
            SnowWhite.logg,
            SnowWhite.e_logg,
            #SnowWhite.v_rel,
            SnowWhite.result_flags,
    
            # healpix, sdss_id will come from Source-level
            *SPECTRUM_FIELDS[BossCombinedSpectrum],
        )
        .distinct(SnowWhite.source_pk)
        .join(Spectrum)
        .join(BossCombinedSpectrum)
        .switch(SnowWhite)
        .join(Source)
        .where(
            (SnowWhite.v_astra == __version__)
        &   (BossCombinedSpectrum.v_astra == __version__)
        )
    )
    if exclude_source_pks is not None:
        q = q.where(~SnowWhite.source_pk.in_(exclude_source_pks))
    if where:
        q = q.where(where)
    
    return q.dicts()