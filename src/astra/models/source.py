import numpy as np
import datetime
from functools import cache
from peewee import fn, PostgresqlDatabase
from playhouse.hybrid import hybrid_method
from astra.models.base import database, BaseModel
from astra.fields import (
    AutoField,
    BitField,
    IntegerField,
    TextField,
    FloatField,
    BigBitField,
    BigIntegerField,
    SmallIntegerField,
    DateTimeField,
    BooleanField
)
from astra.models.spectrum import Spectrum
from astra.glossary import Glossary
from astra.utils import expand_path


class Source(BaseModel):

    """ An astronomical source. """

    pk = AutoField(primary_key=True)

    #> Identifiers    
    sdss_id = BigIntegerField(index=True, unique=True, null=True)
    sdss4_apogee_id = TextField(index=True, unique=True, null=True)
    gaia_dr2_source_id = BigIntegerField(null=True, unique=True)
    gaia_dr3_source_id = BigIntegerField(null=True, unique=True)
    tic_v8_id = BigIntegerField(null=True)
    healpix = IntegerField(null=True)
    
    #> Targeting Provenance 
    lead = TextField(null=True)
    version_id = IntegerField(null=True)
    catalogid = BigIntegerField(null=True)
    catalogid21 = BigIntegerField(null=True)
    catalogid25 = BigIntegerField(null=True)
    catalogid31 = BigIntegerField(null=True)
    n_associated = IntegerField(null=True)
    n_neighborhood = IntegerField(default=-1)
    
    # Only do sdss5_target_flags if we are using a PostgreSQL database, as SQLite does not support it.
    if isinstance(database, PostgresqlDatabase):
        sdss5_target_flags = BigBitField(null=True)

    # https://www.sdss4.org/dr17/irspec/apogee-bitmasks/
    sdss4_apogee_target1_flags = BitField(default=0)
    sdss4_apogee_target2_flags = BitField(default=0)
    sdss4_apogee2_target1_flags = BitField(default=0)
    sdss4_apogee2_target2_flags = BitField(default=0)
    sdss4_apogee2_target3_flags = BitField(default=0)
    sdss4_apogee_member_flags = BitField(default=0)
    sdss4_apogee_extra_target_flags = BitField(default=0)
    
    # sdss4_apogee_target1_flags
    flag_sdss4_apogee_faint = sdss4_apogee_target1_flags.flag(2**0)
    flag_sdss4_apogee_medium = sdss4_apogee_target1_flags.flag(2**1)
    flag_sdss4_apogee_bright = sdss4_apogee_target1_flags.flag(2**2)
    flag_sdss4_apogee_irac_dered = sdss4_apogee_target1_flags.flag(2**3)
    flag_sdss4_apogee_wise_dered = sdss4_apogee_target1_flags.flag(2**4)
    flag_sdss4_apogee_sfd_dered = sdss4_apogee_target1_flags.flag(2**5)
    flag_sdss4_apogee_no_dered = sdss4_apogee_target1_flags.flag(2**6)
    flag_sdss4_apogee_wash_giant = sdss4_apogee_target1_flags.flag(2**7)
    flag_sdss4_apogee_wash_dwarf = sdss4_apogee_target1_flags.flag(2**8)
    flag_sdss4_apogee_sci_cluster = sdss4_apogee_target1_flags.flag(2**9)
    flag_sdss4_apogee_extended = sdss4_apogee_target1_flags.flag(2**10)
    flag_sdss4_apogee_short = sdss4_apogee_target1_flags.flag(2**11)
    flag_sdss4_apogee_intermediate = sdss4_apogee_target1_flags.flag(2**12)
    flag_sdss4_apogee_long = sdss4_apogee_target1_flags.flag(2**13)
    flag_sdss4_apogee_do_not_observe = sdss4_apogee_target1_flags.flag(2**14)
    flag_sdss4_apogee_serendipitous = sdss4_apogee_target1_flags.flag(2**15)
    flag_sdss4_apogee_first_light = sdss4_apogee_target1_flags.flag(2**16)
    flag_sdss4_apogee_ancillary = sdss4_apogee_target1_flags.flag(2**17)
    flag_sdss4_apogee_m31_cluster = sdss4_apogee_target1_flags.flag(2**18)
    flag_sdss4_apogee_mdwarf = sdss4_apogee_target1_flags.flag(2**19)
    flag_sdss4_apogee_hires = sdss4_apogee_target1_flags.flag(2**20)
    flag_sdss4_apogee_old_star = sdss4_apogee_target1_flags.flag(2**21)
    flag_sdss4_apogee_disk_red_giant = sdss4_apogee_target1_flags.flag(2**22)
    flag_sdss4_apogee_kepler_eb = sdss4_apogee_target1_flags.flag(2**23)
    flag_sdss4_apogee_gc_pal1 = sdss4_apogee_target1_flags.flag(2**24)
    flag_sdss4_apogee_massive_star = sdss4_apogee_target1_flags.flag(2**25)
    flag_sdss4_apogee_sgr_dsph = sdss4_apogee_target1_flags.flag(2**26)
    flag_sdss4_apogee_kepler_seismo = sdss4_apogee_target1_flags.flag(2**27)
    flag_sdss4_apogee_kepler_host = sdss4_apogee_target1_flags.flag(2**28)
    flag_sdss4_apogee_faint_extra = sdss4_apogee_target1_flags.flag(2**29)
    flag_sdss4_apogee_segue_overlap = sdss4_apogee_target1_flags.flag(2**30)

    # sdss4_apogee_target2_flags
    flag_sdss4_apogee_light_trap = sdss4_apogee_target2_flags.flag(2**0)
    flag_sdss4_apogee_flux_standard = sdss4_apogee_target2_flags.flag(2**1)
    flag_sdss4_apogee_standard_star = sdss4_apogee_target2_flags.flag(2**2)
    flag_sdss4_apogee_rv_standard = sdss4_apogee_target2_flags.flag(2**3)
    flag_sdss4_apogee_sky = sdss4_apogee_target2_flags.flag(2**4)
    flag_sdss4_apogee_sky_bad = sdss4_apogee_target2_flags.flag(2**5)
    flag_sdss4_apogee_guide_star = sdss4_apogee_target2_flags.flag(2**6)
    flag_sdss4_apogee_bundle_hole = sdss4_apogee_target2_flags.flag(2**7)
    flag_sdss4_apogee_telluric_bad = sdss4_apogee_target2_flags.flag(2**8)
    flag_sdss4_apogee_telluric = sdss4_apogee_target2_flags.flag(2**9)
    flag_sdss4_apogee_calib_cluster = sdss4_apogee_target2_flags.flag(2**10)
    flag_sdss4_apogee_bulge_giant = sdss4_apogee_target2_flags.flag(2**11)
    flag_sdss4_apogee_bulge_super_giant = sdss4_apogee_target2_flags.flag(2**12)
    flag_sdss4_apogee_embedded_cluster_star = sdss4_apogee_target2_flags.flag(2**13)
    flag_sdss4_apogee_long_bar = sdss4_apogee_target2_flags.flag(2**14)
    flag_sdss4_apogee_emission_star = sdss4_apogee_target2_flags.flag(2**15)
    flag_sdss4_apogee_kepler_cool_dwarf = sdss4_apogee_target2_flags.flag(2**16)
    flag_sdss4_apogee_mir_cluster_star = sdss4_apogee_target2_flags.flag(2**17)
    flag_sdss4_apogee_rv_monitor_ic348 = sdss4_apogee_target2_flags.flag(2**18)
    flag_sdss4_apogee_rv_monitor_kepler = sdss4_apogee_target2_flags.flag(2**19)
    flag_sdss4_apogee_ges_calibrate = sdss4_apogee_target2_flags.flag(2**20)
    flag_sdss4_apogee_bulge_rv_verify = sdss4_apogee_target2_flags.flag(2**21)
    flag_sdss4_apogee_1m_target = sdss4_apogee_target2_flags.flag(2**22)

    # sdss4_apogee2_target1_flags
    flag_sdss4_apogee2_onebit_gt_0_5 = sdss4_apogee2_target1_flags.flag(2**0)
    flag_sdss4_apogee2_twobit_0_5_to_0_8 = sdss4_apogee2_target1_flags.flag(2**1)
    flag_sdss4_apogee2_twobit_gt_0_8 = sdss4_apogee2_target1_flags.flag(2**2)
    flag_sdss4_apogee2_irac_dered = sdss4_apogee2_target1_flags.flag(2**3)
    flag_sdss4_apogee2_wise_dered = sdss4_apogee2_target1_flags.flag(2**4)
    flag_sdss4_apogee2_sfd_dered = sdss4_apogee2_target1_flags.flag(2**5)
    flag_sdss4_apogee2_no_dered = sdss4_apogee2_target1_flags.flag(2**6)
    flag_sdss4_apogee2_wash_giant = sdss4_apogee2_target1_flags.flag(2**7)
    flag_sdss4_apogee2_wash_dwarf = sdss4_apogee2_target1_flags.flag(2**8)
    flag_sdss4_apogee2_sci_cluster = sdss4_apogee2_target1_flags.flag(2**9)
    flag_sdss4_apogee2_cluster_candidate = sdss4_apogee2_target1_flags.flag(2**10)
    flag_sdss4_apogee2_short = sdss4_apogee2_target1_flags.flag(2**11)
    flag_sdss4_apogee2_medium = sdss4_apogee2_target1_flags.flag(2**12)
    flag_sdss4_apogee2_long = sdss4_apogee2_target1_flags.flag(2**13)
    flag_sdss4_apogee2_normal_sample = sdss4_apogee2_target1_flags.flag(2**14)
    flag_sdss4_apogee2_manga_led = sdss4_apogee2_target1_flags.flag(2**15)
    flag_sdss4_apogee2_onebin_gt_0_3 = sdss4_apogee2_target1_flags.flag(2**16)
    flag_sdss4_apogee2_wash_noclass = sdss4_apogee2_target1_flags.flag(2**17)
    flag_sdss4_apogee2_stream_member = sdss4_apogee2_target1_flags.flag(2**18)
    flag_sdss4_apogee2_stream_candidate = sdss4_apogee2_target1_flags.flag(2**19)
    flag_sdss4_apogee2_dsph_member = sdss4_apogee2_target1_flags.flag(2**20)
    flag_sdss4_apogee2_dsph_candidate = sdss4_apogee2_target1_flags.flag(2**21)
    flag_sdss4_apogee2_magcloud_member = sdss4_apogee2_target1_flags.flag(2**22)
    flag_sdss4_apogee2_magcloud_candidate = sdss4_apogee2_target1_flags.flag(2**23)
    flag_sdss4_apogee2_rrlyr = sdss4_apogee2_target1_flags.flag(2**24)
    flag_sdss4_apogee2_bulge_rc = sdss4_apogee2_target1_flags.flag(2**25)
    flag_sdss4_apogee2_sgr_dsph = sdss4_apogee2_target1_flags.flag(2**26)
    flag_sdss4_apogee2_apokasc_giant = sdss4_apogee2_target1_flags.flag(2**27)
    flag_sdss4_apogee2_apokasc_dwarf = sdss4_apogee2_target1_flags.flag(2**28)
    flag_sdss4_apogee2_faint_extra = sdss4_apogee2_target1_flags.flag(2**29)
    flag_sdss4_apogee2_apokasc = sdss4_apogee2_target1_flags.flag(2**30)

    # sdss4_apogee2_target2_flags
    flag_sdss4_apogee2_k2_gap = sdss4_apogee2_target2_flags.flag(2**0)
    flag_sdss4_apogee2_ccloud_as4 = sdss4_apogee2_target2_flags.flag(2**1)
    flag_sdss4_apogee2_standard_star = sdss4_apogee2_target2_flags.flag(2**2)
    flag_sdss4_apogee2_rv_standard = sdss4_apogee2_target2_flags.flag(2**3)
    flag_sdss4_apogee2_sky = sdss4_apogee2_target2_flags.flag(2**4)
    flag_sdss4_apogee2_external_calib = sdss4_apogee2_target2_flags.flag(2**5)
    flag_sdss4_apogee2_internal_calib = sdss4_apogee2_target2_flags.flag(2**6)
    flag_sdss4_apogee2_disk_substructure_member = sdss4_apogee2_target2_flags.flag(2**7)
    flag_sdss4_apogee2_disk_substructure_candidate = sdss4_apogee2_target2_flags.flag(2**8)
    flag_sdss4_apogee2_telluric = sdss4_apogee2_target2_flags.flag(2**9)
    flag_sdss4_apogee2_calib_cluster = sdss4_apogee2_target2_flags.flag(2**10)
    flag_sdss4_apogee2_k2_planet_host = sdss4_apogee2_target2_flags.flag(2**11)
    flag_sdss4_apogee2_tidal_binary = sdss4_apogee2_target2_flags.flag(2**12)
    flag_sdss4_apogee2_literature_calib = sdss4_apogee2_target2_flags.flag(2**13)
    flag_sdss4_apogee2_ges_overlap = sdss4_apogee2_target2_flags.flag(2**14)
    flag_sdss4_apogee2_argos_overlap = sdss4_apogee2_target2_flags.flag(2**15)
    flag_sdss4_apogee2_gaia_overlap = sdss4_apogee2_target2_flags.flag(2**16)
    flag_sdss4_apogee2_galah_overlap = sdss4_apogee2_target2_flags.flag(2**17)
    flag_sdss4_apogee2_rave_overlap = sdss4_apogee2_target2_flags.flag(2**18)
    flag_sdss4_apogee2_commis_south_spec = sdss4_apogee2_target2_flags.flag(2**19)
    flag_sdss4_apogee2_halo_member = sdss4_apogee2_target2_flags.flag(2**20)
    flag_sdss4_apogee2_halo_candidate = sdss4_apogee2_target2_flags.flag(2**21)
    flag_sdss4_apogee2_1m_target = sdss4_apogee2_target2_flags.flag(2**22)
    flag_sdss4_apogee2_mod_bright_limit = sdss4_apogee2_target2_flags.flag(2**23)
    flag_sdss4_apogee2_cis = sdss4_apogee2_target2_flags.flag(2**24)
    flag_sdss4_apogee2_cntac = sdss4_apogee2_target2_flags.flag(2**25)
    flag_sdss4_apogee2_external = sdss4_apogee2_target2_flags.flag(2**26)
    flag_sdss4_apogee2_cvz_as4_obaf = sdss4_apogee2_target2_flags.flag(2**27)
    flag_sdss4_apogee2_cvz_as4_gi = sdss4_apogee2_target2_flags.flag(2**28)
    flag_sdss4_apogee2_cvz_as4_ctl = sdss4_apogee2_target2_flags.flag(2**29)
    flag_sdss4_apogee2_cvz_as4_giant = sdss4_apogee2_target2_flags.flag(2**30)

    # sdss4_apogee2_target3_flags
    flag_sdss4_apogee2_koi = sdss4_apogee2_target3_flags.flag(2**0)
    flag_sdss4_apogee2_eb = sdss4_apogee2_target3_flags.flag(2**1)
    flag_sdss4_apogee2_koi_control = sdss4_apogee2_target3_flags.flag(2**2)
    flag_sdss4_apogee2_mdwarf = sdss4_apogee2_target3_flags.flag(2**3)
    flag_sdss4_apogee2_substellar_companions = sdss4_apogee2_target3_flags.flag(2**4)
    flag_sdss4_apogee2_young_cluster = sdss4_apogee2_target3_flags.flag(2**5)
    flag_sdss4_apogee2_k2 = sdss4_apogee2_target3_flags.flag(2**6)
    flag_sdss4_apogee2_object = sdss4_apogee2_target3_flags.flag(2**7)
    flag_sdss4_apogee2_ancillary = sdss4_apogee2_target3_flags.flag(2**8)
    flag_sdss4_apogee2_massive_star = sdss4_apogee2_target3_flags.flag(2**9)
    flag_sdss4_apogee2_qso = sdss4_apogee2_target3_flags.flag(2**10)
    flag_sdss4_apogee2_cepheid = sdss4_apogee2_target3_flags.flag(2**11)
    flag_sdss4_apogee2_low_av_windows = sdss4_apogee2_target3_flags.flag(2**12)
    flag_sdss4_apogee2_be_star = sdss4_apogee2_target3_flags.flag(2**13)
    flag_sdss4_apogee2_young_moving_group = sdss4_apogee2_target3_flags.flag(2**14)
    flag_sdss4_apogee2_ngc6791 = sdss4_apogee2_target3_flags.flag(2**15)
    flag_sdss4_apogee2_label_star = sdss4_apogee2_target3_flags.flag(2**16)
    flag_sdss4_apogee2_faint_kepler_giants = sdss4_apogee2_target3_flags.flag(2**17)
    flag_sdss4_apogee2_w345 = sdss4_apogee2_target3_flags.flag(2**18)
    flag_sdss4_apogee2_massive_evolved = sdss4_apogee2_target3_flags.flag(2**19)
    flag_sdss4_apogee2_extinction = sdss4_apogee2_target3_flags.flag(2**20)
    flag_sdss4_apogee2_kepler_mdwarf_koi = sdss4_apogee2_target3_flags.flag(2**21)
    flag_sdss4_apogee2_agb = sdss4_apogee2_target3_flags.flag(2**22)
    flag_sdss4_apogee2_m33 = sdss4_apogee2_target3_flags.flag(2**23)
    flag_sdss4_apogee2_ultracool = sdss4_apogee2_target3_flags.flag(2**24)
    flag_sdss4_apogee2_distant_segue_giants = sdss4_apogee2_target3_flags.flag(2**25)
    flag_sdss4_apogee2_cepheid_mapping = sdss4_apogee2_target3_flags.flag(2**26)
    flag_sdss4_apogee2_sa57 = sdss4_apogee2_target3_flags.flag(2**27)
    flag_sdss4_apogee2_k2_mdwarf = sdss4_apogee2_target3_flags.flag(2**28)
    flag_sdss4_apogee2_rvvar = sdss4_apogee2_target3_flags.flag(2**29)
    flag_sdss4_apogee2_m31 = sdss4_apogee2_target3_flags.flag(2**30)

    # sdss4_apogee_extra_target_flags
    flag_sdss4_apogee_not_main = sdss4_apogee_extra_target_flags.flag(2**0)
    flag_sdss4_apogee_commissioning = sdss4_apogee_extra_target_flags.flag(2**1)
    flag_sdss4_apogee_telluric = sdss4_apogee_extra_target_flags.flag(2**2)
    flag_sdss4_apogee_apo1m = sdss4_apogee_extra_target_flags.flag(2**3)
    flag_sdss4_apogee_duplicate = sdss4_apogee_extra_target_flags.flag(2**4)

    # sdss4_apogee_member_flags
    flag_sdss4_apogee_member_m92 = sdss4_apogee_member_flags.flag(2**0)
    flag_sdss4_apogee_member_m15 = sdss4_apogee_member_flags.flag(2**1)
    flag_sdss4_apogee_member_m53 = sdss4_apogee_member_flags.flag(2**2)
    flag_sdss4_apogee_member_ngc_5466 = sdss4_apogee_member_flags.flag(2**3)
    flag_sdss4_apogee_member_ngc_4147 = sdss4_apogee_member_flags.flag(2**4)
    flag_sdss4_apogee_member_m2 = sdss4_apogee_member_flags.flag(2**5)
    flag_sdss4_apogee_member_m13 = sdss4_apogee_member_flags.flag(2**6)
    flag_sdss4_apogee_member_m3 = sdss4_apogee_member_flags.flag(2**7)
    flag_sdss4_apogee_member_m5 = sdss4_apogee_member_flags.flag(2**8)
    flag_sdss4_apogee_member_m12 = sdss4_apogee_member_flags.flag(2**9)
    flag_sdss4_apogee_member_m107 = sdss4_apogee_member_flags.flag(2**10)
    flag_sdss4_apogee_member_m71 = sdss4_apogee_member_flags.flag(2**11)
    flag_sdss4_apogee_member_ngc_2243 = sdss4_apogee_member_flags.flag(2**12)
    flag_sdss4_apogee_member_be29 = sdss4_apogee_member_flags.flag(2**13)
    flag_sdss4_apogee_member_ngc_2158 = sdss4_apogee_member_flags.flag(2**14)
    flag_sdss4_apogee_member_m35 = sdss4_apogee_member_flags.flag(2**15)
    flag_sdss4_apogee_member_ngc_2420 = sdss4_apogee_member_flags.flag(2**16)
    flag_sdss4_apogee_member_ngc_188 = sdss4_apogee_member_flags.flag(2**17)
    flag_sdss4_apogee_member_m67 = sdss4_apogee_member_flags.flag(2**18)
    flag_sdss4_apogee_member_ngc_7789 = sdss4_apogee_member_flags.flag(2**19)
    flag_sdss4_apogee_member_pleiades = sdss4_apogee_member_flags.flag(2**20)
    flag_sdss4_apogee_member_ngc_6819 = sdss4_apogee_member_flags.flag(2**21)
    flag_sdss4_apogee_member_coma_berenices = sdss4_apogee_member_flags.flag(2**22)
    flag_sdss4_apogee_member_ngc_6791 = sdss4_apogee_member_flags.flag(2**23)
    flag_sdss4_apogee_member_ngc_5053 = sdss4_apogee_member_flags.flag(2**24)
    flag_sdss4_apogee_member_m68 = sdss4_apogee_member_flags.flag(2**25)
    flag_sdss4_apogee_member_ngc_6397 = sdss4_apogee_member_flags.flag(2**26)
    flag_sdss4_apogee_member_m55 = sdss4_apogee_member_flags.flag(2**27)
    flag_sdss4_apogee_member_ngc_5634 = sdss4_apogee_member_flags.flag(2**28)
    flag_sdss4_apogee_member_m22 = sdss4_apogee_member_flags.flag(2**29)
    flag_sdss4_apogee_member_m79 = sdss4_apogee_member_flags.flag(2**30)
    flag_sdss4_apogee_member_ngc_3201 = sdss4_apogee_member_flags.flag(2**31)
    flag_sdss4_apogee_member_m10 = sdss4_apogee_member_flags.flag(2**32)
    flag_sdss4_apogee_member_ngc_6752 = sdss4_apogee_member_flags.flag(2**33)
    flag_sdss4_apogee_member_omega_centauri = sdss4_apogee_member_flags.flag(2**34)
    flag_sdss4_apogee_member_m54 = sdss4_apogee_member_flags.flag(2**35)
    flag_sdss4_apogee_member_ngc_6229 = sdss4_apogee_member_flags.flag(2**36)
    flag_sdss4_apogee_member_pal5 = sdss4_apogee_member_flags.flag(2**37)
    flag_sdss4_apogee_member_ngc_6544 = sdss4_apogee_member_flags.flag(2**38)
    flag_sdss4_apogee_member_ngc_6522 = sdss4_apogee_member_flags.flag(2**39)
    flag_sdss4_apogee_member_ngc_288 = sdss4_apogee_member_flags.flag(2**40)
    flag_sdss4_apogee_member_ngc_362 = sdss4_apogee_member_flags.flag(2**41)
    flag_sdss4_apogee_member_ngc_1851 = sdss4_apogee_member_flags.flag(2**42)
    flag_sdss4_apogee_member_m4 = sdss4_apogee_member_flags.flag(2**43)
    flag_sdss4_apogee_member_ngc_2808 = sdss4_apogee_member_flags.flag(2**44)
    flag_sdss4_apogee_member_pal6 = sdss4_apogee_member_flags.flag(2**45)
    flag_sdss4_apogee_member_47tuc = sdss4_apogee_member_flags.flag(2**46)
    flag_sdss4_apogee_member_pal1 = sdss4_apogee_member_flags.flag(2**47)
    flag_sdss4_apogee_member_ngc_6539 = sdss4_apogee_member_flags.flag(2**48)
    flag_sdss4_apogee_member_ngc_6388 = sdss4_apogee_member_flags.flag(2**49)
    flag_sdss4_apogee_member_ngc_6441 = sdss4_apogee_member_flags.flag(2**50)
    flag_sdss4_apogee_member_ngc_6316 = sdss4_apogee_member_flags.flag(2**51)
    flag_sdss4_apogee_member_ngc_6760 = sdss4_apogee_member_flags.flag(2**52)
    flag_sdss4_apogee_member_ngc_6553 = sdss4_apogee_member_flags.flag(2**53)
    flag_sdss4_apogee_member_ngc_6528 = sdss4_apogee_member_flags.flag(2**54)
    flag_sdss4_apogee_member_draco = sdss4_apogee_member_flags.flag(2**55)
    flag_sdss4_apogee_member_urminor = sdss4_apogee_member_flags.flag(2**56)
    flag_sdss4_apogee_member_bootes1 = sdss4_apogee_member_flags.flag(2**57)
    flag_sdss4_apogee_member_sexans = sdss4_apogee_member_flags.flag(2**58)
    flag_sdss4_apogee_member_fornax = sdss4_apogee_member_flags.flag(2**59)
    flag_sdss4_apogee_member_sculptor = sdss4_apogee_member_flags.flag(2**60)
    flag_sdss4_apogee_member_carina = sdss4_apogee_member_flags.flag(2**61)

    #> Astrometry
    ra = FloatField(null=True)
    dec = FloatField(null=True)
    l = FloatField(null=True)
    b = FloatField(null=True)
    plx = FloatField(null=True)
    e_plx = FloatField(null=True)
    pmra = FloatField(null=True)
    e_pmra = FloatField(null=True)
    pmde = FloatField(null=True)
    e_pmde = FloatField(null=True)
    gaia_v_rad = FloatField(null=True)
    gaia_e_v_rad = FloatField(null=True)

    # A decision was made here.
    # It would be nice to keep the original field names from each photometric survey so that
    # users know exactly which column means what. In other words, "do we be internally 
    # consistent with naming things, or be consistent with the original names?".
    
    # Unfortunately, some of the original field names are longer than 8 characters, so we 
    # would end up with many HIERARCH cards if we kept the original names. There might be
    # some other downsides if we wrote code that relied on some of those naming conventions.
    # For example, if bitfields always have the suffix `_flags` and then we wanted to create
    # documentation for any flag set (based on the `_flags` suffix instead of BitField type)
    # then some flagging things would not be documented. 

    # Neither avenue is clearly better, so we make a decision, document it, and live with it.
    # For Gaia, we use `_mag`. For 2MASS we use `_mag`. 
    
    # For unWISE they report fluxes, so we keep their naming convention where it fits within 
    # 8 characters, and document when the name differs from the original catalog.

    #> Gaia Photometry
    g_mag = FloatField(null=True)
    bp_mag = FloatField(null=True)
    rp_mag = FloatField(null=True)

    #> 2MASS Photometry
    j_mag = FloatField(null=True)
    e_j_mag = FloatField(null=True)
    h_mag = FloatField(null=True)
    e_h_mag = FloatField(null=True)
    k_mag = FloatField(null=True)
    e_k_mag = FloatField(null=True)
    ph_qual = TextField(null=True)
    bl_flg = TextField(null=True)
    cc_flg = TextField(null=True)
    #< See https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec2_2a.html 

    #> unWISE Photometry
    w1_mag = FloatField(null=True)
    e_w1_mag = FloatField(null=True)
    w1_flux = FloatField(null=True)
    w1_dflux = FloatField(null=True)
    w1_frac = FloatField(null=True)
    w2_mag = FloatField(null=True)
    e_w2_mag = FloatField(null=True)
    w2_flux = FloatField(null=True)
    w2_dflux = FloatField(null=True)
    w2_frac = FloatField(null=True)
    w1uflags = BitField(default=0, null=True)
    w2uflags = BitField(default=0, null=True)
    w1aflags = BitField(default=0, null=True)
    w2aflags = BitField(default=0, null=True)
    #< See https://catalog.unwise.me/catalogs.html
    
    flag_unwise_w1_in_core_or_wings = w1uflags.flag(2**0)
    flag_unwise_w1_in_diffraction_spike = w1uflags.flag(2**1)
    flag_unwise_w1_in_ghost = w1uflags.flag(2**2)
    flag_unwise_w1_in_first_latent = w1uflags.flag(2**3)
    flag_unwise_w1_in_second_latent = w1uflags.flag(2**4)
    flag_unwise_w1_in_circular_halo = w1uflags.flag(2**5)
    flag_unwise_w1_saturated = w1uflags.flag(2**6)
    flag_unwise_w1_in_geometric_diffraction_spike = w1uflags.flag(2**7)
    
    flag_unwise_w2_in_core_or_wings = w2uflags.flag(2**0)
    flag_unwise_w2_in_diffraction_spike = w2uflags.flag(2**1)
    flag_unwise_w2_in_ghost = w2uflags.flag(2**2)
    flag_unwise_w2_in_first_latent = w2uflags.flag(2**3)
    flag_unwise_w2_in_second_latent = w2uflags.flag(2**4)
    flag_unwise_w2_in_circular_halo = w2uflags.flag(2**5)
    flag_unwise_w2_saturated = w2uflags.flag(2**6)
    flag_unwise_w2_in_geometric_diffraction_spike = w2uflags.flag(2**7)
        
    flag_unwise_w1_in_bright_star_psf = w1aflags.flag(2**0)
    flag_unwise_w1_in_hyperleda_galaxy = w1aflags.flag(2**1)
    flag_unwise_w1_in_big_object = w1aflags.flag(2**2)
    flag_unwise_w1_pixel_in_very_bright_star_centroid = w1aflags.flag(2**3)
    flag_unwise_w1_crowdsource_saturation = w1aflags.flag(2**4)
    flag_unwise_w1_possible_nebulosity = w1aflags.flag(2**5)
    flag_unwise_w1_no_aggressive_deblend = w1aflags.flag(2**6)
    flag_unwise_w1_candidate_sources_must_be_sharp = w1aflags.flag(2**7)

    flag_unwise_w2_in_bright_star_psf = w2aflags.flag(2**0)
    flag_unwise_w2_in_hyperleda_galaxy = w2aflags.flag(2**1)
    flag_unwise_w2_in_big_object = w2aflags.flag(2**2)
    flag_unwise_w2_pixel_in_very_bright_star_centroid = w2aflags.flag(2**3)
    flag_unwise_w2_crowdsource_saturation = w2aflags.flag(2**4)
    flag_unwise_w2_possible_nebulosity = w2aflags.flag(2**5)
    flag_unwise_w2_no_aggressive_deblend = w2aflags.flag(2**6)
    flag_unwise_w2_candidate_sources_must_be_sharp = w2aflags.flag(2**7)

    #> GLIMPSE Photometry
    mag4_5 = FloatField(null=True)
    d4_5m = FloatField(null=True)
    rms_f4_5 = FloatField(null=True)
    sqf_4_5 = BitField(default=0)
    mf4_5 = BitField(default=0)
    csf = BitField(default=0)
    #< See https://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/gator_docs/

    flag_glimpse_poor_dark_pixel_current = sqf_4_5.flag(2**0)
    flag_glimpse_flat_field_questionable = sqf_4_5.flag(2**1)
    flag_glimpse_latent_image = sqf_4_5.flag(2**2)
    flag_glimpse_saturated_star_correction = sqf_4_5.flag(2**3)
    flag_glimpse_muxbleed_correction_applied = sqf_4_5.flag(2**6)
    flag_glimpse_hot_or_dead_pixels = sqf_4_5.flag(2**7)
    flag_glimpse_muxbleed_significant = sqf_4_5.flag(2**8)
    flag_glimpse_allstar_tweak_positive = sqf_4_5.flag(2**9)
    flag_glimpse_allstar_tweak_negative = sqf_4_5.flag(2**10)
    flag_glimpse_confusion_in_band_merge = sqf_4_5.flag(2**12)
    flag_glimpse_confusion_in_cross_band_merge = sqf_4_5.flag(2**13)
    flag_glimpse_column_pulldown_correction = sqf_4_5.flag(2**14)
    flag_glimpse_banding_correction = sqf_4_5.flag(2**15)
    flag_glimpse_stray_light = sqf_4_5.flag(2**16)
    flag_glimpse_no_nonlinear_correction = sqf_4_5.flag(2**18)
    flag_glimpse_saturated_star_wing_region = sqf_4_5.flag(2**19)
    flag_glimpse_pre_lumping_in_band_merge = sqf_4_5.flag(2**20)
    flag_glimpse_post_lumping_in_cross_band_merge = sqf_4_5.flag(2**21)
    flag_glimpse_edge_of_frame = sqf_4_5.flag(2**29)
    flag_glimpse_truth_list = sqf_4_5.flag(2**30)

    flag_glimpse_no_source_within_3_arcsecond = csf.flag(2**0)
    flag_glimpse_1_source_within_2p5_and_3_arcsecond = csf.flag(2**1)
    flag_glimpse_2_sources_within_2_and_2p5_arcsecond = csf.flag(2**2)
    flag_glimpse_3_sources_within_1p5_and_2_arcsecond = csf.flag(2**3)
    flag_glimpse_4_sources_within_1_and_1p5_arcsecond = csf.flag(2**4)
    flag_glimpse_5_sources_within_0p5_and_1_arcsecond = csf.flag(2**5)
    flag_glimpse_6_sources_within_0p5_arcsecond = csf.flag(2**6)
    
    #> Gaia XP Stellar Parameters (Zhang, Green & Rix 2023)
    zgr_teff = FloatField(null=True)
    zgr_e_teff = FloatField(null=True)
    zgr_logg = FloatField(null=True)
    zgr_e_logg = FloatField(null=True)
    zgr_fe_h = FloatField(null=True)
    zgr_e_fe_h = FloatField(null=True)
    zgr_e = FloatField(null=True)
    zgr_e_e = FloatField(null=True)
    zgr_plx = FloatField(null=True)
    zgr_e_plx = FloatField(null=True)
    zgr_teff_confidence = FloatField(null=True)
    zgr_logg_confidence = FloatField(null=True)
    zgr_fe_h_confidence = FloatField(null=True)
    zgr_ln_prior = FloatField(null=True)
    zgr_chi2 = FloatField(null=True)
    zgr_quality_flags = BitField(default=0)
    # See https://zenodo.org/record/7811871

    #> Bailer-Jones Distance Estimates (EDR3; 2021)
    r_med_geo = FloatField(null=True)
    r_lo_geo = FloatField(null=True)
    r_hi_geo = FloatField(null=True)
    r_med_photogeo = FloatField(null=True)
    r_lo_photogeo = FloatField(null=True)
    r_hi_photogeo = FloatField(null=True)
    bailer_jones_flags = TextField(null=True)
    # See https://dc.zah.uni-heidelberg.de/tableinfo/gedr3dist.main#note-f

    #> Reddening
    ebv = FloatField(null=True)
    e_ebv = FloatField(null=True)
    ebv_flags = BitField(default=0)
    flag_ebv_upper_limit = ebv_flags.flag(2**0)
    flag_ebv_from_zhang_2023 = ebv_flags.flag(2**1)
    flag_ebv_from_edenhofer_2023 = ebv_flags.flag(2**2)
    flag_ebv_from_sfd = ebv_flags.flag(2**3)
    flag_ebv_from_rjce_glimpse = ebv_flags.flag(2**4)
    flag_ebv_from_rjce_allwise = ebv_flags.flag(2**5)
    flag_ebv_from_bayestar_2019 = ebv_flags.flag(2**6)

    ebv_zhang_2023 = FloatField(null=True)
    e_ebv_zhang_2023 = FloatField(null=True)
    ebv_sfd = FloatField(null=True)
    e_ebv_sfd = FloatField(null=True)
    ebv_rjce_glimpse = FloatField(null=True)
    e_ebv_rjce_glimpse = FloatField(null=True)
    ebv_rjce_allwise = FloatField(null=True)
    e_ebv_rjce_allwise = FloatField(null=True)
    ebv_bayestar_2019 = FloatField(null=True)
    e_ebv_bayestar_2019 = FloatField(null=True)
    ebv_edenhofer_2023 = FloatField(null=True)
    e_ebv_edenhofer_2023 = FloatField(null=True)
    
    #> Synthetic Photometry from Gaia XP Spectra
    c_star = FloatField(null=True)
    u_jkc_mag = FloatField(null=True)
    u_jkc_mag_flag = IntegerField(null=True)
    b_jkc_mag = FloatField(null=True)
    b_jkc_mag_flag = IntegerField(null=True)
    v_jkc_mag = FloatField(null=True)
    v_jkc_mag_flag = IntegerField(null=True)
    r_jkc_mag = FloatField(null=True)
    r_jkc_mag_flag = IntegerField(null=True)
    i_jkc_mag = FloatField(null=True)
    i_jkc_mag_flag = IntegerField(null=True)
    u_sdss_mag = FloatField(null=True)
    u_sdss_mag_flag = IntegerField(null=True)
    g_sdss_mag = FloatField(null=True)
    g_sdss_mag_flag = IntegerField(null=True)
    r_sdss_mag = FloatField(null=True)
    r_sdss_mag_flag = IntegerField(null=True)
    i_sdss_mag = FloatField(null=True)
    i_sdss_mag_flag = IntegerField(null=True)
    z_sdss_mag = FloatField(null=True)
    z_sdss_mag_flag = IntegerField(null=True)
    y_ps1_mag = FloatField(null=True)
    y_ps1_mag_flag = IntegerField(null=True)
    
    #> Observations Summary
    n_boss_visits = IntegerField(null=True)
    boss_min_mjd = IntegerField(null=True)
    boss_max_mjd = IntegerField(null=True)
    n_apogee_visits = IntegerField(null=True)
    apogee_min_mjd = IntegerField(null=True)
    apogee_max_mjd = IntegerField(null=True)

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    

    @property
    def sdss5_cartons(self):
        """Return the cartons that this source is assigned."""
        mapping = get_carton_to_bit_mapping()
        indices = np.searchsorted(mapping["bit"], self.sdss5_target_bits)
        return mapping[indices]

    @property
    def sdss5_target_bits(self):
        """Return the bit positions of targeting flags that this source is assigned."""
        i, bits, cur_size = (0, [], len(self.sdss5_target_flags._buffer))
        while True:
            byte_num, byte_offset = divmod(i, 8)
            if byte_num >= cur_size:
                break
            if bool(self.sdss5_target_flags._buffer[byte_num] & (1 << byte_offset)):
                bits.append(i)
            i += 1
        return tuple(bits)

    @hybrid_method
    def assigned_to_carton_attribute(self, name, value):
        """
        An expression to evaluate whether this source is assigned to a carton with the given attribute name and value.

        :param name:
            The name of the attribute to check.
        
        :param value:
            The value of the attribute to check.
        """
        mapping = get_carton_to_bit_mapping()
        bits = np.array(mapping["bit"][mapping[name] == value], dtype=int)
        return self.is_any_sdss5_target_bit_set(*bits)

    @hybrid_method
    def assigned_to_carton_pk(self, pk):
        """
        An expression to evaluate whether this source is assigned to the given carton.

        :param pk:
            The primary key of the carton.
        """
        return self.assigned_to_carton_attribute("carton_pk", pk)

    @hybrid_method
    def assigned_to_carton_label(self, label):
        """
        An expression to evaluate whether this source is assigned to the given carton.

        :param label:
            The label of the carton.
        """
        return self.assigned_to_carton_attribute("label", label)

    @hybrid_method
    def assigned_to_program(self, program):
        """
        An expression to evaluate whether this source is assigned to any carton in the given program.

        :param program:
            The program name.
        """
        return self.assigned_to_carton_attribute("program", program)

    @hybrid_method
    def assigned_to_mapper(self, mapper):
        """
        An expression to evaluate whether this source is assigned to any carton in the given mapper.

        :param mapper:
            The mapper name.
        """
        return self.assigned_to_carton_attribute("mapper", mapper)
    
    @hybrid_method
    def assigned_to_carton_with_alt_program(self, alt_program):
        """
        An expression to evaluate whether this source is assigned to any carton with the given alternate program.

        :param alt_program:
            The alternate program name.
        """
    
        return self.assigned_to_carton_attribute("alt_program", alt_program)

    @hybrid_method
    def assigned_to_carton_with_alt_name(self, alt_name):
        """
        An expression to evaluate whether this source is assigned to any carton with the given alternate name.
        
        :param alt_name:
            The alternate name.
        """
        return self.assigned_to_carton_attribute("alt_name", alt_name)

    @hybrid_method
    def assigned_to_carton_with_name(self, name):
        """
        An expression to evaluate whether this source is assigned to any carton with the given name.
        
        :param name:
            The carton name.
        """
        return self.assigned_to_carton_attribute("name", name)
    

    @hybrid_method
    def is_sdss5_target_bit_set(self, bit):
        """
        An expression to evaluate whether this source is assigned to the carton with the given bit position.
        
        :param bit:
            The carton bit position.
        """
        return (
            (fn.length(self.sdss5_target_flags) > int(bit / 8))
        &   (fn.get_bit(self.sdss5_target_flags, int(bit)) > 0)
        )
    
    @hybrid_method
    def is_any_sdss5_target_bit_set(self, *bits):
        """
        An expression to evaluate whether this source is assigned to any carton with the given bit positions.
        
        :param bits:
            The carton bit positions.
        """
        expression = self.is_sdss5_target_bit_set(bits[0])
        if len(bits) > 1:
            for bit in bits[1:]:
                expression = expression | self.is_sdss5_target_bit_set(bit)
        return expression
    

    @property
    def spectra(self):
        """A generator that yields all spectra associated with this source."""
        for expr, column in self.dependencies():
            if Spectrum in column.model.__mro__[1:]:
                yield from column.model.select().where(expr)


@cache
def get_carton_to_bit_mapping():
    from astropy.table import Table
    t = Table.read(expand_path("$MWM_ASTRA/aux/targeting-bits/sdss5_target_3_with_groups.csv"))
    t.sort("bit")
    return t

