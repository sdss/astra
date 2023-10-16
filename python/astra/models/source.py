
from peewee import (
    AutoField,
    IntegerField,
    FloatField,
    TextField,
    ForeignKeyField,
    BigBitField,
    BigIntegerField,
    PostgresqlDatabase,
    SmallIntegerField,
    DateTimeField,
    BooleanField,
    fn,
)
import numpy as np
from playhouse.hybrid import hybrid_method
from astra.models.base import database, BaseModel
from astra.models.fields import BitField
from astra.models.spectrum import Spectrum

from astra.glossary import Glossary
from functools import cache

from astropy.table import Table
from astra.utils import expand_path


class Source(BaseModel):

    """ An astronomical source. """

    pk = AutoField(primary_key=True, help_text=Glossary.pk)

    #> Identifiers    
    sdss_id = BigIntegerField(index=True, unique=True, null=True, help_text="SDSS-5 unique identifier")
    # These identifiers are usually unique, but let's not use integrity constraints because there will be things with n_associated > 1.
    sdss4_apogee_id = TextField(index=True, unique=True, null=True, help_text="SDSS-4 DR17 APOGEE identifier")
    gaia_dr2_source_id = BigIntegerField(null=True, help_text="Gaia DR2 source identifier")
    gaia_dr3_source_id = BigIntegerField(null=True, help_text="Gaia DR3 source identifier")
    tic_v8_id = BigIntegerField(null=True, help_text="TESS Input Catalog (v8) identifier")
    healpix = IntegerField(null=True, help_text="HEALPix (128 side)")
    
    #> Targeting provenance 
    carton_0 = TextField(default="", help_text="Highest priority carton name")
    lead = TextField(null=True, help_text="Lead catalog used for cross-match")
    version_id = IntegerField(null=True, help_text="SDSS catalog version for targeting")
    catalogid = BigIntegerField(null=True, help_text=Glossary.catalogid)
    catalogid21 = BigIntegerField(null=True, help_text=Glossary.catalogid21)
    catalogid25 = BigIntegerField(null=True, help_text=Glossary.catalogid25)
    catalogid31 = BigIntegerField(null=True, help_text=Glossary.catalogid31)
    n_associated = IntegerField(null=True, help_text=Glossary.n_associated)
    n_neighborhood = IntegerField(default=-1, help_text="Sources within 3\" and G_MAG < G_MAG_source + 5")
    
    # Only do sdss5_target_flags if we are using a PostgreSQL database, as SQLite does not support it.
    if isinstance(database, PostgresqlDatabase):
        sdss5_target_flags = BigBitField(null=True, help_text=Glossary.sdss5_target_flags)

    # https://www.sdss4.org/dr17/irspec/apogee-bitmasks/
    sdss4_apogee_target1_flags = BitField(default=0, help_text="SDSS4 APOGEE1 targeting flags (1/2)")
    sdss4_apogee_target2_flags = BitField(default=0, help_text="SDSS4 APOGEE1 targeting flags (2/2)")
    sdss4_apogee2_target1_flags = BitField(default=0, help_text="SDSS4 APOGEE2 targeting flags (1/3)")
    sdss4_apogee2_target2_flags = BitField(default=0, help_text="SDSS4 APOGEE2 targeting flags (2/3)")
    sdss4_apogee2_target3_flags = BitField(default=0, help_text="SDSS4 APOGEE2 targeting flags (3/3)")
    sdss4_apogee_member_flags = BitField(default=0, help_text="SDSS4 likely cluster/galaxy member flags")
    sdss4_apogee_extra_target_flags = BitField(default=0, help_text="SDSS4 target info (aka EXTRATARG)")
    
    # sdss4_apogee_target1_flags
    flag_sdss4_apogee_faint = sdss4_apogee_target1_flags.flag(2**0, help_text="Star selected in faint bin of its cohort")
    flag_sdss4_apogee_medium = sdss4_apogee_target1_flags.flag(2**1, help_text="Star selected in medium bin of its cohort")
    flag_sdss4_apogee_bright = sdss4_apogee_target1_flags.flag(2**2, help_text="Star selected in bright bin of its cohort")
    flag_sdss4_apogee_irac_dered = sdss4_apogee_target1_flags.flag(2**3, help_text="Selected w/ RJCE-IRAC dereddening")
    flag_sdss4_apogee_wise_dered = sdss4_apogee_target1_flags.flag(2**4, help_text="Selected w/ RJCE-WISE dereddening")
    flag_sdss4_apogee_sfd_dered = sdss4_apogee_target1_flags.flag(2**5, help_text="Selected w/ SFD dereddening")
    flag_sdss4_apogee_no_dered = sdss4_apogee_target1_flags.flag(2**6, help_text="Selected w/ no dereddening")
    flag_sdss4_apogee_wash_giant = sdss4_apogee_target1_flags.flag(2**7, help_text="Selected as giant using Washington photometry")
    flag_sdss4_apogee_wash_dwarf = sdss4_apogee_target1_flags.flag(2**8, help_text="Selected as dwarf using Washington photometry")
    flag_sdss4_apogee_sci_cluster = sdss4_apogee_target1_flags.flag(2**9, help_text="Selected as probable cluster member")
    flag_sdss4_apogee_extended = sdss4_apogee_target1_flags.flag(2**10, help_text="Extended object")
    flag_sdss4_apogee_short = sdss4_apogee_target1_flags.flag(2**11, help_text="Selected as 'short' (~3 visit) cohort target (includes 1-visit samples)")
    flag_sdss4_apogee_intermediate = sdss4_apogee_target1_flags.flag(2**12, help_text="Selected as 'intermediate' cohort (~6-visit) target")
    flag_sdss4_apogee_long = sdss4_apogee_target1_flags.flag(2**13, help_text="Selected as 'long' cohort (~12- or 24-visit) target")
    flag_sdss4_apogee_do_not_observe = sdss4_apogee_target1_flags.flag(2**14, help_text="Do not observe (again) -- undesired dwarf, galaxy, etc")
    flag_sdss4_apogee_serendipitous = sdss4_apogee_target1_flags.flag(2**15, help_text="Serendipitous interesting target to be re-observed")
    flag_sdss4_apogee_first_light = sdss4_apogee_target1_flags.flag(2**16, help_text="First Light plate target")
    flag_sdss4_apogee_ancillary = sdss4_apogee_target1_flags.flag(2**17, help_text="Ancillary target")
    flag_sdss4_apogee_m31_cluster = sdss4_apogee_target1_flags.flag(2**18, help_text="M31 Clusters (Allende Prieto, Schiavon, Bizyaev, OConnell, Shetrone)")
    flag_sdss4_apogee_mdwarf = sdss4_apogee_target1_flags.flag(2**19, help_text="RVs of M Dwarfs (Blake, Mahadevan, Hearty, Deshpande, Nidever, Bender, Crepp, Carlberg, Terrien, Schneider) -- both original list and second-round extension")
    flag_sdss4_apogee_hires = sdss4_apogee_target1_flags.flag(2**20, help_text="Stars with Optical Hi-Res Spectra (Fabbian, Allende Prieto, Smith, Cunha)")
    flag_sdss4_apogee_old_star = sdss4_apogee_target1_flags.flag(2**21, help_text="Oldest Stars in Galaxy (Harding, Johnson)")
    flag_sdss4_apogee_disk_red_giant = sdss4_apogee_target1_flags.flag(2**22, help_text="Ages/Compositions? of Disk Red Giants (Johnson, Epstein, Pinsonneault, Lai, Bird, Schonrich, Chiappini)")
    flag_sdss4_apogee_kepler_eb = sdss4_apogee_target1_flags.flag(2**23, help_text="Kepler EBs (Mahadevan, Fleming, Bender, Deshpande, Hearty, Nidever, Terrien)")
    flag_sdss4_apogee_gc_pal1 = sdss4_apogee_target1_flags.flag(2**24, help_text="Globular Cluster Pops in the MW (Simmerer, Ivans, Shetrone)")
    flag_sdss4_apogee_massive_star = sdss4_apogee_target1_flags.flag(2**25, help_text="Massive Stars in the MW (Herrero, Garcia-Garcia, Ramirez-Alegria)")
    flag_sdss4_apogee_sgr_dsph = sdss4_apogee_target1_flags.flag(2**26, help_text="Sgr (dSph) member")
    flag_sdss4_apogee_kepler_seismo = sdss4_apogee_target1_flags.flag(2**27, help_text="Kepler asteroseismology program target (Epstein)")
    flag_sdss4_apogee_kepler_host = sdss4_apogee_target1_flags.flag(2**28, help_text="Planet-host program target (Epstein)")
    flag_sdss4_apogee_faint_extra = sdss4_apogee_target1_flags.flag(2**29, help_text="'Faint' target in low-target-density fields")
    flag_sdss4_apogee_segue_overlap = sdss4_apogee_target1_flags.flag(2**30, help_text="SEGUE overlap")

    # sdss4_apogee_target2_flags
    flag_sdss4_apogee_light_trap = sdss4_apogee_target2_flags.flag(2**0, help_text="Light trap")
    flag_sdss4_apogee_flux_standard = sdss4_apogee_target2_flags.flag(2**1, help_text="Flux standard")
    flag_sdss4_apogee_standard_star = sdss4_apogee_target2_flags.flag(2**2, help_text="Stellar abundance/parameters standard")
    flag_sdss4_apogee_rv_standard = sdss4_apogee_target2_flags.flag(2**3, help_text="RV standard")
    flag_sdss4_apogee_sky = sdss4_apogee_target2_flags.flag(2**4, help_text="Sky")
    flag_sdss4_apogee_sky_bad = sdss4_apogee_target2_flags.flag(2**5, help_text="Selected as sky but IDed as bad (via visual examination or observation)")
    flag_sdss4_apogee_guide_star = sdss4_apogee_target2_flags.flag(2**6, help_text="Guide star")
    flag_sdss4_apogee_bundle_hole = sdss4_apogee_target2_flags.flag(2**7, help_text="Bundle hole")
    flag_sdss4_apogee_telluric_bad = sdss4_apogee_target2_flags.flag(2**8, help_text="Selected as telluric std but IDed as too red (via SIMBAD or observation)")
    flag_sdss4_apogee_telluric = sdss4_apogee_target2_flags.flag(2**9, help_text="Hot (telluric) standard")
    flag_sdss4_apogee_calib_cluster = sdss4_apogee_target2_flags.flag(2**10, help_text="Known calibration cluster member")
    flag_sdss4_apogee_bulge_giant = sdss4_apogee_target2_flags.flag(2**11, help_text="Selected as probable giant in bulge")
    flag_sdss4_apogee_bulge_super_giant = sdss4_apogee_target2_flags.flag(2**12, help_text="Selected as probable supergiant in bulge")
    flag_sdss4_apogee_embedded_cluster_star = sdss4_apogee_target2_flags.flag(2**13, help_text="Young nebulous clusters (Covey, Tan)")
    flag_sdss4_apogee_long_bar = sdss4_apogee_target2_flags.flag(2**14, help_text="Milky Way Long Bar (Zasowski)")
    flag_sdss4_apogee_emission_star = sdss4_apogee_target2_flags.flag(2**15, help_text="Be Emission Line Stars (Chojnowski, Whelan)")
    flag_sdss4_apogee_kepler_cool_dwarf = sdss4_apogee_target2_flags.flag(2**16, help_text="Kepler Cool Dwarfs (van Saders)")
    flag_sdss4_apogee_mir_cluster_star = sdss4_apogee_target2_flags.flag(2**17, help_text="Outer Disk MIR Clusters (Beaton)")
    flag_sdss4_apogee_rv_monitor_ic348 = sdss4_apogee_target2_flags.flag(2**18, help_text="RV Variability in IC348 (Nidever, Covey)")
    flag_sdss4_apogee_rv_monitor_kepler = sdss4_apogee_target2_flags.flag(2**19, help_text="RV Variability for Kepler Planet Hosts and Binaries (Deshpande, Fleming, Mahadevan)")
    flag_sdss4_apogee_ges_calibrate = sdss4_apogee_target2_flags.flag(2**20, help_text="Gaia-ESO calibration targets")
    flag_sdss4_apogee_bulge_rv_verify = sdss4_apogee_target2_flags.flag(2**21, help_text="RV Verification (Nidever)")
    flag_sdss4_apogee_1m_target = sdss4_apogee_target2_flags.flag(2**22, help_text="Selected as a 1-m target (Holtzman)")

    # sdss4_apogee2_target1_flags
    flag_sdss4_apogee2_onebit_gt_0_5 = sdss4_apogee2_target1_flags.flag(2**0, help_text="Selected in single (J-Ks)o > 0.5 color bin")
    flag_sdss4_apogee2_twobit_0_5_to_0_8 = sdss4_apogee2_target1_flags.flag(2**1, help_text="Selected in 'blue' 0.5 < (J-Ks)o < 0.8 color bin")
    flag_sdss4_apogee2_twobit_gt_0_8 = sdss4_apogee2_target1_flags.flag(2**2, help_text="Selected in 'red' (J-Ks)o > 0.8 color bin")
    flag_sdss4_apogee2_irac_dered = sdss4_apogee2_target1_flags.flag(2**3, help_text="Selected with RJCE-IRAC dereddening")
    flag_sdss4_apogee2_wise_dered = sdss4_apogee2_target1_flags.flag(2**4, help_text="Selected with RJCE-WISE dereddening")
    flag_sdss4_apogee2_sfd_dered = sdss4_apogee2_target1_flags.flag(2**5, help_text="Selected with SFD_EBV dereddening")
    flag_sdss4_apogee2_no_dered = sdss4_apogee2_target1_flags.flag(2**6, help_text="Selected with no dereddening")
    flag_sdss4_apogee2_wash_giant = sdss4_apogee2_target1_flags.flag(2**7, help_text="Selected as Wash+DDO51 photometric giant")
    flag_sdss4_apogee2_wash_dwarf = sdss4_apogee2_target1_flags.flag(2**8, help_text="Selected as Wash+DDO51 photometric dwarf")
    flag_sdss4_apogee2_sci_cluster = sdss4_apogee2_target1_flags.flag(2**9, help_text="Science cluster candidate member")
    flag_sdss4_apogee2_cluster_candidate = sdss4_apogee2_target1_flags.flag(2**10, help_text="Selected as globular cluster candidate")
    flag_sdss4_apogee2_short = sdss4_apogee2_target1_flags.flag(2**11, help_text="Selected as part of a short cohort")
    flag_sdss4_apogee2_medium = sdss4_apogee2_target1_flags.flag(2**12, help_text="Selected as part of a medium cohort")
    flag_sdss4_apogee2_long = sdss4_apogee2_target1_flags.flag(2**13, help_text="Selected as part of a long cohort")
    flag_sdss4_apogee2_normal_sample = sdss4_apogee2_target1_flags.flag(2**14, help_text="Selected as part of the random sample")
    flag_sdss4_apogee2_manga_led = sdss4_apogee2_target1_flags.flag(2**15, help_text="Star on a shared MaNGA-led design")
    flag_sdss4_apogee2_onebin_gt_0_3 = sdss4_apogee2_target1_flags.flag(2**16, help_text="Selected in single (J-Ks)o > 0.3 color bin")
    flag_sdss4_apogee2_wash_noclass = sdss4_apogee2_target1_flags.flag(2**17, help_text="Selected because it has no W+D classification")
    flag_sdss4_apogee2_stream_member = sdss4_apogee2_target1_flags.flag(2**18, help_text="Selected as confirmed halo tidal stream member")
    flag_sdss4_apogee2_stream_candidate = sdss4_apogee2_target1_flags.flag(2**19, help_text="Selected as potential halo tidal stream member (based on photometry)")
    flag_sdss4_apogee2_dsph_member = sdss4_apogee2_target1_flags.flag(2**20, help_text="Selected as confirmed dSph member (non Sgr)")
    flag_sdss4_apogee2_dsph_candidate = sdss4_apogee2_target1_flags.flag(2**21, help_text="Selected as potential dSph member (non Sgr) (based on photometry)")
    flag_sdss4_apogee2_magcloud_member = sdss4_apogee2_target1_flags.flag(2**22, help_text="Selected as confirmed Mag Cloud member")
    flag_sdss4_apogee2_magcloud_candidate = sdss4_apogee2_target1_flags.flag(2**23, help_text="Selected as potential Mag Cloud member (based on photometry)")
    flag_sdss4_apogee2_rrlyr = sdss4_apogee2_target1_flags.flag(2**24, help_text="Selected as an RR Lyrae star")
    flag_sdss4_apogee2_bulge_rc = sdss4_apogee2_target1_flags.flag(2**25, help_text="Selected as a bulge candidate RC star")
    flag_sdss4_apogee2_sgr_dsph = sdss4_apogee2_target1_flags.flag(2**26, help_text="Selected as confirmed Sgr core/stream member")
    flag_sdss4_apogee2_apokasc_giant = sdss4_apogee2_target1_flags.flag(2**27, help_text="Selected as part of APOKASC 'giant' sample")
    flag_sdss4_apogee2_apokasc_dwarf = sdss4_apogee2_target1_flags.flag(2**28, help_text="Selected as part of APOKASC 'dwarf' sample")
    flag_sdss4_apogee2_faint_extra = sdss4_apogee2_target1_flags.flag(2**29, help_text="'Faint' star (fainter than cohort limit; not required to reach survey S/N requirement)")
    flag_sdss4_apogee2_apokasc = sdss4_apogee2_target1_flags.flag(2**30, help_text="Selected as part of the APOKASC program (incl. seismic/gyro targets and others, both the Cygnus field and K2)")

    # sdss4_apogee2_target2_flags
    flag_sdss4_apogee2_k2_gap = sdss4_apogee2_target2_flags.flag(2**0, help_text="K2 Galactic Archeology Program Star")
    flag_sdss4_apogee2_ccloud_as4 = sdss4_apogee2_target2_flags.flag(2**1, help_text="California Cloud target")
    flag_sdss4_apogee2_standard_star = sdss4_apogee2_target2_flags.flag(2**2, help_text="Stellar parameters/abundance standard")
    flag_sdss4_apogee2_rv_standard = sdss4_apogee2_target2_flags.flag(2**3, help_text="Stellar RV standard")
    flag_sdss4_apogee2_sky = sdss4_apogee2_target2_flags.flag(2**4, help_text="Sky fiber")
    flag_sdss4_apogee2_external_calib = sdss4_apogee2_target2_flags.flag(2**5, help_text="External survey calibration target (generic flag; others below dedicated to specific surveys)")
    flag_sdss4_apogee2_internal_calib = sdss4_apogee2_target2_flags.flag(2**6, help_text="Internal survey calibration target (observed in at least 2 of: APOGEE-1, -2N, -2S)")
    flag_sdss4_apogee2_disk_substructure_member = sdss4_apogee2_target2_flags.flag(2**7, help_text="Bright time extension: outer disk substructure (Triand, GASS, and A13) members")
    flag_sdss4_apogee2_disk_substructure_candidate = sdss4_apogee2_target2_flags.flag(2**8, help_text="Bright time extension: outer disk substructure (Triand, GASS, and A13) candidates")
    flag_sdss4_apogee2_telluric = sdss4_apogee2_target2_flags.flag(2**9, help_text="Telluric standard")
    flag_sdss4_apogee2_calib_cluster = sdss4_apogee2_target2_flags.flag(2**10, help_text="Selected as calibration cluster member")
    flag_sdss4_apogee2_k2_planet_host = sdss4_apogee2_target2_flags.flag(2**11, help_text="Planet host in the K2 field")
    flag_sdss4_apogee2_tidal_binary = sdss4_apogee2_target2_flags.flag(2**12, help_text="Ancillary KOI Program (Simonian)")
    flag_sdss4_apogee2_literature_calib = sdss4_apogee2_target2_flags.flag(2**13, help_text="Overlap with high-resolution literature studies")
    flag_sdss4_apogee2_ges_overlap = sdss4_apogee2_target2_flags.flag(2**14, help_text="Overlap with Gaia-ESO")
    flag_sdss4_apogee2_argos_overlap = sdss4_apogee2_target2_flags.flag(2**15, help_text="Overlap with ARGOS")
    flag_sdss4_apogee2_gaia_overlap = sdss4_apogee2_target2_flags.flag(2**16, help_text="Overlap with Gaia")
    flag_sdss4_apogee2_galah_overlap = sdss4_apogee2_target2_flags.flag(2**17, help_text="Overlap with GALAH")
    flag_sdss4_apogee2_rave_overlap = sdss4_apogee2_target2_flags.flag(2**18, help_text="Overlap with RAVE")
    flag_sdss4_apogee2_commis_south_spec = sdss4_apogee2_target2_flags.flag(2**19, help_text="Commissioning special targets for APOGEE2S")
    flag_sdss4_apogee2_halo_member = sdss4_apogee2_target2_flags.flag(2**20, help_text="Halo Member")
    flag_sdss4_apogee2_halo_candidate = sdss4_apogee2_target2_flags.flag(2**21, help_text="Halo Candidate")
    flag_sdss4_apogee2_1m_target = sdss4_apogee2_target2_flags.flag(2**22, help_text="Selected as a 1-m target")
    flag_sdss4_apogee2_mod_bright_limit = sdss4_apogee2_target2_flags.flag(2**23, help_text="Selected in a cohort with H>10 rather than H>7")
    flag_sdss4_apogee2_cis = sdss4_apogee2_target2_flags.flag(2**24, help_text="Carnegie program target")
    flag_sdss4_apogee2_cntac = sdss4_apogee2_target2_flags.flag(2**25, help_text="Chilean community target")
    flag_sdss4_apogee2_external = sdss4_apogee2_target2_flags.flag(2**26, help_text="Proprietary external target")
    flag_sdss4_apogee2_cvz_as4_obaf = sdss4_apogee2_target2_flags.flag(2**27, help_text="OBAF stars selected for multi-epoc observations Andrew T.")
    flag_sdss4_apogee2_cvz_as4_gi = sdss4_apogee2_target2_flags.flag(2**28, help_text="Submitted program to be on CVZ plate (Known Planets, ATL, Tayar-Subgiant, Canas-Cool-dwarf)")
    flag_sdss4_apogee2_cvz_as4_ctl = sdss4_apogee2_target2_flags.flag(2**29, help_text="Filler CTL star selected from the TESS Input Catalog")
    flag_sdss4_apogee2_cvz_as4_giant = sdss4_apogee2_target2_flags.flag(2**30, help_text="Filler Giant selected with RPMJ")

    # sdss4_apogee2_target3_flags
    flag_sdss4_apogee2_koi = sdss4_apogee2_target3_flags.flag(2**0, help_text="Selected as part of the long cadence KOI study")
    flag_sdss4_apogee2_eb = sdss4_apogee2_target3_flags.flag(2**1, help_text="Selected as part of the EB program")
    flag_sdss4_apogee2_koi_control = sdss4_apogee2_target3_flags.flag(2**2, help_text="Selected as part of the long cadence KOI 'control sample'")
    flag_sdss4_apogee2_mdwarf = sdss4_apogee2_target3_flags.flag(2**3, help_text="Selected as part of the M dwarf study")
    flag_sdss4_apogee2_substellar_companions = sdss4_apogee2_target3_flags.flag(2**4, help_text="Selected as part of the substellar companion search")
    flag_sdss4_apogee2_young_cluster = sdss4_apogee2_target3_flags.flag(2**5, help_text="Selected as part of the young cluster study (IN-SYNC)")
    flag_sdss4_apogee2_k2 = sdss4_apogee2_target3_flags.flag(2**6, help_text="Selected as part of the K2 program (BTX and Main Survey)")
    flag_sdss4_apogee2_object = sdss4_apogee2_target3_flags.flag(2**7, help_text="This object is an APOGEE-2 target")
    flag_sdss4_apogee2_ancillary = sdss4_apogee2_target3_flags.flag(2**8, help_text="Selected as an ancillary target")
    flag_sdss4_apogee2_massive_star = sdss4_apogee2_target3_flags.flag(2**9, help_text="Selected as part of the Massive Star program")
    flag_sdss4_apogee2_qso = sdss4_apogee2_target3_flags.flag(2**10, help_text="Ancillary QSO pilot program (Albareti)")
    flag_sdss4_apogee2_cepheid = sdss4_apogee2_target3_flags.flag(2**11, help_text="Ancillary Cepheid sparse targets (Beaton)")
    flag_sdss4_apogee2_low_av_windows = sdss4_apogee2_target3_flags.flag(2**12, help_text="Ancillary Deep Disk sample (Bovy)")
    flag_sdss4_apogee2_be_star = sdss4_apogee2_target3_flags.flag(2**13, help_text="Ancillary ASHELS sample (Chojnowski)")
    flag_sdss4_apogee2_young_moving_group = sdss4_apogee2_target3_flags.flag(2**14, help_text="Ancillary young moving group members (Downes)")
    flag_sdss4_apogee2_ngc6791 = sdss4_apogee2_target3_flags.flag(2**15, help_text="Ancillary NGC 6791 star (Geisler)")
    flag_sdss4_apogee2_label_star = sdss4_apogee2_target3_flags.flag(2**16, help_text="Ancillary Cannon calibrator Sample (Ness)")
    flag_sdss4_apogee2_faint_kepler_giants = sdss4_apogee2_target3_flags.flag(2**17, help_text="Ancillary APOKASC faint giants (Pinsonneault)")
    flag_sdss4_apogee2_w345 = sdss4_apogee2_target3_flags.flag(2**18, help_text="Ancillary W3/4/5 star forming complex (Roman-Lopes)")
    flag_sdss4_apogee2_massive_evolved = sdss4_apogee2_target3_flags.flag(2**19, help_text="Ancillary massive/evolved star targets (Stringfellow)")
    flag_sdss4_apogee2_extinction = sdss4_apogee2_target3_flags.flag(2**20, help_text="Ancillary extinction targets (Schlafly)")
    flag_sdss4_apogee2_kepler_mdwarf_koi = sdss4_apogee2_target3_flags.flag(2**21, help_text="Ancillary M dwarf targets (Smith)")
    flag_sdss4_apogee2_agb = sdss4_apogee2_target3_flags.flag(2**22, help_text="Ancillary AGB sample (Zamora)")
    flag_sdss4_apogee2_m33 = sdss4_apogee2_target3_flags.flag(2**23, help_text="Ancillary M33 Program (Anguiano)")
    flag_sdss4_apogee2_ultracool = sdss4_apogee2_target3_flags.flag(2**24, help_text="Ancillary Ultracool Dwarfs Program (Burgasser)")
    flag_sdss4_apogee2_distant_segue_giants = sdss4_apogee2_target3_flags.flag(2**25, help_text="Ancillary Distant SEGUE Giants program (Harding)")
    flag_sdss4_apogee2_cepheid_mapping = sdss4_apogee2_target3_flags.flag(2**26, help_text="Ancillary Cepheid Mapping Program (Inno)")
    flag_sdss4_apogee2_sa57 = sdss4_apogee2_target3_flags.flag(2**27, help_text="Ancillary SA57 Kapteyn Field Program (Majewski)")
    flag_sdss4_apogee2_k2_mdwarf = sdss4_apogee2_target3_flags.flag(2**28, help_text="Ancillary K2 M dwarf Program (Smith)")
    flag_sdss4_apogee2_rvvar = sdss4_apogee2_target3_flags.flag(2**29, help_text="Ancillary RV Variables Program (Troup)")
    flag_sdss4_apogee2_m31 = sdss4_apogee2_target3_flags.flag(2**30, help_text="Ancillary M31 Program (Zasowski)")

    # sdss4_apogee_extra_target_flags
    flag_sdss4_apogee_not_main = sdss4_apogee_extra_target_flags.flag(2**0, help_text="Not a main sample target")
    flag_sdss4_apogee_commissioning = sdss4_apogee_extra_target_flags.flag(2**1, help_text="Commissioning observation")
    flag_sdss4_apogee_telluric = sdss4_apogee_extra_target_flags.flag(2**2, help_text="Targeted as telluric")
    flag_sdss4_apogee_apo1m = sdss4_apogee_extra_target_flags.flag(2**3, help_text="APO/NMSU 1M observation")
    flag_sdss4_apogee_duplicate = sdss4_apogee_extra_target_flags.flag(2**4, help_text="Non-primary (not highest S/N) duplicate, excluding SDSS-5")

    # sdss4_apogee_member_flags
    flag_sdss4_apogee_member_m92 = sdss4_apogee_member_flags.flag(2**0, help_text="Likely member of M92")
    flag_sdss4_apogee_member_m15 = sdss4_apogee_member_flags.flag(2**1, help_text="Likely member of M15")
    flag_sdss4_apogee_member_m53 = sdss4_apogee_member_flags.flag(2**2, help_text="Likely member of M53")
    flag_sdss4_apogee_member_ngc_5466 = sdss4_apogee_member_flags.flag(2**3, help_text="Likely member of NGC 5466")
    flag_sdss4_apogee_member_ngc_4147 = sdss4_apogee_member_flags.flag(2**4, help_text="Likely member of NGC 4147")
    flag_sdss4_apogee_member_m2 = sdss4_apogee_member_flags.flag(2**5, help_text="Likely member of M2")
    flag_sdss4_apogee_member_m13 = sdss4_apogee_member_flags.flag(2**6, help_text="Likely member of M13")
    flag_sdss4_apogee_member_m3 = sdss4_apogee_member_flags.flag(2**7,  help_text="Likely member of M3")
    flag_sdss4_apogee_member_m5 = sdss4_apogee_member_flags.flag(2**8,  help_text="Likely member of M5")
    flag_sdss4_apogee_member_m12 = sdss4_apogee_member_flags.flag(2**9,  help_text="Likely member of M12")
    flag_sdss4_apogee_member_m107 = sdss4_apogee_member_flags.flag(2**10, help_text="Likely member of M107")
    flag_sdss4_apogee_member_m71 = sdss4_apogee_member_flags.flag(2**11, help_text="Likely member of M71")
    flag_sdss4_apogee_member_ngc_2243 = sdss4_apogee_member_flags.flag(2**12, help_text="Likely member of NGC 2243")
    flag_sdss4_apogee_member_be29 = sdss4_apogee_member_flags.flag(2**13, help_text="Likely member of Be29")
    flag_sdss4_apogee_member_ngc_2158 = sdss4_apogee_member_flags.flag(2**14,  help_text="Likely member of NGC 2158")
    flag_sdss4_apogee_member_m35 = sdss4_apogee_member_flags.flag(2**15,  help_text="Likely member of M35")
    flag_sdss4_apogee_member_ngc_2420 = sdss4_apogee_member_flags.flag(2**16,  help_text="Likely member of NGC 2420")
    flag_sdss4_apogee_member_ngc_188 = sdss4_apogee_member_flags.flag(2**17, help_text="Likely member of NGC 188")
    flag_sdss4_apogee_member_m67 = sdss4_apogee_member_flags.flag(2**18, help_text="Likely member of M67")
    flag_sdss4_apogee_member_ngc_7789 = sdss4_apogee_member_flags.flag(2**19, help_text="Likely member of NGC 7789")
    flag_sdss4_apogee_member_pleiades = sdss4_apogee_member_flags.flag(2**20, help_text="Likely member of Pleiades")
    flag_sdss4_apogee_member_ngc_6819 = sdss4_apogee_member_flags.flag(2**21, help_text="Likely member of NGC 6819")
    flag_sdss4_apogee_member_coma_berenices = sdss4_apogee_member_flags.flag(2**22, help_text="Likely member of Coma Berenices")
    flag_sdss4_apogee_member_ngc_6791 = sdss4_apogee_member_flags.flag(2**23, help_text="Likely member of NGC 6791")
    flag_sdss4_apogee_member_ngc_5053 = sdss4_apogee_member_flags.flag(2**24, help_text="Likely member of NGC 5053")
    flag_sdss4_apogee_member_m68 = sdss4_apogee_member_flags.flag(2**25, help_text="Likely member of M68")
    flag_sdss4_apogee_member_ngc_6397 = sdss4_apogee_member_flags.flag(2**26, help_text="Likely member of NGC 6397")
    flag_sdss4_apogee_member_m55 = sdss4_apogee_member_flags.flag(2**27, help_text="Likely member of M55")
    flag_sdss4_apogee_member_ngc_5634 = sdss4_apogee_member_flags.flag(2**28, help_text="Likely member of NGC 5634")
    flag_sdss4_apogee_member_m22 = sdss4_apogee_member_flags.flag(2**29, help_text="Likely member of M22")
    flag_sdss4_apogee_member_m79 = sdss4_apogee_member_flags.flag(2**30, help_text="Likely member of M79")
    flag_sdss4_apogee_member_ngc_3201 = sdss4_apogee_member_flags.flag(2**31, help_text="Likely member of NGC 3201")
    flag_sdss4_apogee_member_m10 = sdss4_apogee_member_flags.flag(2**32, help_text="Likely member of M10")
    flag_sdss4_apogee_member_ngc_6752 = sdss4_apogee_member_flags.flag(2**33,help_text="Likely member of NGC 6752")
    flag_sdss4_apogee_member_omega_centauri = sdss4_apogee_member_flags.flag(2**34, help_text="Likely member of Omega Centauri")
    flag_sdss4_apogee_member_m54 = sdss4_apogee_member_flags.flag(2**35, help_text="Likely member of M54")
    flag_sdss4_apogee_member_ngc_6229 = sdss4_apogee_member_flags.flag(2**36, help_text="Likely member of NGC 6229")
    flag_sdss4_apogee_member_pal5 = sdss4_apogee_member_flags.flag(2**37, help_text="Likely member of Pal5")
    flag_sdss4_apogee_member_ngc_6544 = sdss4_apogee_member_flags.flag(2**38, help_text="Likely member of NGC 6544")
    flag_sdss4_apogee_member_ngc_6522 = sdss4_apogee_member_flags.flag(2**39, help_text="Likely member of NGC 6522")
    flag_sdss4_apogee_member_ngc_288 = sdss4_apogee_member_flags.flag(2**40, help_text="Likely member of NGC 288")
    flag_sdss4_apogee_member_ngc_362 = sdss4_apogee_member_flags.flag(2**41, help_text="Likely member of NGC 362")
    flag_sdss4_apogee_member_ngc_1851 = sdss4_apogee_member_flags.flag(2**42, help_text="Likely member of NGC 1851")
    flag_sdss4_apogee_member_m4 = sdss4_apogee_member_flags.flag(2**43, help_text="Likely member of M4")
    flag_sdss4_apogee_member_ngc_2808 = sdss4_apogee_member_flags.flag(2**44, help_text="Likely member of NGC 2808")
    flag_sdss4_apogee_member_pal6 = sdss4_apogee_member_flags.flag(2**45, help_text="Likely member of Pal6")
    flag_sdss4_apogee_member_47tuc = sdss4_apogee_member_flags.flag(2**46, help_text="Likely member of 47 Tucane")
    flag_sdss4_apogee_member_pal1 = sdss4_apogee_member_flags.flag(2**47, help_text="Likely member of Pal1")
    flag_sdss4_apogee_member_ngc_6539 = sdss4_apogee_member_flags.flag(2**48, help_text="Likely member of NGC 6539")
    flag_sdss4_apogee_member_ngc_6388 = sdss4_apogee_member_flags.flag(2**49, help_text="Likely member of NGC 6388")
    flag_sdss4_apogee_member_ngc_6441 = sdss4_apogee_member_flags.flag(2**50, help_text="Likely member of NGC 6441")
    flag_sdss4_apogee_member_ngc_6316 = sdss4_apogee_member_flags.flag(2**51, help_text="Likely member of NGC 6316")
    flag_sdss4_apogee_member_ngc_6760 = sdss4_apogee_member_flags.flag(2**52, help_text="Likely member of NGC 6760")
    flag_sdss4_apogee_member_ngc_6553 = sdss4_apogee_member_flags.flag(2**53, help_text="Likely member of NGC 6553")
    flag_sdss4_apogee_member_ngc_6528 = sdss4_apogee_member_flags.flag(2**54, help_text="Likely member of NGC 6528")
    flag_sdss4_apogee_member_draco = sdss4_apogee_member_flags.flag(2**55, help_text="Likely member of Draco")
    flag_sdss4_apogee_member_urminor = sdss4_apogee_member_flags.flag(2**56, help_text="Likely member of Ursa Minor")
    flag_sdss4_apogee_member_bootes1 = sdss4_apogee_member_flags.flag(2**57, help_text="Likely member of Bootes 1")
    flag_sdss4_apogee_member_sexans = sdss4_apogee_member_flags.flag(2**58, help_text="Likely member of Sextans")
    flag_sdss4_apogee_member_fornax = sdss4_apogee_member_flags.flag(2**59, help_text="Likely member of Fornax")
    flag_sdss4_apogee_member_sculptor = sdss4_apogee_member_flags.flag(2**60, help_text="Likely member of Sculptor")
    flag_sdss4_apogee_member_carina = sdss4_apogee_member_flags.flag(2**61, help_text="Likely member of Carina")

    #> Astrometry
    ra = FloatField(null=True, help_text="Right ascension [deg]")
    dec = FloatField(null=True, help_text="Declination [deg]")
    plx = FloatField(null=True, help_text="Parallax [mas]")
    e_plx = FloatField(null=True, help_text="Error on parallax [mas]")
    pmra = FloatField(null=True, help_text="Proper motion in RA [mas/yr]")
    e_pmra = FloatField(null=True, help_text="Error on proper motion in RA [mas/yr]")
    pmde = FloatField(null=True, help_text="Proper motion in DEC [mas/yr]")
    e_pmde = FloatField(null=True, help_text="Error on proper motion in DEC [mas/yr]")
    gaia_v_rad = FloatField(null=True, help_text="Gaia radial velocity [km/s]")
    gaia_e_v_rad = FloatField(null=True, help_text="Error on Gaia radial velocity [km/s]")

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
    g_mag = FloatField(null=True, help_text="Gaia DR3 mean G band magnitude [mag]")
    bp_mag = FloatField(null=True, help_text="Gaia DR3 mean BP band magnitude [mag]")
    rp_mag = FloatField(null=True, help_text="Gaia DR3 mean RP band magnitude [mag]")

    #> 2MASS Photometry
    j_mag = FloatField(null=True, help_text="2MASS J band magnitude [mag]")
    e_j_mag = FloatField(null=True, help_text="Error on 2MASS J band magnitude [mag]")
    h_mag = FloatField(null=True, help_text="2MASS H band magnitude [mag]")
    e_h_mag = FloatField(null=True, help_text="Error on 2MASS H band magnitude [mag]")
    k_mag = FloatField(null=True, help_text="2MASS K band magnitude [mag]")
    e_k_mag = FloatField(null=True, help_text="Error on 2MASS K band magnitude [mag]")
    ph_qual = TextField(null=True, help_text="2MASS photometric quality flag")
    bl_flg = TextField(null=True, help_text="Number of components fit per band (JHK)")
    cc_flg = TextField(null=True, help_text="Contamination and confusion flag")
    #< See https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec2_2a.html 

    #> unWISE Photometry
    w1_flux = FloatField(null=True, help_text="W1 flux [Vega nMgy]")
    w1_dflux = FloatField(null=True, help_text="Error on W1 flux [Vega nMgy]")
    w2_flux = FloatField(null=True, help_text="W2 flux [Vega nMgy]")
    w2_dflux = FloatField(null=True, help_text="Error on W2 flux [Vega nMgy]")
    w1_frac = FloatField(null=True, help_text="Fraction of W1 flux from this object")
    w2_frac = FloatField(null=True, help_text="Fraction of W2 flux from this object")
    w1uflags = BitField(default=0, null=True, help_text="unWISE flags for W1")
    w2uflags = BitField(default=0, null=True, help_text="unWISE flags for W2")
    w1aflags = BitField(default=0, null=True, help_text="Additional flags for W1")
    w2aflags = BitField(default=0, null=True, help_text="Additional flags for W2")
    #< See https://catalog.unwise.me/catalogs.html
    
    flag_unwise_w1_in_core_or_wings = w1uflags.flag(2**0, "In core or wings")
    flag_unwise_w1_in_diffraction_spike = w1uflags.flag(2**1, "In diffraction spike")
    flag_unwise_w1_in_ghost = w1uflags.flag(2**2, "In ghost")
    flag_unwise_w1_in_first_latent = w1uflags.flag(2**3, "In first latent")
    flag_unwise_w1_in_second_latent = w1uflags.flag(2**4, "In second latent")
    flag_unwise_w1_in_circular_halo = w1uflags.flag(2**5, "In circular halo")
    flag_unwise_w1_saturated = w1uflags.flag(2**6, "Saturated")
    flag_unwise_w1_in_geometric_diffraction_spike = w1uflags.flag(2**7, "In geometric diffraction spike")
    
    flag_unwise_w2_in_core_or_wings = w2uflags.flag(2**0, "In core or wings")
    flag_unwise_w2_in_diffraction_spike = w2uflags.flag(2**1, "In diffraction spike")
    flag_unwise_w2_in_ghost = w2uflags.flag(2**2, "In ghost")
    flag_unwise_w2_in_first_latent = w2uflags.flag(2**3, "In first latent")
    flag_unwise_w2_in_second_latent = w2uflags.flag(2**4, "In second latent")
    flag_unwise_w2_in_circular_halo = w2uflags.flag(2**5, "In circular halo")
    flag_unwise_w2_saturated = w2uflags.flag(2**6, "Saturated")
    flag_unwise_w2_in_geometric_diffraction_spike = w2uflags.flag(2**7, "In geometric diffraction spike")
        
    flag_unwise_w1_in_bright_star_psf = w1aflags.flag(2**0, "In PSF of bright star falling off coadd")
    flag_unwise_w1_in_hyperleda_galaxy = w1aflags.flag(2**1, "In HyperLeda large galaxy")
    flag_unwise_w1_in_big_object = w1aflags.flag(2**2, "In \"big object\" (e.g., a Magellanic cloud)")
    flag_unwise_w1_pixel_in_very_bright_star_centroid = w1aflags.flag(2**3, "Pixel may contain the centroid of a very bright star")
    flag_unwise_w1_crowdsource_saturation = w1aflags.flag(2**4, "crowdsource considers this pixel potentially affected by saturation")
    flag_unwise_w1_possible_nebulosity = w1aflags.flag(2**5, "Pixel may contain nebulosity")
    flag_unwise_w1_no_aggressive_deblend = w1aflags.flag(2**6, "Sources in this pixel will not be aggressively deblended")
    flag_unwise_w1_candidate_sources_must_be_sharp = w1aflags.flag(2**7, "Candidate sources in this pixel must be \"sharp\" to be optimized")

    flag_unwise_w2_in_bright_star_psf = w2aflags.flag(2**0, "In PSF of bright star falling off coadd")
    flag_unwise_w2_in_hyperleda_galaxy = w2aflags.flag(2**1, "In HyperLeda large galaxy")
    flag_unwise_w2_in_big_object = w2aflags.flag(2**2, "In \"big object\" (e.g., a Magellanic cloud)")
    flag_unwise_w2_pixel_in_very_bright_star_centroid = w2aflags.flag(2**3, "Pixel may contain the centroid of a very bright star")
    flag_unwise_w2_crowdsource_saturation = w2aflags.flag(2**4, "crowdsource considers this pixel potentially affected by saturation")
    flag_unwise_w2_possible_nebulosity = w2aflags.flag(2**5, "Pixel may contain nebulosity")
    flag_unwise_w2_no_aggressive_deblend = w2aflags.flag(2**6, "Sources in this pixel will not be aggressively deblended")
    flag_unwise_w2_candidate_sources_must_be_sharp = w2aflags.flag(2**7, "Candidate sources in this pixel must be \"sharp\" to be optimized")

    #> GLIMPSE Photometry
    mag4_5 = FloatField(null=True, help_text="IRAC band 4.5 micron magnitude [mag]")
    d4_5m = FloatField(null=True, help_text="Error on IRAC band 4.5 micron magnitude [mag]")
    rms_f4_5 = FloatField(null=True, help_text="RMS deviations from final flux [mJy]")
    sqf_4_5 = BitField(default=0, help_text="Source quality flag for IRAC band 4.5 micron")
    mf4_5 = BitField(default=0, help_text="Flux calculation method flag")
    csf = BitField(default=0, help_text="Close source flag")
    #< See https://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/gator_docs/

    flag_glimpse_poor_dark_pixel_current = sqf_4_5.flag(2**0, "Poor pixels in dark current")
    flag_glimpse_flat_field_questionable = sqf_4_5.flag(2**1, "Flat field applied using questionable value")
    flag_glimpse_latent_image = sqf_4_5.flag(2**2, "Latent image")
    flag_glimpse_saturated_star_correction = sqf_4_5.flag(2**3, "Sat star correction")
    flag_glimpse_muxbleed_correction_applied = sqf_4_5.flag(2**6, "Muxbleed correction applied")
    flag_glimpse_hot_or_dead_pixels = sqf_4_5.flag(2**7, "Hot, dead or otherwise unacceptable pixel")
    flag_glimpse_muxbleed_significant = sqf_4_5.flag(2**8, "Muxbleed > 3-sigma above the background")
    flag_glimpse_allstar_tweak_positive = sqf_4_5.flag(2**9, "Allstar tweak positive")
    flag_glimpse_allstar_tweak_negative = sqf_4_5.flag(2**10, "Allstar tweak negative")
    flag_glimpse_confusion_in_band_merge = sqf_4_5.flag(2**12, "Confusion in in-band merge")
    flag_glimpse_confusion_in_cross_band_merge = sqf_4_5.flag(2**13, "Confusion in cross-band merge")
    flag_glimpse_column_pulldown_correction = sqf_4_5.flag(2**14, "Column pulldown correction")
    flag_glimpse_banding_correction = sqf_4_5.flag(2**15, "Banding correction")
    flag_glimpse_stray_light = sqf_4_5.flag(2**16, "Stray light")
    flag_glimpse_no_nonlinear_correction = sqf_4_5.flag(2**18, "Nonlinear correction not applied")
    flag_glimpse_saturated_star_wing_region = sqf_4_5.flag(2**19, "Saturated star wing region")
    flag_glimpse_pre_lumping_in_band_merge = sqf_4_5.flag(2**20, "Pre-lumping in in-band merge")
    flag_glimpse_post_lumping_in_cross_band_merge = sqf_4_5.flag(2**21, "Post-lumping in cross-band merge")
    flag_glimpse_edge_of_frame = sqf_4_5.flag(2**29, "Edge of frame (within 3 pixels of edge)")
    flag_glimpse_truth_list = sqf_4_5.flag(2**30, "Truth list (for simulated data)")

    flag_glimpse_no_source_within_3_arcsecond = csf.flag(2**0, "No sources in GLIMPSE within 3\" of the source")
    flag_glimpse_1_source_within_2p5_and_3_arcsecond = csf.flag(2**1, "1 sources in GLIMPSE between 2.5\" and 3\" of the source")
    flag_glimpse_2_sources_within_2_and_2p5_arcsecond = csf.flag(2**2, "2 sources in GLIMPSE within 2\" and 2.5\" of the source")
    flag_glimpse_3_sources_within_1p5_and_2_arcsecond = csf.flag(2**3, "3 sources in GLIMPSE within 1.5\" and 2\" of the source")
    flag_glimpse_4_sources_within_1_and_1p5_arcsecond = csf.flag(2**4, "4 sources in GLIMPSE within 1\" and 1.5\" of the source")
    flag_glimpse_5_sources_within_0p5_and_1_arcsecond = csf.flag(2**5, "5 sources in GLIMPSE within 0.5\" and 1.0\" of the source")
    flag_glimpse_6_sources_within_0p5_arcsecond = csf.flag(2**6, "6 sources in GLIMPSE within 0.5\" of this source")

    #> Gaia XP Stellar Parameters (Zhang, Green & Rix 2023)
    zgr_teff = FloatField(null=True, help_text=Glossary.teff)
    zgr_e_teff = FloatField(null=True, help_text=Glossary.e_teff)
    zgr_logg = FloatField(null=True, help_text=Glossary.logg)
    zgr_e_logg = FloatField(null=True, help_text=Glossary.e_logg)
    zgr_fe_h = FloatField(null=True, help_text=Glossary.fe_h)
    zgr_e_fe_h = FloatField(null=True, help_text=Glossary.e_fe_h)
    zgr_e = FloatField(null=True, help_text="Extinction [mag]")
    zgr_e_e = FloatField(null=True, help_text="Error on extinction [mag]")
    zgr_plx = FloatField(null=True, help_text=Glossary.plx)
    zgr_e_plx = FloatField(null=True, help_text=Glossary.e_plx)
    zgr_teff_confidence = FloatField(null=True, help_text="Confidence estimate in TEFF")
    zgr_logg_confidence = FloatField(null=True, help_text="Confidence estimate in LOGG")
    zgr_fe_h_confidence = FloatField(null=True, help_text="Confidence estimate in FE_H")
    zgr_ln_prior = FloatField(null=True, help_text="Log prior probability")
    zgr_chi2 = FloatField(null=True, help_text=Glossary.chi2)
    zgr_quality_flags = BitField(default=0, help_text="Quality flags")
    # See https://zenodo.org/record/7811871

    #> Bailer-Jones Distance Estimates (EDR3; 2021)
    r_med_geo = FloatField(null=True, help_text="Median geometric distance [pc]")
    r_lo_geo = FloatField(null=True, help_text="16th percentile of geometric distance [pc]")
    r_hi_geo = FloatField(null=True, help_text="84th percentile of geometric distance [pc]")
    r_med_photogeo = FloatField(null=True, help_text="50th percentile of photogeometric distance [pc]")
    r_lo_photogeo = FloatField(null=True, help_text="16th percentile of photogeometric distance [pc]")
    r_hi_photogeo = FloatField(null=True, help_text="84th percentile of photogeometric distance [pc]")
    bailer_jones_flags = TextField(null=True, help_text="Bailer-Jones quality flags") # TODO: omg change this to a bitfield and give flag definitions
    # See https://dc.zah.uni-heidelberg.de/tableinfo/gedr3dist.main#note-f

    #> Reddening
    ebv = FloatField(null=True, help_text="E(B-V) [mag]")
    e_ebv = FloatField(null=True, help_text="Error on E(B-V) [mag]")
    flag_ebv_upper_limit = BooleanField(default=False, help_text="E(B-V) is an upper limit")
    ebv_method_flags = BitField(default=0, help_text="Flags indicating the source of E(B-V)")

    flag_ebv_from_zhang_2023 = ebv_method_flags.flag(2**0, "E(B-V) from Zhang et al. (2023)")
    flag_ebv_from_edenhofer_2023 = ebv_method_flags.flag(2**1, "E(B-V) from Edenhofer et al. (2023)")
    flag_ebv_from_sfd = ebv_method_flags.flag(2**2, "E(B-V) from SFD")
    flag_ebv_from_rjce_glimpse = ebv_method_flags.flag(2**3, "E(B-V) from RJCE GLIMPSE")
    flag_ebv_from_rjce_allwise = ebv_method_flags.flag(2**4, "E(B-V) from RJCE AllWISE")
    flag_ebv_from_bayestar_2019 = ebv_method_flags.flag(2**5, "E(B-V) from Bayestar 2019")

    ebv_zhang_2023 = FloatField(null=True, help_text="E(B-V) from Zhang et al. (2023) [mag]")
    e_ebv_zhang_2023 = FloatField(null=True, help_text="Error on E(B-V) from Zhang et al. (2023) [mag]")
    ebv_sfd = FloatField(null=True, help_text="E(B-V) from SFD [mag]")
    # In these help_texts they vary a little in format from convention (e.g., "Error on X E(B-V)" instead of "Error on E(B-V) from X")
    # but that's because they are too long to fit in the FITS header. We'll see if anyone ever notices.
    e_ebv_sfd = FloatField(null=True, help_text="Error on E(B-V) from SFD [mag]")    
    ebv_rjce_glimpse = FloatField(null=True, help_text="E(B-V) from RJCE GLIMPSE [mag]")
    e_ebv_rjce_glimpse = FloatField(null=True, help_text="Error on RJCE GLIMPSE E(B-V) [mag]")
    ebv_rjce_allwise = FloatField(null=True, help_text="E(B-V) from RJCE AllWISE [mag]")
    e_ebv_rjce_allwise = FloatField(null=True, help_text="Error on RJCE AllWISE E(B-V)[mag]")
    ebv_bayestar_2019 = FloatField(null=True, help_text="E(B-V) from Bayestar 2019 [mag]")
    e_ebv_bayestar_2019 = FloatField(null=True, help_text="Error on Bayestar 2019 E(B-V) [mag]")
    ebv_edenhofer_2023 = FloatField(null=True, help_text="E(B-V) from Edenhofer et al. (2023) [mag]")
    e_ebv_edenhofer_2023 = FloatField(null=True, help_text="Error on Edenhofer et al. (2023) E(B-V) [mag]")
    flag_ebv_edenhofer_2023_upper_limit = BooleanField(default=False, help_text="Upper limit on Edenhofer E(B-V)")
    
    #> Synthetic Photometry from Gaia XP Spectra
    c_star = FloatField(null=True, help_text="Quality parameter (see Riello et al. 2021)")
    u_jkc_mag = FloatField(null=True, help_text="Gaia XP synthetic U-band (JKC) [mag]")
    u_jkc_mag_flag = IntegerField(null=True, help_text="U-band (JKC) is within valid range")
    b_jkc_mag = FloatField(null=True, help_text="Gaia XP synthetic B-band (JKC) [mag]")
    b_jkc_mag_flag = IntegerField(null=True, help_text="B-band (JKC) is within valid range")                    
    v_jkc_mag = FloatField(null=True, help_text="Gaia XP synthetic V-band (JKC) [mag]")
    v_jkc_mag_flag = IntegerField(null=True, help_text="V-band (JKC) is within valid range")
    r_jkc_mag = FloatField(null=True, help_text="Gaia XP synthetic R-band (JKC) [mag]")
    r_jkc_mag_flag = IntegerField(null=True, help_text="R-band (JKC) is within valid range")
    i_jkc_mag = FloatField(null=True, help_text="Gaia XP synthetic I-band (JKC) [mag]")
    i_jkc_mag_flag = IntegerField(null=True, help_text="I-band (JKC) is within valid range")                                                        
    u_sdss_mag = FloatField(null=True, help_text="Gaia XP synthetic u-band (SDSS) [mag]")
    u_sdss_mag_flag = IntegerField(null=True, help_text="u-band (SDSS) is within valid range")
    g_sdss_mag = FloatField(null=True, help_text="Gaia XP synthetic g-band (SDSS) [mag]")
    g_sdss_mag_flag = IntegerField(null=True, help_text="g-band (SDSS) is within valid range")
    r_sdss_mag = FloatField(null=True, help_text="Gaia XP synthetic r-band (SDSS) [mag]")
    r_sdss_mag_flag = IntegerField(null=True, help_text="r-band (SDSS) is within valid range")
    i_sdss_mag = FloatField(null=True, help_text="Gaia XP synthetic i-band (SDSS) [mag]")
    i_sdss_mag_flag = IntegerField(null=True, help_text="i-band (SDSS) is within valid range")
    z_sdss_mag = FloatField(null=True, help_text="Gaia XP synthetic z-band (SDSS) [mag]")
    z_sdss_mag_flag = IntegerField(null=True, help_text="z-band (SDSS) is within valid range")
    y_ps1_mag = FloatField(null=True, help_text="Gaia XP synthetic Y-band (PS1) [mag]")
    y_ps1_mag_flag = IntegerField(null=True, help_text="Y-band (PS1) is within valid range")
    
    #> Observations Summary
    n_boss_visits = IntegerField(null=True, help_text="Number of BOSS visits")
    boss_min_mjd = IntegerField(null=True, help_text="Minimum MJD of BOSS visits")
    boss_max_mjd = IntegerField(null=True, help_text="Maximum MJD of BOSS visits")
    n_apogee_visits = IntegerField(null=True, help_text="Number of APOGEE visits")    
    apogee_min_mjd = IntegerField(null=True, help_text="Minimum MJD of APOGEE visits")
    apogee_max_mjd = IntegerField(null=True, help_text="Maximum MJD of APOGEE visits")

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
    t = Table.read(expand_path("$MWM_ASTRA/aux/targeting-bits/sdss5_target_1_with_groups.csv"))
    t.sort("bit")
    return t

