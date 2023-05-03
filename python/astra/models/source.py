
from peewee import (
    AutoField,
    IntegerField,
    FloatField,
    TextField,
    BigBitField,
    BigIntegerField,
    PostgresqlDatabase,
    SmallIntegerField,
    DateTimeField,
    fn,
)
from playhouse.hybrid import hybrid_method
from astra.models.base import database, BaseModel
from astra.models.fields import BitField
from astra.models.spectrum import Spectrum


class Source(BaseModel):

    """ An astronomical source. """

    #> Identifiers
    sdss_id = AutoField()
    healpix = IntegerField(null=True)
    gaia_dr3_source_id = BigIntegerField(null=True)
    tic_v8_id = BigIntegerField(null=True)
    sdss4_dr17_apogee_id = TextField(null=True)
    sdss4_dr17_field = TextField(null=True)

    #> Astrometry
    ra = FloatField()
    dec = FloatField()
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
    # 
    # For unWISE they report fluxes, so we keep their naming convention where it fits within 
    # 8 characters, and document when the name differs from the original catalog.

    #> Photometry
    g_mag = FloatField(null=True)
    bp_mag = FloatField(null=True)
    rp_mag = FloatField(null=True)

    # 2MASS
    j_mag = FloatField(null=True)
    e_j_mag = FloatField(null=True)
    h_mag = FloatField(null=True)
    e_h_mag = FloatField(null=True)
    k_mag = FloatField(null=True)
    e_k_mag = FloatField(null=True)
    ph_qual = TextField(null=True)
    bl_flg = TextField(null=True)
    cc_flg = TextField(null=True)

    # unWISE
    w1_flux = FloatField(null=True)
    w1_dflux = FloatField(null=True)
    w2_flux = FloatField(null=True)
    w2_dflux =  FloatField(null=True)
    w1_frac = FloatField(null=True)
    w2_frac = FloatField(null=True)
    w1_uflags = BitField(default=0)
    w2_uflags = BitField(default=0)
    w1_aflags = BitField(default=0)
    w2_aflags = BitField(default=0)
    
    flag_unwise_w1_in_core_or_wings = w1_uflags.flag(2**0, "In core or wings")
    flag_unwise_w1_in_diffraction_spike = w1_uflags.flag(2**1, "In diffraction spike")
    flag_unwise_w1_in_ghost = w1_uflags.flag(2**2, "In ghost")
    flag_unwise_w1_in_first_latent = w1_uflags.flag(2**3, "In first latent")
    flag_unwise_w1_in_second_latent = w1_uflags.flag(2**4, "In second latent")
    flag_unwise_w1_in_circular_halo = w1_uflags.flag(2**5, "In circular halo")
    flag_unwise_w1_saturated = w1_uflags.flag(2**6, "Saturated")
    flag_unwise_w1_in_geometric_diffraction_spike = w1_uflags.flag(2**7, "In geometric diffraction spike")
    
    flag_unwise_w2_in_core_or_wings = w2_uflags.flag(2**0, "In core or wings")
    flag_unwise_w2_in_diffraction_spike = w2_uflags.flag(2**1, "In diffraction spike")
    flag_unwise_w2_in_ghost = w2_uflags.flag(2**2, "In ghost")
    flag_unwise_w2_in_first_latent = w2_uflags.flag(2**3, "In first latent")
    flag_unwise_w2_in_second_latent = w2_uflags.flag(2**4, "In second latent")
    flag_unwise_w2_in_circular_halo = w2_uflags.flag(2**5, "In circular halo")
    flag_unwise_w2_saturated = w2_uflags.flag(2**6, "Saturated")
    flag_unwise_w2_in_geometric_diffraction_spike = w2_uflags.flag(2**7, "In geometric diffraction spike")
        
    flag_unwise_w1_in_bright_star_psf = w1_aflags.flag(2**0, "In PSF of bright star falling off coadd")
    flag_unwise_w1_in_hyperleda_galaxy = w1_aflags.flag(2**1, "In HyperLeda large galaxy")
    flag_unwise_w1_in_big_object = w1_aflags.flag(2**2, "In \"big object\" (e.g., a Magellanic cloud)")
    flag_unwise_w1_pixel_in_very_bright_star_centroid = w1_aflags.flag(2**3, "Pixel may contain the centroid of a very bright star")
    flag_unwise_w1_crowdsource_saturation = w1_aflags.flag(2**4, "crowdsource considers this pixel potentially affected by saturation")
    flag_unwise_w1_possible_nebulosity = w1_aflags.flag(2**5, "Pixel may contain nebulosity")
    flag_unwise_w1_no_aggressive_deblend = w1_aflags.flag(2**6, "Sources in this pixel will not be aggressively deblended")
    flag_unwise_w1_candidate_sources_must_be_sharp = w1_aflags.flag(2**7, "Candidate sources in this pixel must be \"sharp\" to be optimized")

    flag_unwise_w2_in_bright_star_psf = w2_aflags.flag(2**0, "In PSF of bright star falling off coadd")
    flag_unwise_w2_in_hyperleda_galaxy = w2_aflags.flag(2**1, "In HyperLeda large galaxy")
    flag_unwise_w2_in_big_object = w2_aflags.flag(2**2, "In \"big object\" (e.g., a Magellanic cloud)")
    flag_unwise_w2_pixel_in_very_bright_star_centroid = w2_aflags.flag(2**3, "Pixel may contain the centroid of a very bright star")
    flag_unwise_w2_crowdsource_saturation = w2_aflags.flag(2**4, "crowdsource considers this pixel potentially affected by saturation")
    flag_unwise_w2_possible_nebulosity = w2_aflags.flag(2**5, "Pixel may contain nebulosity")
    flag_unwise_w2_no_aggressive_deblend = w2_aflags.flag(2**6, "Sources in this pixel will not be aggressively deblended")
    flag_unwise_w2_candidate_sources_must_be_sharp = w2_aflags.flag(2**7, "Candidate sources in this pixel must be \"sharp\" to be optimized")

    # GLIMPSE
    mag4_5 = FloatField(null=True)
    d4_5m = FloatField(null=True)
    rms_f4_5 = FloatField(null=True)
    sqf_4_5 = BitField(default=0)
    mf_4_5 = BitField(default=0) # TODO: unclear what these definitions are
    csf = IntegerField(default=0) # no idea what this is

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

    #> Targeting
    carton_0 = TextField(default="")

    # Only do carton_flags if we have a postgresql database.
    if isinstance(database, PostgresqlDatabase):
        carton_flags = BigBitField(null=True)

    sdss4_apogee_target1_flags = BitField(default=0, help_text="SDSS4 APOGEE1 targeting flags (1/2)")
    sdss4_apogee_target2_flags = BitField(default=0, help_text="SDSS4 APOGEE1 targeting flags (2/2)")
    sdss4_apogee2_target1_flags = BitField(default=0, help_text="SDSS4 APOGEE2 targeting flags (1/3)")
    sdss4_apogee2_target2_flags = BitField(default=0, help_text="SDSS4 APOGEE2 targeting flags (2/3)")
    sdss4_apogee2_target3_flags = BitField(default=0, help_text="SDSS4 APOGEE2 targeting flags (3/3)")
    sdss4_apogee_member_flags = BitField(default=0, help_text="SDSS4 likely cluster/galaxy member flags")
    sdss4_apogee_extra_target_flags = BitField(default=0, help_text="SDSS4 target info (aka EXTRATARG)")

    # Define flags for all bit fields
    # TODO: Should we only bind these flags when asked? Do a speed time comparison of Source() and `import Source` with and without them.
    
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
    flag_sdss4_apogee_mir_cluster_star = sdss4_apogee_target2_flags.flag(2**16, help_text="Outer Disk MIR Clusters (Beaton)")
    flag_sdss4_apogee_rv_monitor_ic348 = sdss4_apogee_target2_flags.flag(2**17, help_text="RV Variability in IC348 (Nidever, Covey)")
    flag_sdss4_apogee_rv_monitor_kepler = sdss4_apogee_target2_flags.flag(2**18, help_text="RV Variability for Kepler Planet Hosts and Binaries (Deshpande, Fleming, Mahadevan)")
    flag_sdss4_apogee_ges_calibrate = sdss4_apogee_target2_flags.flag(2**19, help_text="Gaia-ESO calibration targets")
    flag_sdss4_apogee_bulge_rv_verify = sdss4_apogee_target2_flags.flag(2**20, help_text="RV Verification (Nidever)")
    flag_sdss4_apogee_1m_target = sdss4_apogee_target2_flags.flag(2**21, help_text="Selected as a 1-m target (Holtzman)")

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


    @property
    def cartons(self):
        """ Return the cartons that this source is assigned. """
        return (
            Carton
            .select()
            .where(Carton.pk << self.carton_primary_keys)
        )
    

    @property
    def carton_primary_keys(self):
        """ Return the primary keys of the cartons that this source is assigned. """
        i, carton_pks, cur_size = (0, [], len(self.carton_flags._buffer))
        while True:
            byte_num, byte_offset = divmod(i, 8)
            if byte_num >= cur_size:
                break
            if bool(self.carton_flags._buffer[byte_num] & (1 << byte_offset)):
                carton_pks.append(i)
            i += 1
        return tuple(carton_pks)


    @hybrid_method
    def in_carton(self, carton):
        """
        An expression to evaluate whether this source is assigned to the given carton.
        
        :param carton:
            A `Carton` or carton primary key.
        """
        carton_pk = carton.pk if isinstance(carton, Carton) else carton
        return (
            (fn.length(self.carton_flags) > int(carton_pk / 8))
        &   (fn.get_bit(self.carton_flags, carton_pk) > 0)
        )
    

    @hybrid_method
    def in_any_carton(self, *cartons):
        """An expression to evaluate whether this source is assigned to any of the given cartons."""
        return fn.OR(*[self.in_carton(carton) for carton in cartons])


    @property
    def spectra(self):
        """A generator that yields all spectra associated with this source."""
        for expr, column in self.dependencies():
            if Spectrum in column.model.__mro__[1:]:
                yield from column.model.select().where(expr)




class Carton(BaseModel):

    """A carton to which sources are assigned."""
    
    pk = AutoField()
    mapper_pk = SmallIntegerField()
    category_pk = SmallIntegerField()
    version_pk = SmallIntegerField()

    carton = TextField()
    program = TextField()
    run_on = DateTimeField(null=True)
