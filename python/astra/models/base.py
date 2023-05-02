
__all__ = ["Source", "UniqueSpectrum", "Carton", "SpectrumMixin", "BaseModel", "database"]

from peewee import (
    fn,
    AutoField,
    FloatField,
    Field,
    DateTimeField,
    BigIntegerField,
    SmallIntegerField,
    IntegerField,
    TextField,
    Model,
    BigBitField,
    PostgresqlDatabase
)
import re
from inspect import getsource
from playhouse.hybrid import hybrid_method
from playhouse.sqlite_ext import SqliteExtDatabase

from astra import config
from astra.utils import log, get_config_paths, expand_path
from astra.models.fields import BitField
from astra.models.glossary import Glossary

# Note that we can't use a DatabaseProxy and define it later because we also need to be
# able to dynamically set the schema, which seems to be impossible with a DatabaseProxy.

def get_database_and_schema(config):
    """
    Return a database and schema given some configuration.
    
    :param config:
        A dictionary of configuration values, read from a config file.
    """
        
    sqlite_kwargs = dict(
        thread_safe=True,
        pragmas={
            'journal_mode': 'wal',
            'cache_size': -1 * 64000,  # 64MB
            'foreign_keys': 1,
            'ignore_check_constraints': 0,
            'synchronous': 0
        }
    )

    config_placement_message = (
        "These are the places where Astra looks for a config file:\n"
    )
    for path in get_config_paths():
        config_placement_message += f"  - {path}\n"    
    config_placement_message += (
        "\n\n"
        "This is what the `database` entry could look like in the config file:\n"
        "  # For PostgreSQL (preferred)\n"
        "  database:\n"
        "    dbname: <DATABASE_NAME>\n"
        "    user: [USERNAME]       # can be optional\n"
        "    host: [HOSTNAME]       # can be optional\n"
        "    password: [PASSWORD]   # can be optional\n"
        "    port: [PORT]           # can be optional\n"
        "    schema: [SCHEMA]       # can be optional\n"            
        "\n"
        "  # For SQLite\n"
        "  database:\n"
        "    path: <PATH_TO_DATABASE>\n\n"
    )

    if config.get("DEBUG", False):
        log.warning("In DEBUG mode")
        database_path_key = "debug_mode_database_path"
        database_path = config.get(database_path_key, ":memory:")
        log.info(f"Setting database path to {database_path}.")
        log.info(f"You can change this using the `debug_mode_database_path` key in the Astra config file.")
        log.info(f"These are the locations where Astra looks for a config file:\n")
        for path in get_config_paths():
            log.info(f"  - {path}")

        database = SqliteExtDatabase(database_path, **sqlite_kwargs)
        return (database, None)
        
    elif config.get("TESTING", False):
        log.warning("In TESTING mode, using in-memory SQLite database")
        database = SqliteExtDatabase(":memory:", **sqlite_kwargs)
        return (database, None)

    else:
        if "database" in config and isinstance(config["database"], dict):
            # Prefer postgresql
            if "dbname" in config["database"]:
                try:
                    keys = ("user", "host", "password", "port")
                    kwds = dict([(k, config["database"][k]) for k in keys if k in config["database"]])
                    database = PostgresqlDatabase(config["database"]["dbname"], **kwds)
                    schema = config["database"].get("schema", None)

                except:
                    log.exception(f"Could not create PostgresqlDatabase from config.\n{config_placement_message}")
                    raise
                else:
                    return (database, schema)

            elif "path" in config["database"]:
                try:
                    database = SqliteExtDatabase(expand_path(config["database"]["path"]), **sqlite_kwargs)
                except:
                    log.exception(f"Could not create SqliteExtDatabase from config.\n{config_placement_message}")
                    raise 
                else:
                    return (database, None)

        
        log.warning(f"No valid `database` entry found in Astra config file.\n{config_placement_message}")
        log.info(f"Defaulting to in-memory SQLite database.")
        database = SqliteExtDatabase(":memory:", **sqlite_kwargs)
    
        return (database, None)


database, schema = get_database_and_schema(config)

class BaseModel(Model):
    
    class Meta:
        database = database
        schema = schema
        legacy_table_names = False

    @classmethod
    @property
    def field_category_headers(cls):
        """
        Return a tuple of category headers for the data model fields based on the source code.
        Category headers are defined in the source code like this:

        ```python
        #> Category header
        teff = FloatField(...)

        #> New category header
        """

        pattern = '\s{4}#>\s*(.+)\n\s{4}([\w|\d|_]+)\s*='
        source_code = getsource(cls)
        category_headers = []
        for header, field_name in re.findall(pattern, source_code):
            if hasattr(cls, field_name) and isinstance(getattr(cls, field_name), Field):
                category_headers.append((header, field_name))
            else:
                log.warning(
                    f"Found category header '{header}', starting above '{field_name}' in {cls}, "
                    f"but {cls}.{field_name} is not an attribute of type `peewee.Field`."
                )
        return tuple(category_headers)


class Source(BaseModel):

    """ An astronomical source. """

    #> Identifiers
    sdss_id = AutoField(help_text=Glossary.sdss_id)
    healpix = IntegerField(help_text=Glossary.healpix, null=True)
    gaia_dr3_source_id = BigIntegerField(help_text=Glossary.gaia_dr3_source_id, null=True)
    tic_v8_id = BigIntegerField(help_text=Glossary.tic_v8_id, null=True)
    sdss4_dr17_apogee_id = TextField(help_text=Glossary.sdss4_dr17_apogee_id, null=True)
    sdss4_dr17_field = TextField(help_text=Glossary.sdss4_dr17_field, null=True)

    #> Astrometry
    ra = FloatField(help_text=Glossary.ra)
    dec = FloatField(help_text=Glossary.dec)
    plx = FloatField(help_text=Glossary.plx, null=True, verbose_name="parallax")
    e_plx = FloatField(help_text=Glossary.e_plx, null=True, verbose_name="e_parallax")
    pmra = FloatField(help_text=Glossary.pmra, null=True)
    e_pmra = FloatField(help_text=Glossary.e_pmra, null=True)
    pmde = FloatField(help_text=Glossary.pmde, null=True)
    e_pmde = FloatField(help_text=Glossary.e_pmde, null=True)
    gaia_v_rad = FloatField(help_text=Glossary.gaia_v_rad, null=True)
    gaia_e_v_rad = FloatField(help_text=Glossary.gaia_e_v_rad, null=True)

    #> Photometry
    g_mag = FloatField(help_text=Glossary.g_mag, null=True)
    bp_mag = FloatField(help_text=Glossary.bp_mag, null=True)
    rp_mag = FloatField(help_text=Glossary.rp_mag, null=True)
    j_mag = FloatField(help_text=Glossary.j_mag, null=True)
    e_j_mag = FloatField(help_text=Glossary.e_j_mag, null=True)
    h_mag = FloatField(help_text=Glossary.h_mag, null=True)
    e_h_mag = FloatField(help_text=Glossary.e_h_mag, null=True)
    k_mag = FloatField(help_text=Glossary.k_mag, null=True)
    e_k_mag = FloatField(help_text=Glossary.e_k_mag, null=True)

    # TODO: Add requisite WISE photometry columns from Sayjederi

    #> Targeting
    carton_0 = TextField(help_text=Glossary.carton_0, default="")

    # Only do carton_flags if we have a postgresql database.
    if isinstance(database, PostgresqlDatabase):
        carton_flags = BigBitField(help_text=Glossary.carton_flags, null=True)

    sdss4_apogee_target1_flags = BitField(default=0, help_text=Glossary.sdss4_apogee_target1_flags)
    sdss4_apogee_target2_flags = BitField(default=0, help_text=Glossary.sdss4_apogee_target2_flags)
    sdss4_apogee2_target1_flags = BitField(default=0, help_text=Glossary.sdss4_apogee2_target1_flags)
    sdss4_apogee2_target2_flags = BitField(default=0, help_text=Glossary.sdss4_apogee2_target2_flags)
    sdss4_apogee2_target3_flags = BitField(default=0, help_text=Glossary.sdss4_apogee2_target3_flags)
    sdss4_apogee_member_flags = BitField(default=0, help_text=Glossary.sdss4_apogee_member_flags)
    sdss4_apogee_extra_target_flags = BitField(default=0, help_text=Glossary.sdss4_apogee_extra_target_flags)

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
            if SpectrumMixin in column.model.__mro__[1:]:
                yield from column.model.select().where(expr)



    


class UniqueSpectrum(BaseModel):

    """ A one dimensional spectrum. """

    spectrum_id = AutoField(help_text=Glossary.spectrum_id)
    

class SpectrumMixin:

    def plot(self, rectified=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = (self.wavelength, self.flux)
        c = self.continuum if rectified else 1
        ax.plot(x, y / c, c='k')

        #ax.plot(x, self.model_flux)
        return fig


class Carton(BaseModel):

    """A carton to which stars are assigned."""
    
    pk = AutoField(help_text=Glossary.carton_id)
    mapper_pk = SmallIntegerField()
    category_pk = SmallIntegerField()
    version_pk = SmallIntegerField()

    carton = TextField(help_text=Glossary.carton)
    program = TextField()
    run_on = DateTimeField(null=True)



if __name__ == "__main__":
    # TODO: move these to tests
    import numpy as np
    from tqdm import tqdm
    
    database.drop_tables([
        Source,
        Carton
    ])

    database.create_tables([
        Source,
        Carton,
    ])

    carton_names = (
        "bhm_aqmes_bonus_bright",
        "bhm_aqmes_bonus-bright",
        "bhm_aqmes_bonus_core",
        "bhm_aqmes_bonus-dark",
        "bhm_aqmes_bonus_faint",
        "bhm_aqmes_med",
        "bhm_aqmes_med_faint",
        "bhm_aqmes_med-faint",
        "bhm_aqmes_wide2",
        "bhm_aqmes_wide2_faint",
        "bhm_aqmes_wide2-faint",
        "bhm_aqmes_wide3",
        "bhm_aqmes_wide3-faint",
        "bhm_colr_galaxies_lsdr10",
        "bhm_colr_galaxies_lsdr8",
        "bhm_csc_apogee",
        "bhm_csc_boss",
        "bhm_csc_boss_bright",
        "bhm_csc_boss-bright",
        "bhm_csc_boss_dark",
        "bhm_csc_boss-dark",
        "bhm_gua_bright",
        "bhm_gua_dark",
        "bhm_rm_ancillary",
        "bhm_rm_core",
        "bhm_rm_known_spec",
        "bhm_rm_known-spec",
        "bhm_rm_var",
        "bhm_rm_xrayqso",
        "bhm_spiders_agn-efeds",
        "bhm_spiders_agn_efeds_stragglers",
        "bhm_spiders_agn_gaiadr2",
        "bhm_spiders_agn_gaiadr3",
        "bhm_spiders_agn_hard",
        "bhm_spiders_agn_lsdr10",
        "bhm_spiders_agn_lsdr8",
        "bhm_spiders_agn_ps1dr2",
        "bhm_spiders_agn_sep",
        "bhm_spiders_agn_skymapperdr2",
        "bhm_spiders_agn_supercosmos",
        "bhm_spiders_agn_tda",
        "bhm_spiders_clusters-efeds-erosita",
        "bhm_spiders_clusters-efeds-hsc-redmapper",
        "bhm_spiders_clusters-efeds-ls-redmapper",
        "bhm_spiders_clusters-efeds-sdss-redmapper",
        "bhm_spiders_clusters_efeds_stragglers",
        "bhm_spiders_clusters_lsdr10",
        "bhm_spiders_clusters_lsdr8",
        "bhm_spiders_clusters_ps1dr2",
        "comm_pleiades",
        "comm_spectrophoto",
        "manual_bhm_spiders_comm",
        "manual_bhm_spiders_comm_lco",
        "manual_bright_target",
        "manual_bright_target_offsets_1",
        "manual_bright_target_offsets_1_g13",
        "manual_bright_target_offsets_2",
        "manual_bright_target_offsets_2_g13",
        "manual_bright_target_offsets_3",
        "manual_bright_targets",
        "manual_bright_targets_g13",
        "manual_bright_targets_g13_offset_fixed_1",
        "manual_bright_targets_g13_offset_fixed_2",
        "manual_bright_targets_g13_offset_fixed_3",
        "manual_bright_targets_g13_offset_fixed_4",
        "manual_bright_targets_g13_offset_fixed_5",
        "manual_bright_targets_g13_offset_fixed_6",
        "manual_bright_targets_g13_offset_fixed_7",
        "manual_fps_position_stars",
        "manual_fps_position_stars_10",
        "manual_fps_position_stars_apogee_10",
        "manual_fps_position_stars_lco_apogee_10",
        "manual_mwm_crosscalib_apogee",
        "manual_mwm_crosscalib_yso_apogee",
        "manual_mwm_crosscalib_yso_boss",
        "manual_mwm_halo_distant_bhb",
        "manual_mwm_halo_distant_kgiant",
        "manual_mwm_halo_mp_bbb",
        "manual_mwm_magcloud_massive_apogee",
        "manual_mwm_magcloud_massive_boss",
        "manual_mwm_magcloud_symbiotic_apogee",
        "manual_mwm_planet_ca_legacy_v1",
        "manual_mwm_planet_gaia_astrometry_v1",
        "manual_mwm_planet_gpi_v1",
        "manual_mwm_planet_harps_v1",
        "manual_mwm_planet_known_v1",
        "manual_mwm_planet_sophie_v1",
        "manual_mwm_planet_sphere_v1",
        "manual_mwm_planet_tess_eb_v1",
        "manual_mwm_planet_tess_pc_v1",
        "manual_mwm_planet_transiting_bd_v1",
        "manual_mwm_tess_ob",
        "manual_mwm_validation_cool_apogee",
        "manual_mwm_validation_cool_boss",
        "manual_mwm_validation_hot_apogee",
        "manual_mwm_validation_hot_boss",
        "manual_mwm_validation_rv",
        "manual_nsbh_apogee",
        "manual_nsbh_boss",
        "manual_offset_mwmhalo_off00",
        "manual_offset_mwmhalo_off05",
        "manual_offset_mwmhalo_off10",
        "manual_offset_mwmhalo_off20",
        "manual_offset_mwmhalo_off30",
        "manual_offset_mwmhalo_offa",
        "manual_offset_mwmhalo_offb",
        "manual_planet_ca_legacy_v0",
        "manual_planet_gaia_astrometry_v0",
        "manual_planet_gpi_v0",
        "manual_planet_harps_v0",
        "manual_planet_known_v0",
        "manual_planet_sophie_v0",
        "manual_planet_sphere_v0",
        "manual_planet_tess_eb_v0",
        "manual_planet_tess_pc_v0",
        "manual_planet_transiting_bd_v0",
        "manual_validation_apogee",
        "manual_validation_boss",
        "manual_validation_cool_apogee",
        "manual_validation_cool_boss",
        "manual_validation_rv",
        "mwm_bin_rv_long",
        "mwm_bin_rv_short",
        "mwm_cb_300pc",
        "mwm_cb_300pc_apogee",
        "mwm_cb_300pc_boss",
        "mwm_cb_cvcandidates",
        "mwm_cb_cvcandidates_apogee",
        "mwm_cb_cvcandidates_boss",
        "mwm_cb_gaiagalex",
        "mwm_cb_gaiagalex_apogee",
        "mwm_cb_gaiagalex_boss",
        "mwm_cb_uvex1",
        "mwm_cb_uvex2",
        "mwm_cb_uvex3",
        "mwm_cb_uvex4",
        "mwm_cb_uvex5",
        "mwm_dust_core",
        "mwm_erosita_compact",
        "mwm_erosita_compact_deep",
        "mwm_erosita_compact_gen",
        "mwm_erosita_compact_var",
        "mwm_erosita_stars",
        "mwm_galactic_core",
        "mwm_galactic_core_dist",
        "mwm_gg_core",
        "mwm_halo_bb",
        "mwm_halo_bb_apogee",
        "mwm_halo_bb_boss",
        "mwm_halo_sm",
        "mwm_halo_sm_apogee",
        "mwm_halo_sm_boss",
        "mwm_legacy_ir2opt",
        "mwm_ob_cepheids",
        "mwm_ob_core",
        "mwm_planet_tess",
        "mwm_rv_long_bplates",
        "mwm_rv_long-bplates",
        "mwm_rv_long_fps",
        "mwm_rv_long-fps",
        "mwm_rv_long_rm",
        "mwm_rv_long-rm",
        "mwm_rv_short_bplates",
        "mwm_rv_short-bplates",
        "mwm_rv_short_fps",
        "mwm_rv_short-fps",
        "mwm_rv_short_rm",
        "mwm_rv_short-rm",
        "mwm_snc_100pc",
        "mwm_snc_100pc_apogee",
        "mwm_snc_100pc_boss",
        "mwm_snc_250pc",
        "mwm_snc_250pc_apogee",
        "mwm_snc_250pc_boss",
        "mwm_tess_2min",
        "mwm_tess_ob",
        "mwm_tess_planet",
        "mwm_tess_rgb",
        "mwm_tessrgb_core",
        "mwm_wd_core",
        "mwm_wd_pwd",
        "mwm_yso_cluster",
        "mwm_yso_cluster_apogee",
        "mwm_yso_cluster_boss",
        "mwm_yso_cmz",
        "mwm_yso_cmz_apogee",
        "mwm_yso_disk_apogee",
        "mwm_yso_disk_boss",
        "mwm_yso_embedded_apogee",
        "mwm_yso_nebula_apogee",
        "mwm_yso_ob",
        "mwm_yso_ob_apogee",
        "mwm_yso_ob_boss",
        "mwm_yso_pms_apogee",
        "mwm_yso_pms_apogee_sagitta_edr3",
        "mwm_yso_pms_apogee_zari18pms",
        "mwm_yso_pms_boss",
        "mwm_yso_pms_boss_sagitta_edr3",
        "mwm_yso_pms_boss_zari18pms",
        "mwm_yso_s1",
        "mwm_yso_s2",
        "mwm_yso_s2-5",
        "mwm_yso_s3",
        "mwm_yso_variable_apogee",
        "mwm_yso_variable_boss",
        "openfiberstargets_test",
        "openfibertargets_nov2020_10",
        "openfibertargets_nov2020_1000",
        "openfibertargets_nov2020_1001a",
        "openfibertargets_nov2020_1001b",
        "openfibertargets_nov2020_11",
        "openfibertargets_nov2020_12",
        "openfibertargets_nov2020_14",
        "openfibertargets_nov2020_15",
        "openfibertargets_nov2020_17",
        "openfibertargets_nov2020_18",
        "openfibertargets_nov2020_19a",
        "openfibertargets_nov2020_19b",
        "openfibertargets_nov2020_19c",
        "openfibertargets_nov2020_22",
        "openfibertargets_nov2020_24",
        "openfibertargets_nov2020_25",
        "openfibertargets_nov2020_26",
        "openfibertargets_nov2020_27",
        "openfibertargets_nov2020_28a",
        "openfibertargets_nov2020_28b",
        "openfibertargets_nov2020_28c",
        "openfibertargets_nov2020_29",
        "openfibertargets_nov2020_3",
        "openfibertargets_nov2020_30",
        "openfibertargets_nov2020_31",
        "openfibertargets_nov2020_32",
        "openfibertargets_nov2020_33",
        "openfibertargets_nov2020_34a",
        "openfibertargets_nov2020_34b",
        "openfibertargets_nov2020_35a",
        "openfibertargets_nov2020_35b",
        "openfibertargets_nov2020_35c",
        "openfibertargets_nov2020_46",
        "openfibertargets_nov2020_47a",
        "openfibertargets_nov2020_47b",
        "openfibertargets_nov2020_47c",
        "openfibertargets_nov2020_47d",
        "openfibertargets_nov2020_47e",
        "openfibertargets_nov2020_5",
        "openfibertargets_nov2020_6a",
        "openfibertargets_nov2020_6b",
        "openfibertargets_nov2020_6c",
        "openfibertargets_nov2020_8",
        "openfibertargets_nov2020_9",
        "ops_2mass_psc_brightneighbors",
        "ops_apogee_stds",
        "ops_gaia_brightneighbors",
        "ops_sky_apogee",
        "ops_sky_apogee_best",
        "ops_sky_apogee_good",
        "ops_sky_boss",
        "ops_sky_boss_best",
        "ops_sky_boss_fallback",
        "ops_sky_boss_good",
        "ops_std_apogee",
        "ops_std_boss",
        "ops_std_boss_gdr2",
        "ops_std_boss_lsdr10",
        "ops_std_boss_lsdr8",
        "ops_std_boss_ps1dr2",
        "ops_std_boss_red",
        "ops_std_boss-red",
        "ops_std_boss_tic",
        "ops_std_eboss",
        "ops_tycho2_brightneighbors" 
    )


    kwds = dict(mapper_pk=0,category_pk=0, version_pk=0, program="")
    cartons = [Carton(carton=carton_name, **kwds) for carton_name in carton_names]
    with database.atomic():
        (
            Carton
            .bulk_create(cartons)
        )


    np.random.seed(0)

    C = len(cartons)
    N = 10_000 
    N_cartons_assigned_to = np.random.randint(0, C, size=N)    

    print("Preparing sources")
    assignments = {}
    
    sources = []
    for i in tqdm(range(N)):
        ra, dec = np.random.uniform(size=2)
        source = Source.create(ra=ra, dec=dec)

    
        carton_assignments = np.random.choice(cartons, size=N_cartons_assigned_to[i], replace=False)
        for carton in carton_assignments:
            source.carton_flags.set_bit(carton.pk)
            assignments.setdefault(carton.pk, [])
            assignments[carton.pk].append(source.id)
        
        source.save()
        sources.append(source)

    
    for carton_pk, expected_source_ids in tqdm(assignments.items(), desc="Checking.."):
        q = (
            Source
            .select(Source.sdss_id)
            .where(Source.in_carton(carton_pk))
            .tuples()
        )
        actual_source_ids = [source_id for source_id, in q]
        diff = set(expected_source_ids).symmetric_difference(actual_source_ids)
        assert len(diff) == 0




'''
class MWMStarBossSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(Source, index=True, backref="mwm_star_boss_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    release = TextField(help_text=Glossary.release)
    v_astra = TextField(help_text=Glossary.v_astra)
    run2d = TextField(help_text=Glossary.run2d)
    apred = TextField(help_text=Glossary.apred)
    catalogid = BigIntegerField(help_text=Glossary.catalogid, index=True)
    component = TextField(help_text=Glossary.component, default="")

    # mjd_start
    # mjd_end
    # n_visits

    telescope = TextField(help_text=Glossary.telescope)

    _get_ext = lambda x: dict(apo25m=1, lco25m=2)[x.telescope]
    flux = PixelArray(ext=_get_ext, column_name="flux", transform=lambda x: x[0])
    ivar = PixelArray(ext=_get_ext, column_name="e_flux", transform=lambda x: x[0]**-2)
    
    @property
    def path(self):
        return (
            f"$MWM_ASTRA/"
            f"{self.v_astra}/{self.run2d}-{self.apred}/spectra/star/"
            f"{(int(self.catalogid) // 100) % 100:0>2.0f}/{int(self.catalogid) % 100:0>2.0f}/"
            f"mwmStar-{self.v_astra}-{self.catalogid}{self.component}.fits"
        )

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "v_astra",
                    "run2d",
                    "apred",
                    "catalogid",
                    "component",
                    "telescope",
                    "mjd_start",
                    "mjd_end",
                ),
                True,
            ),
        )


# MWMStarApogeeSpectrum? ApogeeCoaddedSpectrum?
class MWMStarApogeeSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(Source, index=True, backref="mwm_star_apogee_spectra")
    spectrum_id = ForeignKeyField(UniqueSpectrum, index=True, lazy_load=False)

    release = TextField(help_text=Glossary.release)
    v_astra = TextField(help_text=Glossary.v_astra)
    run2d = TextField(help_text=Glossary.run2d)
    apred = TextField(help_text=Glossary.apred)
    catalogid = BigIntegerField(help_text=Glossary.catalogid, index=True)
    component = TextField(help_text=Glossary.component, default="")

    # mjd_start
    # mjd_end
    # n_visits

    telescope = TextField(help_text=Glossary.telescope)

    _get_ext = lambda x: dict(apo25m=3, lco25m=4)[x.telescope]
    flux = PixelArray(ext=_get_ext, column_name="flux", transform=lambda x: x[0])
    ivar = PixelArray(ext=_get_ext, column_name="e_flux", transform=lambda x: x[0]**-2)
    
    @property
    def path(self):
        return (
            f"$MWM_ASTRA/"
            f"{self.v_astra}/{self.run2d}-{self.apred}/spectra/star/"
            f"{(int(self.catalogid) // 100) % 100:0>2.0f}/{int(self.catalogid) % 100:0>2.0f}/"
            f"mwmStar-{self.v_astra}-{self.catalogid}{self.component}.fits"
        )

    class Meta:
        indexes = (
            (
                (
                    "release",
                    "v_astra",
                    "run2d",
                    "apred",
                    "catalogid",
                    "component",
                    "telescope",
                    "mjd_start",
                    "mjd_end",
                ),
                True,
            ),
        )

class MWMBossVisitSpectrum(BaseModel, Spectrum):

    source = ForeignKeyField(
        Source, 
        index=True, 
        backref="mwm_boss_visit_spectra",
        help_text=Glossary.source_id
    )
    spectrum_id = ForeignKeyField(
        UniqueSpectrum, 
        index=True, 
        lazy_load=False,
        help_text=Glossary.spectrum_id
    )
    
    #: Data product keywords
    release = TextField(help_text=Glossary.release)
    v_astra = TextField(help_text=Glossary.v_astra)
    run2d = TextField(help_text=Glossary.run2d)
    apred = TextField(help_text=Glossary.apred)
    mjd = IntegerField(help_text=Glossary.mjd)
    fieldid = IntegerField(help_text=Glossary.fieldid)
    catalogid = BigIntegerField(help_text=Glossary.catalogid)
    component = TextField(help_text=Glossary.component, default="")


    #: Observing conditions
    telescope = TextField(help_text=Glossary.telescope)
    alt = FloatField(null=True, help_text=Glossary.alt)
    az = FloatField(null=True, help_text=Glossary.az)
    seeing = FloatField(null=True, help_text=Glossary.seeing)
    airmass = FloatField(null=True, help_text=Glossary.airmass)
    airtemp = FloatField(null=True, help_text=Glossary.airtemp)
    dewpoint = FloatField(null=True, help_text=Glossary.dewpoint)
    humidity = FloatField(null=True, help_text=Glossary.humidity)
    pressure = FloatField(null=True, help_text=Glossary.pressure)
    gust_direction = FloatField(null=True, help_text=Glossary.gust_direction)
    gust_speed = FloatField(null=True, help_text=Glossary.gust_speed)
    wind_direction = FloatField(null=True, help_text=Glossary.wind_direction)
    wind_speed = FloatField(null=True, help_text=Glossary.wind_speed)
    moon_dist_mean = FloatField(null=True, help_text=Glossary.moon_dist_mean)
    moon_phase_mean = FloatField(null=True, help_text=Glossary.moon_phase_mean)
    n_exp = IntegerField(null=True, help_text=Glossary.n_exp)
    n_guide = IntegerField(null=True, help_text=Glossary.n_guide)
    tai_beg = DateTimeField(null=True, help_text=Glossary.tai_beg)
    tai_end = DateTimeField(null=True, help_text=Glossary.tai_end)
    fiber_offset = BooleanField(null=True, help_text=Glossary.fiber_offset)
    delta_ra = FloatField(null=True, help_text=Glossary.delta_ra)
    delta_dec = FloatField(null=True, help_text=Glossary.delta_dec)
    zwarning = IntegerField(null=True, help_text=Glossary.zwarning)

    #: Data reduction pipeline
    v_boss = TextField(null=True, help_text=Glossary.v_boss)
    v_jaeger = TextField(null=True, help_text=Glossary.v_jaeger)
    v_kaiju = TextField(null=True, help_text=Glossary.v_kaiju)
    v_coord = TextField(null=True, help_text=Glossary.v_coord)
    v_calibs = TextField(null=True, help_text=Glossary.v_calibs)
    v_idl = TextField(null=True, help_text=Glossary.v_idl)
    v_util = TextField(null=True, help_text=Glossary.v_util)
    v_read = TextField(null=True, help_text=Glossary.v_read)
    v_2d = TextField(null=True, help_text=Glossary.v_2d)
    v_comb = TextField(null=True, help_text=Glossary.v_comb)
    v_log = TextField(null=True, help_text=Glossary.v_log)
    v_flat = TextField(null=True, help_text=Glossary.v_flat)
    didflush = BooleanField(null=True, help_text=Glossary.didflush)
    cartid = TextField(null=True, help_text=Glossary.cartid)
    psfsky = IntegerField(null=True, help_text=Glossary.psfsky)
    preject = FloatField(null=True, help_text=Glossary.preject)
    lowrej = IntegerField(null=True, help_text=Glossary.lowrej)
    highrej = IntegerField(null=True, help_text=Glossary.highrej)
    scatpoly = IntegerField(null=True, help_text=Glossary.scatpoly)
    proftype = IntegerField(null=True, help_text=Glossary.proftype)
    nfitpoly = IntegerField(null=True, help_text=Glossary.nfitpoly)
    skychi2 = FloatField(null=True, help_text=Glossary.skychi2)
    schi2min = FloatField(null=True, help_text=Glossary.schi2min)
    schi2max = FloatField(null=True, help_text=Glossary.schi2max)
    rdnoise0 = FloatField(null=True, help_text=Glossary.rdnoise0)

    #: XCSAO pipeline
    xcsao_teff = FloatField(null=True, help_text=xcsao_glossary.teff)
    xcsao_e_teff = FloatField(null=True, help_text=xcsao_glossary.e_teff)
    xcsao_logg = FloatField(null=True, help_text=xcsao_glossary.logg)
    xcsao_e_logg = FloatField(null=True, help_text=xcsao_glossary.e_logg)
    xcsao_fe_h = FloatField(null=True, help_text=xcsao_glossary.fe_h)
    xcsao_e_fe_h = FloatField(null=True, help_text=xcsao_glossary.e_fe_h)
    xcsao_rxc = FloatField(null=True, help_text=xcsao_glossary.rxc)

    # TODO: 
    # [ ] used in stack
    # [ ] v_shift
    # [ ] v_bc
    # [ ] pixel_flags


    # TODO: accessor function that takes instance information as well so that we can use one lambda for all?

    _get_ext = lambda x: dict(apo25m=1, lco25m=2)[x.telescope]
    flux = PixelArray(ext=_get_ext, column_name="flux", transform=lambda x: x)
    ivar = PixelArray(ext=_get_ext, column_name="e_flux", transform=lambda x: x**-2) 
'''



"""
if __name__ == "__main__":
    database.create_tables([
        Source,
        UniqueSpectrum,
        BossVisitSpectrum,
        ApogeeVisitSpectrum,
        apMADGICSSpectrum
    ])

    source = Source.create(ra=1, dec=1, healpix=1)


    spectrum_id = UniqueSpectrum.create().id
    spec = apMADGICSSpectrum(
        source=source,
        spectrum_id=spectrum_id,
        row_index=1
    )
    #  "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u6039752/working/2023_03_24/input_list.txt"

    # Ingest each spectrum, get initial values for ASPCAP.

    '''
        ap = ApogeeVisitSpectrum.create(
        release="dr17",
        apred="dr17",
        prefix="ap",
        field="AQM_216.74_+36.33",
        fiber=26,
        mjd=59313,
        plate=15305,
        telescope="apo25m",
        spectrum_id=UniqueSpectrum.create().id, 
        source=source
    )
    
    raise a
    from astra.database.astradb import DataProduct

    dp = (
        DataProduct
        .select()
        .where(
            DataProduct.filetype == "specFull"
        )
        .order_by(DataProduct.id.desc())
        .first()
    )


    kwargs = {'mjd': 59146,
    'run2d': 'v6_0_9',
    'fieldid': 15000,
    'isplate': 'p',
    'catalogid': 4375787390}

    # At ingestion time:
    # - would like to get photometry information from catalog
    # - would like to have source information, but not necessary for running tests (eg with fake spectra)
    # - needs to ingest metadata from the file

    kwargs = {"mjd": 59976, "run2d": "v6_0_9", "fieldid": 102906, "isplate": "", "catalogid": 27021597917769274}        


    source = Source.create(ra=1, dec=1, healpix=1)
    spec = BossVisitSpectrum.create(
        release="sdss5", 
        run2d=kwargs["run2d"],
        fieldid=kwargs["fieldid"],
        catalogid=kwargs["catalogid"],
        mjd=kwargs["mjd"], 
        spectrum_id=UniqueSpectrum.create().id, 
        source=source, 
    )


    ap = ApogeeVisitSpectrum.create(
        release="sdss5",
        apred="1.0",
        prefix
        mjd=5000,
        plate="foo",
        telescope="apo25m",
        field="bar",
        fiber=200,
        spectrum_id=UniqueSpectrum.create().id, 
        source=source
    )
    
    raise a
    from astra.tools.spectrum import Spectrum1D


    from time import time


    t_init = time()
    spec.flux
    spec.ivar
    print(time() - t_init)

    N = spec.flux.size
    f = np.memmap("flux.memmap", dtype=np.float32, mode="w+", shape=spec.flux.shape)
    f[:] = spec.flux
    f.flush()
    del f
    
    #t_init = time()
    #np.memmap("flux.memmap", dtype=np.float32, offset=4, mode="r+", shape=10)
    #print(time() - t_init)



    t_init = time()
    s = Spectrum1D.read(spec.path)
    print(time() - t_init)

    '''

"""