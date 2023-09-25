import datetime
from astropy.io import fits
from astra import __version__

from astra.glossary import Glossary
from astra.products.utils import (add_category_headers, add_category_comments, warn_on_long_name_or_comment)


DATETIME_FMT = "%y-%m-%d %H:%M:%S"
BLANK_CARD = (" ", " ", None)
FILLER_CARD = (FILLER_CARD_KEY, *_) = ("TTYPE0", "Water cuggle", None)


def create_source_primary_hdu(
    source, 
    ignore_fields=(
        "sdss5_target_flags", 
        "sdss4_apogee_target1_flags", 
        "sdss4_apogee_target2_flags",
        "sdss4_apogee2_target1_flags", 
        "sdss4_apogee2_target2_flags",         
        "sdss4_apogee2_target3_flags",
        "sdss4_apogee_member_flags", 
        "sdss4_apogee_extra_target_flags",
    ),
    upper=True
):

    shortened_names = {
        "GAIA_DR2_SOURCE_ID": "GAIA2_ID",
        "GAIA_DR3_SOURCE_ID": "GAIA3_ID",
        "SDSS4_APOGEE_ID": "APOGEEID",
        "TIC_V8_ID": "TIC_ID",
        "VERSION_ID": "VER_ID",
        "SDSS5_CATALOGID_V1": "CAT_ID",
        "GAIA_V_RAD": "V_RAD",
        "GAIA_E_V_RAD": "E_V_RAD",
        "ZGR_TEFF": "Z_TEFF",
        "ZGR_E_TEFF": "Z_E_TEFF",
        "ZGR_LOGG": "Z_LOGG",
        "ZGR_E_LOGG": "Z_E_LOGG",
        "ZGR_FE_H": "Z_FE_H",
        "ZGR_E_FE_H": "Z_E_FE_H",
        "ZGR_E": "Z_E",
        "ZGR_E_E": "Z_E_E",
        "ZGR_PLX": "Z_PLX",
        "ZGR_E_PLX": "Z_E_PLX",
        "ZGR_TEFF_CONFIDENCE": "Z_TEFF_C",
        "ZGR_LOGG_CONFIDENCE": "Z_LOGG_C",
        "ZGR_FE_H_CONFIDENCE": "Z_FE_H_C",
        "ZGR_QUALITY_FLAGS": "Z_FLAGS",
    }

    created = datetime.datetime.utcnow().strftime(DATETIME_FMT)

    cards = [
        BLANK_CARD,
        (" ", "METADATA", None),
        BLANK_CARD,
        ("V_ASTRA", __version__, Glossary.v_astra),
        ("CREATED", created, f"File creation time (UTC {DATETIME_FMT})"),
    ]
    original_names = {}
    for name, field in source._meta.fields.items():
        if ignore_fields is None or name not in ignore_fields:
            use_name = shortened_names.get(name.upper(), name.upper())

            value = getattr(source, name)

            cards.append((use_name, value, field.help_text))
            original_names[use_name] = name

        warn_on_long_name_or_comment(field)

    cards.extend([
        BLANK_CARD,
        (" ", "HDU DESCRIPTIONS", None),
        BLANK_CARD,
        ("COMMENT", "HDU 0: Summary information only", None),
        ("COMMENT", "HDU 1: BOSS spectra taken at Apache Point Observatory"),
        ("COMMENT", "HDU 2: BOSS spectra taken at Las Campanas Observatory"),
        ("COMMENT", "HDU 3: APOGEE spectra taken at Apache Point Observatory"),
        ("COMMENT", "HDU 4: APOGEE spectra taken at Las Campanas Observatory"),
    ])
    

    hdu = fits.PrimaryHDU(header=fits.Header(cards=cards))

    # Add category groupings.
    add_category_headers(hdu, (source.__class__, ), original_names, upper, use_ttype=False)
    add_category_comments(hdu, (source.__class__, ), original_names, upper, use_ttype=False)

    # Add checksums.
    hdu.add_checksum()
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.header.insert("CHECKSUM", (" ", "DATA INTEGRITY"))
    hdu.header.insert("CHECKSUM", BLANK_CARD)
    hdu.add_checksum()
    return hdu
