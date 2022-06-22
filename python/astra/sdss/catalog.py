# TODO: This should be re-organised with other SDSS stuff, away from just "operators"
from peewee import JOIN

def get_gaia_dr2_photometry(catalogid):
    """
    Return the Gaia DR2 photometry given a SDSS-V catalog identifier.
    
    """
    from sdssdb.peewee.sdss5db import database as sdss5_database

    # Need to set the profile before importing catalogdb, otherwise we don't
    # get the relationship tables.
    # TODO: HOW CAN WE SET THIS AS DEFAULT!?!
    sdss5_database.set_profile("operations")
    from sdssdb.peewee.sdss5db.catalogdb import (Catalog, Gaia_DR2, CatalogToTIC_v8, TIC_v8)

    return (
        Gaia_DR2.select()
                .join(TIC_v8, JOIN.LEFT_OUTER)
                .join(CatalogToTIC_v8)
                .join(Catalog, JOIN.LEFT_OUTER)
                .where(Catalog.catalogid == catalogid)
                .dicts()
                .first()
    )


