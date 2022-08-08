# TODO: This should be re-organised with other SDSS stuff, away from just "operators"
from peewee import JOIN
from sdssdb.peewee.sdss5db import database as sdss5_database
sdss5_database.set_profile("operations") # TODO: HOW CAN WE SET THIS AS DEFAULT!?!

from sdssdb.peewee.sdss5db.catalogdb import (Catalog, Gaia_DR2, CatalogToTIC_v8, TIC_v8)
from sdssdb.peewee.sdss5db.targetdb import (Target, CartonToTarget, Carton)

from astra.utils import flatten

def get_sky_position(catalogid):
    return (
        Catalog.select(
                    Catalog.ra,
                    Catalog.dec
                )
                .where(Catalog.catalogid == catalogid)
                .tuples()
                .first()
    )

def get_gaia_dr2_photometry(catalogid):
    """
    Return the Gaia DR2 photometry given a SDSS-V catalog identifier.
    
    """
    from sdssdb.peewee.sdss5db import database as sdss5_database

    # Need to set the profile before importing catalogdb, otherwise we don't
    # get the relationship tables.

    return (
        Gaia_DR2.select()
                .join(TIC_v8, JOIN.LEFT_OUTER)
                .join(CatalogToTIC_v8)
                .join(Catalog, JOIN.LEFT_OUTER)
                .where(Catalog.catalogid == catalogid)
                .dicts()
                .first()
    )



def filter_sources(
        catalogids, 
        cartons=None, 
        programs=None,
        mappers=None
    ):
    """
    Filter catalog sources that belong to a given carton or program.

    :param catalogids:
        A list-like of catalogids.
    
    :param cartons: [optional]
        The name (or tuple or names) of the carton(s) to filter by (e.g., mwm_wd_core).
    
    :param programs: [optional]
        The name (or tuple of names) of the program(s) to filter by (e.g., mwm_gg).

    :param mappers: [optional]
        The name (or tuple of names) of the mapper(s) to filter by (e.g., mwm). This just
        checks if the `Carton.program` starts with the matching text (e.g., `Carton.program.startswith('mwm_')`).
    """
    q = (
        Target.select(Target.catalogid)
              .join(CartonToTarget)
              .join(Carton)
              .where(Target.catalogid.in_(catalogids))
    )

    if cartons is not None:
        cartons = (cartons, ) if isinstance(cartons, str) else cartons
        for carton in cartons:
            q = q.where(Carton.carton == carton)
    
    if programs is not None:
        programs = (programs, ) if isinstance(programs, str) else programs
        for program in programs:
            q = q.where(Carton.program == program)
    
    if mappers is not None:
        mappers = (mappers, ) if isinstance(mappers, str) else mappers
        for mapper in mappers:
            q = q.where(Carton.program.startswith(f"{mapper}_"))
    
    return flatten(q.tuples())
