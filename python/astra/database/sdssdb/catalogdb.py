from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship

from astra.database.sdssdb import SDSSBase, database

class Base(AbstractConcreteBase, SDSSBase):
    __abstract__ = True
    _schema = "catalogdb"
    _relations = "define_relations"

    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }


class SDSSDR16ApogeeStar(Base):
    __tablename__ = "sdss_dr16_apogeestar"

class SDSSDR16ApogeeVisit(Base):
    __tablename__ = "sdss_dr16_apogeevisit"


class SDSSVBossSpall(Base):
    __tablename__ = "sdssv_boss_spall"
    

class SDSSApogeeAllStarMergeR13(Base):
    __tablename__ = "sdss_apogeeallstarmerge_r13"


class Catalog(Base):
    __tablename__ = "catalog"


class TICV8(Base):
    __tablename__ = "tic_v8"


class CatalogToTICV8(Base):
    __tablename__ = "catalog_to_tic_v8"


class GaiaDR2Source(Base):
    __tablename__ = "gaia_dr2_source"
    

def define_relations():
    pass

database.add_base(Base)