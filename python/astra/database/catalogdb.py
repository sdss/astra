from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship

from astra.database import AstraBase, database

class Base(AbstractConcreteBase, AstraBase):
    __abstract__ = True
    _schema = "catalogdb"
    _relations = "define_relations"

    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }


class SDSSDR16ApogeeStar(Base):
    __tablename__ = "sdss_dr16_apogeestar"


class SDSSVBossSpall(Base):
    __tablename__ = "sdssv_boss_spall"
    

def define_relations():
    pass

database.add_base(Base)