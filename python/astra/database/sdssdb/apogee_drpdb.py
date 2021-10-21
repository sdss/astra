from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship

from astra.database.sdssdb import SDSSBase, database

class Base(AbstractConcreteBase, SDSSBase):
    __abstract__ = True
    _schema = "apogee_drp"
    _relations = "define_relations"


    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }


class Visit(Base):
    __tablename__ = "visit"


class Star(Base):
    __tablename__ = "star"



def define_relations():
    pass


database.add_base(Base)
