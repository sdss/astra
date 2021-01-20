from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship

from astra.database import AstraBase, database

class Base(AbstractConcreteBase, AstraBase):
    __abstract__ = True
    _schema = "apogee_drp"
    _relations = "define_relations"


    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }



class Visit(Base):
    __tablename__ = "visit"



def define_relations():
    pass


database.add_base(Base)
