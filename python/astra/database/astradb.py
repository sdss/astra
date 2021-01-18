import json
import hashlib
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship

from astra.new_database import AstraBase, database

class Base(AbstractConcreteBase, AstraBase):
    __abstract__ = True
    _schema = "astra"
    _relations = "define_relations"


    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }




class TaskState(Base):
    __tablename__ = "task_state"

    def __repr__(self):
        return f"<TaskState (task_id={self.task_id}, code={self.code}, pk={self.pk})>"


class TaskParameter(Base):
    __tablename__ = "task_parameter"
    
    def __repr__(self):
        return f"<TaskParameter ({self.pk:x}, pk={self.pk})>"


class ApogeeVisit(Base):
    __tablename__ = "apogee_visit"

    

class ApogeeStar(Base):
    __tablename__ = "apogee_star"


class BossSpec(Base):
    __tablename__ = "boss_spec"


class Classification(Base):
    __tablename__ = "classification"


class ClassificationClass(Base):
    __tablename__ = "classification_class"


class ContinuumNormalization(Base):
    __tablename__ = "continuum_normalization"


class ApogeeNet(Base):
    __tablename__ = "apogeenet"


class ThePayne(Base):
    __tablename__ = "thepayne"

class Ferre(Base):
    __tablename__ = "ferre"

class Aspcap(Base):
    __tablename__ = "aspcap"


def define_relations():    
    TaskState._parameter = relationship(TaskParameter, backref="task_state")
    TaskState.parameters = association_proxy("_parameter", "parameters")



database.add_base(Base)

# [ ] base task; when completed it triggers a database row to be created in task.
#       -> for batch tasks it does not put in the parameters_hash,
# [ ] 16-bit hex string, with task_pk as well
# [ ] .complete() task checks database row instead of outputs.
# [ ] 
