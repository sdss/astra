import json
import hashlib
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship
from sqlalchemy_utils import dependent_objects

from astra.database import AstraBase, database

class Base(AbstractConcreteBase, AstraBase):
    __abstract__ = True
    _schema = "astra"
    _relations = "define_relations"


    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }



class Task(Base):
    __tablename__ = "task"

    output_interface = relationship("OutputInterface", backref="task")

    def _output_dependent_objects(self):
        if self.output_interface is not None:
            yield from dependent_objects(self.output_interface)


    @property
    def output(self):
        # Tasks can only have one database target output. Sorry!
        if self.output_interface is None:
            return None
        
        for instance in dependent_objects(self.output_interface):
            if not isinstance(instance, self.__class__):
                return instance




class OutputInterface(Base):
    __tablename__ = "output_interface"


    @property
    def referenced_tasks(self):
        """ A generator that yields tasks that point to this database output. """
        for instance in dependent_objects(self):
            if isinstance(instance, Task):
                yield instance

        


class TaskState(Base):
    __tablename__ = "task_state"

    print_keys = ["pk", "status_code", "task_id"]

    #def __repr__(self):
    #    return f"<TaskState (task_id={self.task_id}, code={self.status_code}, pk={self.pk})>"

class TheCannon(Base):
    __tablename__ = "thecannon"
    


class TaskOutput(Base):
    __tablename__ = "task_output"


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


#class ClassificationClass(Base):
#    __tablename__ = "classification_class"


#class ContinuumNormalization(Base):
#    __tablename__ = "continuum_normalization"


class ApogeeNet(Base):
    __tablename__ = "apogeenet"


class ThePayne(Base):
    __tablename__ = "thepayne"

class Ferre(Base):
    __tablename__ = "ferre"

class Aspcap(Base):
    __tablename__ = "aspcap"


class SDSS4ApogeeStar(Base):
    __tablename__ = "sdss4_apogee_star"


class SDSS4ApogeeVisit(Base):
    __tablename__ = "sdss4_apogee_visit"


def define_relations():    
    
    #Output.result = relationship("", backref="output")
    #TaskState._parameter = relationship(TaskParameter, backref="task_state")
    #TaskState.parameters = association_proxy("_parameter", "parameters")
    pass


database.add_base(Base)