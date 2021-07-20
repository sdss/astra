import json
import hashlib
from luigi.task_register import Register
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import AbstractConcreteBase, declared_attr
from sqlalchemy.orm import relationship
from sqlalchemy_utils import dependent_objects

from astra.database import AstraBase, database, session


class Base(AbstractConcreteBase, AstraBase):
    __abstract__ = True
    _schema = "astra_v02"
    _relations = "define_relations"


    @declared_attr
    def __table_args__(cls):
        return { "schema": cls._schema }



class TaskInstance(Base):
    __tablename__ = "ti"

    output_interface = relationship("OutputInterface", backref="ti")

    def _output_dependent_objects(self):
        if self.output_interface is not None:
            yield from dependent_objects(self.output_interface)

    @property
    def parameters(self):
        q = session.query(Parameter).join(TaskInstanceParameter).filter(TaskInstanceParameter.ti_pk==self.pk)
        return dict(((p.parameter_name, p.parameter_value) for p in q.all()))

    
    @property
    def output(self):
        # Tasks can only have one database target output. Sorry!
        if self.output_pk is None:
            return None
        
        for instance in dependent_objects(self.output_interface):
            if not isinstance(instance, self.__class__):

                return instance



class OutputInterface(Base):
    __tablename__ = "output_interface"


    @property
    def referenced_task_instances(self):
        """ A generator that yields tasks that point to this database output. """
        for instance in dependent_objects(self):
            if isinstance(instance, TaskInstance):
                yield instance



class TaskInstanceParameter(Base):
    __tablename__ = "ti_parameter"
    

class Parameter(Base):
    __tablename__ = "parameter"
    

class OutputMixin:

    def get_task_instances(self):
        return session.query(TaskInstance).filter_by(output_pk=self.output_pk).all()


class ApogeeNet(Base, OutputMixin):
    __tablename__ = "apogeenet"


class Doppler(Base, OutputMixin):
    __tablename__ = "doppler"


class TheCannon(Base, OutputMixin):
    __tablename__ = "thecannon"
    

class ThePayne(Base, OutputMixin):
    __tablename__ = "thepayne"
    

class Ferre(Base, OutputMixin):
    __tablename__ = "ferre"

class Aspcap(Base, OutputMixin):
    __tablename__ = "aspcap"

class Classification(Base, OutputMixin):
    __tablename__ = "classification"


class WDClassification(Base, OutputMixin):
    __tablename__ = "wd_classification"


def define_relations():    
    
    #Task.batch_interface = relationship("BatchInterface", backref="task", foreign_keys="BatchInterface.parent_task_pk")
    #Task.output_interface = relationship("OutputInterface", backref="task")


    pass


database.add_base(Base)
