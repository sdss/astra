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
    def batch_tasks(self):
        pks = (bi.child_task_pk for bi in self.batch_interface)
        return tuple(session.query(self.__class__).filter(self.__class__.pk.in_(pks)).all())


    @property
    def parameters(self):
        if self.batch_interface:
            # Assemble from individual tasks.
            # We will need to know which parameters are batch parameters.
            task_cls = self._get_task_class()
            batch_param_names = task_cls.batch_param_names()

            kwds = {}
            for i, task in enumerate(self.batch_tasks):
                if i == 0:
                    kwds.update(task.parameters)
                    kwds.update({ key: [] for key in batch_param_names })
                
                for key in batch_param_names:
                    kwds[key].append(task.parameters[key])
            
            return kwds
            
        else:
            q = session.query(Parameter).join(TaskParameter).filter(TaskParameter.task_pk==self.pk)
            return dict(((p.parameter_name, p.parameter_value) for p in q.all()))


    
    @property
    def output(self):
        # Tasks can only have one database target output. Sorry!
        if self.output_pk is None:
            return None
        
        for instance in dependent_objects(self.output_interface):
            if not isinstance(instance, self.__class__):

                return instance

    def _get_task_class(self):
        if self.task_module is not None:
            __import__(self.task_module)
        
        task_name, task_hash = self.task_id.split("_")
        return Register.get_task_cls(task_name)


    def load_task(self):
        """ Recreate the task instance from this database record. """

        task = self._get_task_class().from_str_params(self.parameters)
        assert task.task_id == self.task_id
        return task
        


class OutputInterface(Base):
    __tablename__ = "output_interface"


    @property
    def referenced_tasks(self):
        """ A generator that yields tasks that point to this database output. """
        for instance in dependent_objects(self):
            if isinstance(instance, Task):
                yield instance


class BatchInterface(Base):
    __tablename__ = "batch_interface"




class TaskParameter(Base):
    __tablename__ = "task_parameter"
    

class Parameter(Base):
    __tablename__ = "parameter"
    

class OutputMixin:

    def get_tasks(self):
        return session.query(Task).filter_by(output_pk=self.output_pk).all()


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
    
    Task.batch_interface = relationship("BatchInterface", backref="task", foreign_keys="BatchInterface.parent_task_pk")
    #Task.output_interface = relationship("OutputInterface", backref="task")


    pass


database.add_base(Base)
