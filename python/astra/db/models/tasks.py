
import datetime
from sqlalchemy import (Boolean, Column, DateTime, String, Integer, ForeignKey,
                        UniqueConstraint)
from astra.db.connection import Base


class Task(Base):

    __tablename__ = "task"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("component.id"))
    data_subset_id = Column(Integer, ForeignKey("data_subset.id"))

    # TODO: worker_id ? how to represent the worker information?

    status = Column(String, default="CREATED")
    created = Column(DateTime, default=datetime.datetime.utcnow)
    scheduled = Column(DateTime, default=datetime.datetime.utcnow)
    modified = Column(DateTime, default=datetime.datetime.utcnow)

    log_path = Column(String)
    output_dir = Column(String)

    # TODO: resources!


    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, component_id={self.component_id}, data_subset_id={self.data_subset_id})>"