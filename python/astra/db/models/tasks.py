
import datetime
from sqlalchemy import (Boolean, Column, DateTime, String, Integer, ForeignKey,
                        UniqueConstraint)
from astra.db.connection import Base


class Tasks(Base):

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"))

    # TODO: worker_id ? how to represent the worker information?

    status = Column(String, default="CREATED")
    created = Column(DateTime, default=datetime.datetime.utcnow)
    scheduled = Column(DateTime, default=datetime.datetime.utcnow)

    subset_id = Column(Integer, ForeignKey("data_subsets.id"))

    log_path = Column(String, nullable=False)
    output_dirname = Column(String, nullable=False)

    # TOOD: assign someone who 'owns' this task

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, component_id={self.component_id}, subset_id={self.subset_id})>"