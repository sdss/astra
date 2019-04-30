
import datetime
from sqlalchemy import (Boolean, Column, DateTime, String, Integer, ForeignKey,
                        UniqueConstraint)
from astra.db.connection import Base


class Task(Base):

    __tablename__ = "task"

    id = Column(Integer, primary_key=True)
    subset_id = Column(Integer, ForeignKey("data_subset.id"))
    component_id = Column(Integer, ForeignKey("component.id"))

    args = Column(String, nullable=True)

    status = Column(String, default="CREATED")

    created = Column(DateTime, default=datetime.datetime.utcnow)
    scheduled = Column(DateTime, default=datetime.datetime.utcnow)
    modified = Column(DateTime, default=datetime.datetime.utcnow)

    output_dir = Column(String)

    def __repr__(self):
        return (f"<{self.__class__.__name__}(id={self.id}, component_id={self.component_id}, "
                f"subset_id={self.subset_id}, status={self.status})>")

