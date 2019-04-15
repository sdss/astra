
import datetime
from sqlalchemy import Boolean, Column, DateTime, String, Integer
from astra.db.connection import Base

class WatchedFolder(Base):

    __tablename__ = "watched_folder"

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    is_active = Column(Boolean, default=True)
    path = Column(String, nullable=False, unique=True)
    interval = Column(Integer, default=3600)
    recursive = Column(Boolean, default=False)
    regex_ignore_pattern = Column(String, nullable=True)

    last_checked = Column(DateTime)
    created = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, path={self.path})>"