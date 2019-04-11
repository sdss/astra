
import datetime
from sqlalchemy import Boolean, Column, DateTime, String, Integer
from astra.db.connection import Base, Session

class WatchedFolders(Base):

    __tablename__ = "watched_folders"

    id = Column(Integer, primary_key=True)
    is_active = Column(Boolean, default=True)
    path = Column(String, nullable=False)
    update_interval_seconds = Column(Integer, default=3600)
    recursive = Column(Boolean, default=False)
    regex_pattern = Column(String, nullable=True)
    last_checked = Column(DateTime)
    created = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, path=self.path)>"


# Base.metadata.create_all(engine)