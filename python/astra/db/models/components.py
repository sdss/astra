
import datetime
from sqlalchemy import Boolean, Column, DateTime, String, Integer, UniqueConstraint
from astra.db.connection import Base

class Component(Base):

    __tablename__ = "component"

    id = Column(Integer, primary_key=True)
    is_active = Column(Boolean, default=True)
    auto_update = Column(Boolean, default=True)
    # TODO: consider a check interval?

    github_repo_slug = Column(String, nullable=False)
    release = Column(String, nullable=False)

    short_name = Column(String, nullable=True)
    long_description = Column(String, nullable=True)

    owner_name = Column(String, nullable=True)
    owner_email_address = Column(String, nullable=True)

    execution_order = Column(Integer, default=0)
    component_cli = Column(String, nullable=False)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    modified = Column(DateTime, default=datetime.datetime.utcnow)

    local_path = Column(String)

    __table_args__ = (
        UniqueConstraint("github_repo_slug", "release", name="_repo_release"),
    )

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, github_repo_slug={self.github_repo_slug}, release={self.release})>"


#Base.metadata.create_all(engine)