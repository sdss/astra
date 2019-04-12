
import datetime
from sqlalchemy import Boolean, Column, DateTime, String, Integer, UniqueConstraint
from astra.db.connection import Base

class Components(Base):

    __tablename__ = "components"

    id = Column(Integer, primary_key=True)
    is_active = Column(Boolean, default=True)
    auto_update = Column(Boolean, default=True)
    # TODO: consider a check interval?

    github_repo_slug = Column(String, nullable=False)
    tag = Column(String, nullable=False)
    short_name = Column(String, nullable=True)
    long_description = Column(String, nullable=True)

    owner_name = Column(String, nullable=False)
    owner_email_address = Column(String, nullable=False)

    execution_order = Column(Integer, default=0)
    component_cli = Column(String, nullable=False)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    modified = Column(DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("github_repo_slug", "tag", name="_slug_tag_uc"),
    )

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, github_repo_slug={self.github_repo_slug}, tag={self.tag})>"


#Base.metadata.create_all(engine)