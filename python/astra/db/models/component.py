
import datetime
from sqlalchemy import Boolean, Column, DateTime, String, Integer, UniqueConstraint
from astra.db.connection import Base

class Component(Base):

    __tablename__ = "component"

    id = Column(Integer, primary_key=True)
    is_active = Column(Boolean, default=True)
    auto_update = Column(Boolean, default=True)

    # TODO: consider a check interval?

    owner = Column(String, nullable=False)
    product = Column(String, nullable=False)
    version = Column(String, nullable=False)


    description = Column(String, nullable=True)
    module_name = Column(String, nullable=False)

    execution_order = Column(Integer, default=0)
    command = Column(String, nullable=False)
    default_args = Column(String, nullable=True)

    created = Column(DateTime, default=datetime.datetime.utcnow)
    modified = Column(DateTime, default=datetime.datetime.utcnow)

    local_path = Column(String)

    __table_args__ = (
        UniqueConstraint("owner", "product", "version", name="_unique_component"),
    )

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, owner={self.owner}, "\
               f"product={self.product}, version={self.version})>"
