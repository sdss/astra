
import datetime
from sqlalchemy import (Boolean, Column, DateTime, String, Integer, ForeignKey,
                        UniqueConstraint)
from astra.db.connection import Base, Session

class DataProducts(Base):

    __tablename__ = "data_products"

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, unique=True)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    folder_id = Column(Integer, ForeignKey("watched_folders.id"))

    # TODO: datamodel_id ? Infer this from the path?

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, path={self.path})>"


class DataSubsets(Base):

    __tablename__ = "data_subsets"

    id = Column(Integer, primary_key=True)
    is_visible = Column(Boolean, default=True)

    name = Column(String, nullable=False, unique=True)
    regex_pattern_match = Column(String)
    auto_update = Column(Boolean, default=False)

    created = Column(DateTime, default=datetime.datetime.utcnow)

    # TODO: owner information
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, name={self.name})>"


class DataProductsSubsetsBridge(Base):

    __tablename__ = "data_products_subsets"

    data_subsets_id = Column(Integer, ForeignKey("data_subsets.id"))
    data_products_id = Colunn(Integer, ForeignKey("data_products.id"))

    __table_args__ = (
        UniqueConstraint("data_subsets_id", "data_products_id", name="_id_uc"),
    )

    def __repr__(self):
        return f"<{self.__class__.__name__}(subset_id={self.data_subsets_id}, product_id={self.data_products_id})>"
