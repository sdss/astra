from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
from sqlalchemy import (Boolean, Column, DateTime, String, Integer, ForeignKey,
                        UniqueConstraint)
from astra.db.connection import Base

class DataProduct(Base):

    __tablename__ = "data_product"

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, unique=True)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    folder_id = Column(Integer, ForeignKey("folder.id"))

    # TODO: datamodel_id ? Infer this from the path?

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, path={self.path})>"


class DataSubset(Base):

    __tablename__ = "data_subset"

    id = Column(Integer, primary_key=True)
    is_visible = Column(Boolean, default=True)

    # Name should be unique if it is visible.
    name = Column(String)
    regex_pattern_match = Column(String)
    auto_update = Column(Boolean, default=False)

    created = Column(DateTime, default=datetime.datetime.utcnow)
    modified = Column(DateTime, default=datetime.datetime.utcnow)

    # TODO: owner information
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id}, name={self.name})>"


class DataProductSubsetBridge(Base):

    __tablename__ = "data_product_data_subset"

    id = Column(Integer, primary_key=True)
    data_subset_id = Column(Integer, ForeignKey("data_subset.id"))
    data_product_id = Column(Integer, ForeignKey("data_product.id"))

    __table_args__ = (
        UniqueConstraint("data_subset_id", "data_product_id", name="_id_uc"),
    )

    def __repr__(self):
        return f"<{self.__class__.__name__}(subset_id={self.data_subset_id}, product_id={self.data_product_id})>"