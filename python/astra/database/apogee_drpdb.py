from peewee import (fn, TextField, SmallIntegerField)
from playhouse.hybrid import (hybrid_method, hybrid_property)
from astra.database.sdss5db import ReflectBaseModel

class ApogeeDRPBaseModel(ReflectBaseModel):
    class Meta:
        use_reflection = True
        schema = "apogee_drp"
    

class Star(ApogeeDRPBaseModel):
    class Meta:
        table_name = "star"
        #print_fields = ["pk", "obj", "healpix", "telescope", "apred", "apstar"]

    # Fix inconsistencies between the apogee_drp.star table and the ApStar data model.
    # These have been raised with Nidever a few times.
    # TODO: Raise this issue again.
    apred = TextField(column_name="apred_vers")
    obj = TextField(column_name="apogee_id")

    # These columns are missing from the database, but exist in the ApStar data model:
    @hybrid_property
    def release(self):
        return "sdss5"
    
    @hybrid_property
    def apstar(self):
        return "stars"
    
    @hybrid_property
    def filetype(self):
        return "apStar"


class Visit(ApogeeDRPBaseModel):
    class Meta:
        table_name = "visit"    
        #print_fields = ["pk", "fiber", "plate", "mjd", "field", "telescope", "apred"]

    # Fix inconsistencies between the apogee_drp.visit table and the ApVisit data model.
    # These have been raised with Nidever a few times.
    # TODO: Raise this issue again.
    apred = TextField(column_name="apred_vers")
    obj = TextField(column_name="apogee_id")
    fiber = SmallIntegerField(column_name="fiberid")

    # These columns are missing from the database, but exist in the ApVisit data model:
    @hybrid_property
    def prefix(self):
        return self.file.strip()[:2]
    
    @prefix.expression
    def prefix(self):
        return fn.left(self.file, 2)

    @hybrid_property
    def release(self):
        return "sdss5"
        
    @hybrid_property
    def filetype(self):
        return "apVisit"
