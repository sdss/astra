from astra.database.sdss5db import ReflectBaseModel

class ApogeeDRPBaseModel(ReflectBaseModel):
    class Meta:
        use_reflection = True
        schema = "apogee_drp"
    

class Star(ApogeeDRPBaseModel):
    class Meta:
        table_name = "star"

class Visit(ApogeeDRPBaseModel):
    class Meta:
        table_name = "visit"    