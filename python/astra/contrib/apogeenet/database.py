from astra.database.astradb import (AstraBaseModel, Task, Output)
from peewee import (ForeignKeyField, IntegerField, FloatField)


class ApogeeNet(AstraBaseModel):

    output = ForeignKeyField(Output, on_delete="CASCADE", primary_key=True)
    task = ForeignKeyField(Task)

    snr = FloatField()
    teff = FloatField()
    logg = FloatField()
    fe_h = FloatField()
    u_teff = FloatField()
    u_logg = FloatField()
    u_fe_h = FloatField()
    teff_sample_median = FloatField()
    logg_sample_median = FloatField()
    fe_h_sample_median = FloatField()
    bitmask_flag = IntegerField()
