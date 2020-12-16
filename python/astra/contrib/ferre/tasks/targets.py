
import json
import re
from astra.tasks.targets import (LocalTarget, DatabaseTarget, BaseDatabaseTarget)
from sqlalchemy import (Column, Float, String)
from sqlalchemy.types import ARRAY as Array
from luigi import ExternalTask, Parameter, DictParameter

class FerreResult(DatabaseTarget):

    """ A database row representing an output target from FERRE. """

    # We use arrays to allow for multiple analyses (individual visits and stacks)
    # per ApStar object.

    teff = Column("TEFF", Array(Float))
    logg = Column("LOGG", Array(Float))
    metals = Column("METALS", Array(Float))
    alpha_m = Column("O Mg Si S Ca Ti", Array(Float))
    n_m = Column("N", Array(Float))
    c_m = Column("C", Array(Float))
    log10vdop = Column("LOG10VDOP", Array(Float))
    # Not all grids have LGVSINI.
    lgvsini = Column("LGVSINI", Array(Float), nullable=True)

    e_teff = Column("E_TEFF", Array(Float))
    e_logg = Column("E_LOGG", Array(Float))
    e_metals = Column("E_METALS", Array(Float))
    e_alpha_m = Column("E_O Mg Si S Ca Ti", Array(Float))
    e_n_m = Column("E_N", Array(Float))
    e_c_m = Column("E_C", Array(Float))
    e_log10vdop = Column("E_LOG10VDOP", Array(Float))
    # Not all grids have LGVSINI.
    e_lgvsini = Column("E_LGVSINI", Array(Float), nullable=True)

    log_snr_sq = Column("log_snr_sq", Array(Float))
    log_chisq_fit = Column("log_chisq_fit", Array(Float))
    


class FerreResultProxy(DatabaseTarget):

    proxy_task_id = Column("proxy_task_id", String(length=128))

    def copy_from(self, source):
        return self.write(dict(proxy_task_id=source.task_id))

    def read(self, as_dict=False, include_parameters=True):
        return self.resolve().read(as_dict=as_dict, include_parameters=include_parameters)


    def resolve(self):
        """ Resolve the proxy of this object and return the FerreResult target. """

        proxy_task_id, = self._read(self.table_bound, as_dict=False, include_parameters=False)
        task_id = proxy_task_id
        task_family = task_id.split("_")[0]
        task_namespace = task_family.split(".")[0]
        
        db = BaseDatabaseTarget(
            self.connection_string,
            task_namespace,    
            task_family,
            task_id,
            []
        )

        # TODO: Should probably resolve this the same way luigi does..
        from astra.contrib.ferre.tasks.aspcap import FerreGivenApStarFile
        kwds = db.read(as_dict=True, include_parameters=True)
        task_kwds = {}
        for k, p in FerreGivenApStarFile.get_params():

            try:
                value = kwds[k]

            except KeyError:
                try:
                    parsed_value = getattr(self, k)

                except AttributeError:
                    # Default value.
                    continue
            else:
                try:
                    parsed_value = p.parse(value)

                except json.JSONDecodeError:

                    # See https://stackoverflow.com/questions/39491420/python-jsonexpecting-property-name-enclosed-in-double-quotes
                    # Basically, postgres/sql can store the dict parameters using single quotes around keys, but to
                    # parse this back in Python, JSON *requires* double quotes. So we need to do a hack to fix this.
                    if isinstance(p, DictParameter):
                        sub = re.compile('(?<!\\\\)\'')
                        value = sub.sub('\"', value)

                        # And it turns out Nones are shit.
                        # https://stackoverflow.com/questions/3548635/python-json-what-happened-to-none
                        value = value.replace(" None", " null")
                        parsed_value = p.parse(value)
                    else:
                        raise
            
            task_kwds[k] = parsed_value

        task = FerreGivenApStarFile(**task_kwds)
        assert task.task_id == proxy_task_id
        return FerreResult(task)
