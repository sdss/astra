
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


    teff = Column("teff", Array(Float))
    logg = Column("logg", Array(Float))
    metals = Column("metals", Array(Float))
    alpha_m = Column("o_mg_si_s_ca_ti", Array(Float))
    n_m = Column("n", Array(Float))
    c_m = Column("c", Array(Float))
    log10vdop = Column("log10vdop", Array(Float))
    # Not all grids have LGVSINI.
    lgvsini = Column("lgvsini", Array(Float), nullable=True)

    u_teff = Column("u_teff", Array(Float))
    u_logg = Column("u_logg", Array(Float))
    u_metals = Column("u_metals", Array(Float))
    u_alpha_m = Column("u_o_mg_si_s_ca_ti", Array(Float))
    u_n_m = Column("u_n", Array(Float))
    u_c_m = Column("u_c", Array(Float))
    u_log10vdop = Column("u_log10vdop", Array(Float))
    # Not all grids have lgvsini.
    u_lgvsini = Column("u_lgvsini", Array(Float), nullable=True)

    log_snr_sq = Column("log_snr_sq", Array(Float))
    log_chisq_fit = Column("log_chisq_fit", Array(Float))
    
    
    @property
    def table_name(self):
        return "ferre"


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
        task_namespace = task_family.split(".")[0].lower()
        
        db = BaseDatabaseTarget(
            self.connection_string,
            task_namespace,    
            task_id,
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
