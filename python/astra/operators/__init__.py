from astra import models
from astra.utils import log, callable
from inspect import getfullargspec
from peewee import fn, JOIN
from astropy.time import Time

try:
    from airflow.models.baseoperator import BaseOperator
except ImportError:
    log.warning(f"Cannot import `airflow`: this functionality will not be available")
    BaseOperator = object


class Operator(BaseOperator):

    template_fields = ("task_kwargs", "limit", "model_name", "where")

    def __init__(
        self,
        task_name,
        model_name=None,
        task_kwargs=None,
        where=None,
        limit=None,
        **kwargs
    ):
        super(Operator, self).__init__(**kwargs)
        self.task_name = task_name
        self.model_name = model_name
        self.task_kwargs = task_kwargs or {}
        self.where = where
        self.limit = limit
        return None
    

    def where_by_execution_date(self, input_model, context):
        where_by_execution_date = {
            "mjd": lambda m: m.mjd.between(Time(context["next_execution_date"]).mjd, Time(context["prev_execution_date"]).mjd),
            "date_obs": lambda m: m.date_obs.between(Time(context["next_execution_date"]).datetime, Time(context["prev_execution_date"]).datetime),
            "max_mjd": lambda m: m.max_mjd.between(Time(context["next_execution_date"]).mjd, Time(context["prev_execution_date"]).mjd),
        }

        if context["next_execution_date"] is not None and context["prev_execution_date"] is not None:
            for k, f in where_by_execution_date.items():
                if hasattr(input_model, k):
                    log.info(f"Restricting {input_model} by {k} between {context['next_execution_date']} and {context['prev_execution_date']}")
                    return f(input_model)
                
        return None


    def execute(self, context):
        
        kwds = self.task_kwargs.copy()

        task = callable(self.task_name) 
        if self.model_name is not None:

            # Query for spectra that does not have a result in this output model
            # translate `-> Iterable[OutputModel]` annotation
            output_model = getfullargspec(task).annotations["return"].__args__[0]
            input_model = getattr(models, self.model_name)
            q = (
                input_model
                .select()
                .join(
                    output_model,
                    JOIN.LEFT_OUTER,
                    on=(input_model.spectrum_pk == output_model.spectrum_pk)
                )
            )
            where = output_model.spectrum_pk.is_null()
            for k, v in (self.where or {}).items():
                where = where & (getattr(input_model, k) == v)

            where_by_execution_date = self.where_by_execution_date(input_model, context)
            if where_by_execution_date is not None:
                where = where & where_by_execution_date

            '''
            # If the DAG has many active runs, then we will add a clause to modulate the spectra
            # so DAG executions do not try executing the same spectra at once.
            if self.modulate_spectra and context["dag"].max_active_runs > 1:                
                max_active_runs = context["dag"].max_active_runs
                remainder = int(Time(context["dag_run"].logical_date).mjd % max_active_runs)

                log.info(f"Modulating spectra because there are {max_active_runs} active runs.")
                log.info(f"Requiring spectra with spectrum_pk % {max_active_runs} == {remainder}")
                
                q = q.where(
                    fn.mod(input_model.spectrum_pk, max_active_runs) == remainder
                )
            '''

            q = (
                q
                .where(where)
                .limit(self.limit)
            )
            kwds.setdefault("spectra", q)

        n = 0 
        for n, item in enumerate(task(**kwds), start=1):
            None
        print(f"There were {n} results")
    
