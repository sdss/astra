from astra import models
from astra.utils import log, callable
from inspect import getfullargspec
from peewee import fn, JOIN
from astropy.time import Time
from airflow.exceptions import AirflowSkipException
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
    

    def where_by_execution_date(self, input_model, context, left_key="prev_execution_date", right_key="execution_date"):

        # A decision was made here.
        # If you cut between prev_execution_date and next_execution_date then it actually runs the same stuff 3x because a DAG on a daily frequency will
        # search 24 hrs before and 24 hrs after.

        # You can do between prev_execution_date and execution_date to get up until the current execution (probably the right thing), or you could do
        # a 'look ahead' between execution_date and next_execution_date.

        # It would seem like the int()s that follow are superfluous, but they are ABSOLUTELY NECESSARY.
        # This is because the .mjd attribute is a np.float64, which gets passed to SQL as a bool, and means we end up getting
        # spectra multiple times in different executions.

        where_by_execution_date = {
            "mjd": lambda m: (int(Time(context[left_key]).mjd) <= m.mjd) & (m.mjd < int(Time(context[right_key]).mjd)),
            "date_obs": lambda m: (Time(context[left_key]).datetime <= m.date_obs) & (m.date_obs < Time(context[right_key]).datetime),
            "max_mjd": lambda m: (int(Time(context[left_key]).mjd) <= m.max_mjd) & (m.max_mjd < int(Time(context[right_key]).mjd)),
        }
        if context[left_key] is not None and context[right_key] is not None:
            for k, f in where_by_execution_date.items():
                if hasattr(input_model, k):
                    log.info(f"Restricting {input_model} to have {k} between: {context[left_key]} <= {k} < {context[right_key]}")
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
