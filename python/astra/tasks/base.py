
import luigi

class BaseTask(luigi.Task):

    def get_common_param_kwargs(self, klass, include_significant=True):
        a = self.get_param_names(include_significant=include_significant)
        b = klass.get_param_names(include_significant=include_significant)
        return dict([(k, getattr(self, k)) for k in set(a).intersection(b)])