
import luigi

class BaseTask(luigi.Task):

    def get_common_param_kwargs(self, klass):
        common_keys = set(klass.get_param_names()).intersection(self.get_param_names())
        return dict([(k, getattr(self, k)) for k in common_keys])