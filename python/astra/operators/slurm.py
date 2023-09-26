import json
from airflow.sensors.base import BaseSensorOperator

class SlurmSensor(BaseSensorOperator):

    template_fields = ("job_ids", )

    def __init__(self, job_ids, **kwargs) -> None:
        super(SlurmSensor, self).__init__(**kwargs)
        self.job_ids = job_ids
        return None

    def poke(self, context):
        try:
            job_ids = set(map(int, self.job_ids.split(" ")))
        except:
            try:
                job_ids = set(map(int, json.loads(self.job_ids)))
            except:
                print(f"No valid job IDs given: ({type(self.job_ids)}) {self.job_ids}")
                print(f"Returning success because no job identifier(s) to wait on")
                return True
        
        print(f"self job ids:({type(self.job_ids)}): {self.job_ids}")
        print(f"job ids: {job_ids}")
        from astra.utils.slurm import get_queue
                    
        queue = get_queue()

        in_queue_or_running = job_ids.intersection(queue)

        if in_queue_or_running:
            print(f"Jobs still running: {in_queue_or_running}")
        
        return (len(queue) > 0) and (not in_queue_or_running)

