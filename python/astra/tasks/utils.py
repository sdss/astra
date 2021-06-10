import hashlib
import json
from luigi.parameter import _DictParamEncoder

def hashify(params, max_length=8):
    """
    Create a short hashed string of the given parameters.

    :param params:
        A dictionary of key, value pairs for parameters.
    
    :param max_length: [optional]
        The maximum length of the hashed string.
    """
    param_str = json.dumps(params, cls=_DictParamEncoder, separators=(',', ':'), sort_keys=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    return param_hash[:max_length]


def task_id_str(task_family, params):
    """
    Returns a canonical string used to identify a particular task
    :param task_family: The task family (class name) of the task
    :param params: a dict mapping parameter names to their serialized values
    :return: A unique, shortened identifier corresponding to the family and params
    """
    return f"{task_family}_{hashify(params)}"   


def batch_tasks_together(tasks, skip_complete=False):
    """
    Return a list of tasks that can be batched together.

    :param tasks:
        A list of tasks, where some might be able to be batched together.
    """

    # Separate by family first.
    task_families = {}
    for i, task in enumerate(tasks):
        task_families.setdefault(task.task_family, [])
        task_families[task.task_family].append(i)
    
    batch_tasks = []
    for task_family, indices in task_families.items():
        batch_tasks.extend(
            _batch_tasks_from_same_family(
                [tasks[index] for index in indices if not (skip_complete and tasks[index].complete())]
            )
        )
    
    return batch_tasks


def _batch_tasks_from_same_family(tasks):
    if not tasks: 
        return tasks

    task = tasks[0]
    batch_param_names = task.batch_param_names()
    unbatched_param_names = set(task.param_kwargs).difference(task.batch_param_names())

    task_classes = {}
    batch_indices = {}
    unbatched_parameters = {}

    # Get set of tasks with identical unbatched param names.
    for i, task in enumerate(tasks):
        unbatched_parameters_ = { name: getattr(task, name) for name in unbatched_param_names }

        batch_id = hashify(unbatched_parameters_, -1)

        task_classes[batch_id] = task.__class__
        batch_indices.setdefault(batch_id, [])
        batch_indices[batch_id].append(i)
        unbatched_parameters[batch_id] = unbatched_parameters_
        

    # Construct the batch parameters.
    batch_parameters = {}
    for batch_id, indices in batch_indices.items():

        batch_parameters[batch_id] = {}
        for param_name in batch_param_names:
            batch_parameters[batch_id][param_name] = []
        
        for index in indices:
            task = tasks[index]
            for param_name in batch_param_names:
                value = getattr(task, param_name)
                if not task.is_batch_mode:
                    value = [value]
                elif isinstance(value, tuple):
                    value = list(value)
                
                batch_parameters[batch_id][param_name].extend(value)
                
    batch_tasks = []
    for batch_id, indices in batch_indices.items():
        task_class = task_classes[batch_id]

        kwds = unbatched_parameters[batch_id]
        bp = batch_parameters[batch_id]
        if len(indices) == 1:
            bp = { k: v[0] for k, v in bp.items() }
        kwds.update(bp)
    
        task = task_class(**kwds)
        batch_tasks.append(task)
    return batch_tasks

        