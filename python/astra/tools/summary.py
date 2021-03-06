import numpy as np
from astra.tasks.targets import DatabaseTarget
from luigi.task import flatten
from astropy.table import Table


def _get_database_output(outputs):  
    database_outputs = []
    for output in outputs:
        if isinstance(output, DatabaseTarget):
            database_outputs.append(output)

    if not database_outputs:
        raise ValueError("could not find database target")
    elif len(database_outputs) > 1:
        raise ValueError(f"too many output database targets: {database_outputs}")
    
    output, = database_outputs
    return output


class DummyTask(object):

    task_id = None
    is_batch_mode = False
    healpix = 0

    def __init__(self, task_factory):
        self.task_family = task_factory.task_family
        self.connection_string = task_factory.connection_string.task_value(task_factory, "connection_string")
        
    def get_params(self):
        return []
    
    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError:
            return "UNKNOWN"


def get_database_output(task):
    """
    Return the database target output class for the given task, or task class.

    :param task:
        A single task or a task class.
    """

    # Check to see whether we have a task, or a task class.
    try:
        task.task_id

    except AttributeError:
        # Task class.

        # Create a dummy task that we will use to resolve the outputs.
        dummy = DummyTask(task)
        
        output = _get_database_output(flatten(task.output(dummy)))

    else:
        # Task.
        output = _get_database_output(flatten(task.output()))
        # This output class has a task_id, and we want one without so we can select all rows etc.
        output.task_id = None

    return output


def create_summary_table(task):
    """
    Return a table of results generated by the given task and stored in a database.

    :param task:
        Either a single task, or a task class, which has a `DatabaseTarget` output.

    :returns:
        An `astropy.table.Table` of database results generated by this task.
    """
    targets = get_database_output(task)
    rows = targets.read(limit=None)

    # Handle arrays nicely.
    names = targets.column_names
    array_columns = [
        (i, name) for i, (name, value) in enumerate(zip(names, rows[0])) if isinstance(value, list)
    ]
    
    if any(array_columns):
        lengths = np.max(np.array([[len(row[i]) for i, name in array_columns] for row in rows]), axis=0)
        array_indices = [i for i, name in array_columns]

        N = len(rows)
        data = { name: [] for name in names }
        for (i, name), L in zip(array_columns, lengths):
            data[name] = np.nan * np.ones((N, L))

        raise a    
        for j, row in enumerate(rows):
            for i, name in enumerate(names):
                if i in array_indices:
                    data[name][j, :len(row[i])] = row[i]
                else:
                    data[name].append(row[i])

        return Table(data=data)
    
    else:
        return Table(rows=rows, names=targets.column_names)

