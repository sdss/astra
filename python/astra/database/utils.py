import json

from astra.database import astradb, session

def create_task_instance_in_database(task_id, parameters=None):
    """
    Create an entry in the database for a task instance with the given parameters.

    :param task_id:
        The task identifier.
    
    :param parameters: [optional]
        A dictionary of parameters to also include in the database.
    """

    parameters = (parameters or dict())

    # Get or create the parameter rows first.
    parameter_pks = []
    for name, value in parameters.items():

        if not isinstance(value, str):
            value = json.dumps(value)

        kwds = dict(
            parameter_name=name, 
            parameter_value=value
        )
        q = session.query(astradb.Parameter).filter_by(**kwds)
        instance = q.one_or_none()
        create = (instance is None)
        if create:
            instance = astradb.Parameter(**kwds)
            with session.begin():
                session.add(instance)
        
        parameter_pks.append(instance.pk)
    
    # Create task instance.
    ti = astradb.TaskInstance(task_id=task_id)
    with session.begin():
        session.add(ti)
    
    # Link the parameters.
    with session.begin():
        for parameter_pk in parameter_pks:
            session.add(astradb.TaskInstanceParameter(
                ti_pk=ti.pk,
                parameter_pk=parameter_pk
            ))

    session.flush()

    return ti
