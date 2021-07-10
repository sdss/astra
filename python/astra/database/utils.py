import json

from astra.database import astradb, session
from typing import Dict
from sqlalchemy import (or_, and_, func, distinct)



def get_sdss5_apstar_kwds(mjd, min_ngoodrvs=1):
    """
    Get identifying keywords for SDSS-V APOGEE stars observed on the given MJD.

    :param mjd:
        The Modified Julian Date of the observations.
    
    :param min_ngoodrvs: [optional]
        The minimum number of good radial velocity measurements (default: 1).
    
    :returns:
        A three length tuple containing the data release name, the data model name,
        and a list of dictionaries containing the identifying keywords for SDSS-V
        APOGEE stars observed on the given MJD.
    """
    release, data_model_name = ("sdss5", "apStar")
    columns = (
        apogee_drpdb.Star.apred_vers.label("apred"), # TODO: Raise with Nidever
        apogee_drpdb.Star.healpix,
        apogee_drpdb.Star.telescope,
        apogee_drpdb.Star.apogee_id.label("obj"), # TODO: Raise with Nidever
    )

    q = session.query(*columns).distinct(*columns)
    q = q.filter(apogee_drpdb.Star.mjdend == mjd)\
         .filter(apogee_drpdb.Star.ngoodrvs >= min_ngoodrvs)

    rows = q.all()
    keys = [column.name for column in columns]

    data_model_kwds = []
    for values in rows:
        d = dict(zip(keys, values))
        d.update(apstar="stars") # TODO: Raise with Nidever
        data_model_kwds.append(d)
        
    return (release, data_model_name, data_model_kwds)


def get_sdss5_apvisit_kwds(mjd):
    """
    Get identifying keywords for SDSS-V APOGEE visits taken on the given MJD.

    :param mjd:
        The Modified Julian Date of the observations.
    
    :returns:
        A three length tuple containing the data release name, the data model name,
        and a list of dictionaries containing the identifying keywords for SDSS-V
        APOGEE visits taken on the given MJD.
    """

    release, data_model_name = ("sdss5", "apVisit")
    columns = (
        apogee_drpdb.Visit.fiberid.label("fiber"), # TODO: Raise with Nidever
        apogee_drpdb.Visit.plate,
        apogee_drpdb.Visit.field,
        apogee_drpdb.Visit.mjd,
        apogee_drpdb.Visit.apred_vers.label("apred"), # TODO: Raise with Nidever
        apogee_drpdb.Visit.file
    )
    q = session.query(*columns).distinct(*columns).q.filter(apogee_drpdb.Visit.mjd == mjd)

    data_model_kwds = []
    for fiber, plate, field, mjd, apred, filename in q.all():
        data_model_kwds.append(dict(
            fiber=fiber,
            plate=plate,
            field=field,
            mjd=mjd,
            apred=apred,
            prefix=filename[:2]
        ))

    return (release, data_model_name, data_model_kwds)


def create_task_instances_for_sdss5_apstars(dag_id, task_id, mjd, **kwargs):
    raise a




def get_task_instance(
        dag_id: str, 
        task_id: str, 
        parameters: Dict,
    ):
    """
    Get a task instance exactly matching the given DAG and task identifiers, and the given parameters.

    :param dag_id:
        The identifier of the directed acyclic graph (DAG).
    
    :param task_id:
        The identifier of the task.
    
    :param parameters:
        The parameters of the task, as a dictionary
    """

    # TODO: Profile this and consider whether it should be used.
    if False:
        # Quick check for things matching dag_id or task_id, which is cheaper than checking all parameters.
        q_ti = session.query(astradb.TaskInstance).filter(
            astradb.TaskInstance.dag_id == dag_id,
            astradb.TaskInstance.task_id == task_id
        )
        if q_ti.count() == 0:
            return None

    # Get primary keys of the individual parameters, and then check by task.
    q_p = session.query(astradb.Parameter.pk).filter(
        or_(*(and_(astradb.Parameter.parameter_name == k, astradb.Parameter.parameter_value == v) for k, v in parameters.items()))
    )
    N_p = q_p.count()
    if N_p < len(parameters):
        return None
    
    # Perform subquery to get primary keys of task instances that have all of these parameters.
    sq = session.query(astradb.TaskInstanceParameter.ti_pk)\
                .filter(astradb.TaskInstanceParameter.parameter_pk.in_(pk for pk, in q_p.all()))\
                .group_by(astradb.TaskInstanceParameter.ti_pk)\
                .having(func.count(distinct(astradb.TaskInstanceParameter.parameter_pk)) == N_p).subquery()
    
    # Query for task instances that match the subquery and match our additional constraints.
    q = session.query(astradb.TaskInstance).join(
        sq,
        astradb.TaskInstance.pk == sq.c.ti_pk
    )
    if dag_id is not None:
        q = q.filter(astradb.TaskInstance.dag_id == dag_id)
    if task_id is not None:
        q = q.filter(astradb.TaskInstance.task_id == task_id)

    return q.one_or_none()


def get_or_create_parameter_pk(name, value):
    """
    Get or create the primary key for a parameter key/value pair in the database.

    :param name:
        the name of the parameter
    
    :param value:
        the value of the parameter, serialized or not
    
    :returns:
        A two-length tuple containing the integer of the primary key, and a boolean
        indicating whether the entry in the database was created by this function call.
    """

    if not isinstance(value, str):
        value = json.dumps(value)

    kwds = dict(parameter_name=parameter_name, parameter_value=value)
    q = session.query(astradb.Parameter).filter_by(**kwds)
    instance = q.one_or_none()
    create = (instance is None)
    if create:
        instance = astradb.Parameter(**kwds)
        with session.begin():
            session.add(instance)
    
    return (instance.pk, create)



def create_task_instance(dag_id, task_id, parameters=None):
    """
    Create a task instance in the database with the given identifiers and parameters.

    :param dag_id:
        the identifier string of the directed acyclic graph

    :param task_id:
        the task identifier
    
    :param parameters: [optional]
        a dictionary of parameters to also include in the database
    """

    parameters = (parameters or dict())

    # Get or create the parameter rows first.
    parameter_pks = (pk for pk, created in (get_or_create_parameter_pk(k, v) for k, v in parameters.items()))
    
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


def get_or_create_task_instance(dag_id, task_id, parameters=None):
    """
    Get or create a task instance given the identifiers of the directed acyclic graph, the task,
    and the parameters of the task instance.

    :param dag_id:
        the identifier string of the directed acyclic graph

    :param task_id:
        the task identifier
    
    :param parameters: [optional]
        a dictionary of parameters to also include in the database
    """
    
    parameters = (parameters or dict())

    instance = get_task_instance(dag_id, task_id, parameters)
    if instance is None:
        return create_task_instance(dag_id, task_id, parameters)
