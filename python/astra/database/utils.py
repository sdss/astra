import json

from astra.database import astradb, session
from typing import Dict
from sqlalchemy import (or_, and_, func, distinct)


deserialize_pks = lambda pks: json.loads(pks) if isinstance(pks, str) else pks


def parse_mjd(
        mjd
    ):
    """
    Parse Modified Julian Date, which might be in the form of an execution date
    from Apache Airflow (e.g., YYYY-MM-DD), or as a MJD integer. The order of
    checks here is:

        1. if it is not a string, just return the input
        2. if it is a string, try to parse the input as an integer
        3. if it is a string and cannot be parsed as an integer, parse it as
           a date time string

    :param mjd:
        the Modified Julian Date, in various possible forms.
    
    :returns:
        the parsed Modified Julian Date
    """
    if isinstance(mjd, str):
        try:
            mjd = int(mjd)
        except:
            return Time(mjd).mjd
    return mjd


def get_sdss4_apstar_kwds(limit=None, **kwargs):
    """
    Get identifying keywords for SDSS-IV APOGEE stars observed.
    """

    # We need; 'apstar', 'apred', 'obj', 'telescope', 'field', 'prefix'
    release, filetype = ("DR16", "apStar")
    columns = (
        catalogdb.SDSSDR16ApogeeStar.apogee_id.label("obj"),
        catalogdb.SDSSDR16ApogeeStar.field,
        catalogdb.SDSSDR16ApogeeStar.telescope,
        catalogdb.SDSSDR16ApogeeStar.apstar_version.label("apstar"),
        catalogdb.SDSSDR16ApogeeStar.file, # for prefix and apred
    )
    q = session.query(*columns).distinct(*columns)
    if kwargs:
        q = q.filter(**kwargs)

    if limit is not None:
        q = q.limit(limit)

    data_model_kwds = []
    for obj, field, telescope, apstar, filename in q.all():

        prefix = filename[:2]
        apred = filename.split("-")[1]

        data_model_kwds.append(dict(
            release=release,
            filetype=filetype,
            obj=obj,
            field=field,
            telescope=telescope,
            apstar=apstar,
            prefix=prefix,
            apred=apred
        ))

    return data_model_kwds
    

def get_sdss5_apstar_kwds(
        mjd, 
        min_ngoodrvs=1
    ):
    """
    Get identifying keywords for SDSS-V APOGEE stars observed on the given MJD.

    :param mjd:
        the Modified Julian Date of the observations
    
    :param min_ngoodrvs: [optional]
        the minimum number of good radial velocity measurements (default: 1)
    
    :returns:
        a list of dictionaries containing the identifying keywords for SDSS-V
        APOGEE stars observed on the given MJD, including the `release` and
        `filetype` keys necessary to identify the path
    """
    mjd = parse_mjd(mjd)

    # TODO: Consider switching to 'filetype' instead of 'filetype' to be
    #       consistent with SDSSPath.full() argument names.
    release, filetype = ("sdss5", "apStar")
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

    kwds = []
    for values in rows:
        d = dict(zip(keys, values))
        d.update(
            release=release,
            filetype=filetype,
            apstar="stars", # TODO: Raise with Nidever
        )
        kwds.append(d)
        
    return kwds


def get_sdss5_apvisit_kwds(
        mjd
    ):
    """
    Get identifying keywords for SDSS-V APOGEE visits taken on the given MJD.

    :param mjd:
        The Modified Julian Date of the observations.
    
    :returns:
        a list of dictionaries containing the identifying keywords for SDSS-V
        APOGEE visits observed on the given MJD, including the `release` and
        `filetype` keys necessary to identify the path
    """

    mjd = parse_mjd(mjd)
    release, filetype = ("sdss5", "apVisit")
    columns = (
        apogee_drpdb.Visit.fiberid.label("fiber"), # TODO: Raise with Nidever
        apogee_drpdb.Visit.plate,
        apogee_drpdb.Visit.field,
        apogee_drpdb.Visit.mjd,
        apogee_drpdb.Visit.apred_vers.label("apred"), # TODO: Raise with Nidever
        apogee_drpdb.Visit.file
    )
    q = session.query(*columns).distinct(*columns).q.filter(apogee_drpdb.Visit.mjd == mjd)

    kwds = []
    for fiber, plate, field, mjd, apred, filename in q.all():
        kwds.append(dict(
            release=release,
            filetype=filetype,
            fiber=fiber,
            plate=plate,
            field=field,
            mjd=mjd,
            apred=apred,
            prefix=filename[:2]
        ))

    return kwds
    

def create_task_instances_for_sdss5_apvisits(
        dag_id,
        task_id,
        mjd,
        full_output=False,
        **parameters
    ):
    """
    Create task instances for SDSS5 APOGEE visits taken on a Modified Julian Date,
    with the given identifiers for the directed acyclic graph and the task.

    :param dag_id:
        the identifier string of the directed acyclic graph

    :param task_id:
        the task identifier

    :param mjd:
        the Modified Julian Date of the observations
    
    :param \**parameters: [optional]
        additional parameters to be assigned to the task instances
    """

    all_kwds = get_sdss5_apvisit_kwds(mjd)

    instances = []
    for kwds in all_kwds:
        instances.append(
            get_or_create_task_instance(
                dag_id,
                task_id,
                **kwds,
                **parameters
        ))

    if full_output:
        return instances
    return [instance.pk for instance in instances]


def create_task_instances_for_sdss5_apstars(
        dag_id,
        task_id,
        mjd,
        full_output=False,
        **parameters
    ):
    """
    Create task instances for SDSS5 APOGEE stars taken on a Modified Julian Date,
    with the given identifiers for the directed acyclic graph and the task.

    :param dag_id:
        the identifier string of the directed acyclic graph

    :param task_id:
        the task identifier

    :param mjd:
        the Modified Julian Date of the observations
    
    :param \**parameters: [optional]
        additional parameters to be assigned to the task instances
    """

    all_kwds = get_sdss5_apstar_kwds(mjd)

    instances = []
    for kwds in all_kwds:
        instances.append(
            get_or_create_task_instance(
                dag_id,
                task_id,
                **kwds,
                **parameters
        ))

    if full_output:
        return instances
    return [instance.pk for instance in instances]


def create_task_instances_for_sdss5_apstars_from_apvisits(
        dag_id,
        task_id,
        apvisit_pks,
        full_output=False,
        **parameters
    ):
    """
    Create task instances for SDSS-V ApStar objects, given some primary keys for task instances 
    that reference SDSS-V ApVisit objects.

    :param dag_id:
        the identifier string of the directed acyclic graph

    :param task_id:
        the task identifier

    :param apvisit_pks:
        primary keys of task instances that refer to SDSS-V ApVisit objects
    
    :param full_output: [optional]
        If true, return the instances created. Otherwise just return the primary keys of the instances.
    
    :param \**parameters: [optional]
        additional parameters to be assigned to the task instances
    """

    # Get the unique stars from the primary keys.
    q = session.join(astradb.TaskInstance)\
               .filter(astradb.TaskInstance.pk._in(deserialize_pks(apvisit_pks)))

    # Match stars to visits by:
    keys = ("telescope", "obj", "apred")
    
    star_keywords = set([[ti.parameters[k] for k in keys] for ti in q.all()])

    # Match these to the apogee_drp.Star table.
    columns = (
        apogee_drpdb.Star.apred_vers.label("apred"), # TODO: Raise with Nidever
        apogee_drpdb.Star.healpix,
        apogee_drpdb.Star.telescope,
        apogee_drpdb.Star.apogee_id.label("obj"), # TODO: Raise with Nidever
    )
    common_kwds = dict(apstar="stars") # TODO: Raise with Nidever

    instances = []
    for telescope, obj, apred in star_keywords:
        q = session.query(apogee_drpdb.Star.healpix)\
                   .distinct(apogee_drpdb.Star.healpix)\
                   .filter(
                       apogee_drpdb.Star.apred_vers == apred,
                       apogee_drpdb.Star.telescope == telescope,
                       apogee_drpdb.Star.apogee_id == obj
                    )
        r = q.one_or_none()
        if r is None: 
            continue
        healpix, = r

        kwds = dict(
            apstar="stars", # TODO: Raise with Nidever
            release="sdss5",
            filetype="apStar",
            healpix=healpix,
            apred=apred,
            telescope=telescope,
            obj=obj,
        )

        instances.append(
            get_or_create_task_instance(
                dag_id,
                task_id,
                **kwds,
                **parameters
            )
        )

    if full_output:
        return instances
    return [instance.pk for instance in instances]



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


def get_or_create_parameter_pk(
        name, 
        value
    ):
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


def create_task_instance(
        dag_id, 
        task_id, 
        parameters=None
    ):
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


def get_or_create_task_instance(
        dag_id, 
        task_id, 
        parameters=None
    ):
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


def create_task_output(
        task_instance_or_pk, 
        model, 
        **kwargs
    ):
    """
    Create a new entry in the database for the output of a task.

    :param task_instance_or_pk:
        the task instance (or its primary key) to reference this output to
        
    :param model:
        the database model to store the result (e.g., `astra.database.astradb.Ferre`)
    
    :param \**kwargs:
        the keyword arguments that will be stored in the database
    
    :returns:
        A two-length tuple containing the task instance, and the output instance
    """

    # Get the task instance.
    if not isinstance(task_instance_or_pk, astradb.TaskInstance):
        task_instance = session.query(astradb.TaskInstance)\
                            .filter(astradb.TaskInstance.pk == task_instance_pk)\
                            .one_or_none()
                            
        if task_instance is None:
            raise ValueError(f"no task instance found matching primary key {task_instance_pk}")
    else:
        task_instance = task_instance_or_pk

    # Create a new output interface entry.
    with session.begin():
        output = astradb.OutputInterface()
        session.add(output)

    assert output.pk is not None

    kwds = dict(output_pk=output.pk)
    kwds.update(kwargs)

    # Create the instance of the result.
    instance = model(**kwds)
    with session.begin():
        session.add(instance)
    
    # Reference the output to the task instance.
    task_instance.output_pk = output.pk

    session.flush()
    
    return (task_instance, instance)