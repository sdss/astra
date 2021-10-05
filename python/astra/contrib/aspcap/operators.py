import numpy as np
from airflow.exceptions import AirflowSkipException
from astropy.io.fits import getheader
from sdss_access import SDSSPath

from astra.database import astradb, session
from astra.database.utils import (create_task_instance, create_task_output)
from astra.contrib.ferre.utils import safe_read_header
from astra.operators import ApStarOperator
from astra.utils import flatten, log

from astra.contrib.aspcap import (bitmask, utils)


def branch(task_id_callable, task, ti, **kwargs):
    """
    A function to branch specific downstream tasks, given the primary keys
    returned by the upstream tasks.
    
    :param task_id_callable:
        A Python callable that takes in as input the `header_path` and
        returns a task ID.
    
    :param task:    
        The task being executed. This is supplied by the DAG context.
    
    :param ti:
        The task instance. This is supplied by the DAG context.
    
    :returns:
        A list of task IDs that should execute next.
    """

    # Get primary keys from upstream tasks.
    pks = []
    for upstream_task in task.upstream_list:
        pks.append(ti.xcom_pull(task_ids=upstream_task.task_id))

    pks = flatten(pks)
    log.debug(f"Upstream primary keys: {pks}")
    log.debug(f"Downstream task IDs: {task.downstream_list}")

    # Get unique header paths for the primary keys given.
    # TODO: This query could fail if the number of primary keys provided
    #       is yuuge. May consider changing this query.
    q = session.query(astradb.TaskInstanceParameter.ti_pk, astradb.Parameter.parameter_value)\
               .join(astradb.TaskInstanceParameter, 
                     astradb.TaskInstanceParameter.parameter_pk == astradb.Parameter.pk)\
               .filter(astradb.Parameter.parameter_name == "header_path")\
               .filter(astradb.TaskInstanceParameter.ti_pk.in_(pks))    

    log.debug(f"Found:")
    downstream_task_ids = []
    for pk, header_path in q.all():
        log.debug(f"\t{pk}: {header_path}")

        telescope, lsf, spectral_type_desc = utils.task_id_parts(header_path)
        if telescope is None and lsf is None:
            # Special hack for BA grids, where telescope/lsf information cannot be found from header path.
            # TODO: Consider removing this hack entirely. This could be fixed by symbolicly linking the BA grids to locations
            #       for each telescope/fibre combination.

            instance = session.query(astradb.TaskInstance)\
                              .filter(astradb.TaskInstance.pk == pk).one_or_none()
            
            tree = SDSSPath(release=instance.parameters["release"])
            path = tree.full(**instance.parameters)

            header = getheader(path)
            downstream_task_ids.append(
                task_id_callable(
                    header_path,
                    # TODO: This is matching the telescope styling in utils.task_id_parts, but these should have a common place.
                    telescope=instance.parameters["telescope"].upper()[:3],
                    lsf=utils.get_lsf_grid_name(header["MEANFIB"])
                )
            )
        else:
            downstream_task_ids.append(task_id_callable(header_path))
        log.debug(f"\t\tadded {downstream_task_ids[-1]}")


    downstream_task_ids = sorted(set(downstream_task_ids))

    log.debug(f"Downstream tasks to execute:")
    for task_id in downstream_task_ids:
        log.debug(f"\t{task_id}")

    return downstream_task_ids


def get_best_result(task, ti, **kwargs):
    """
    When there are numerous FERRE tasks that are upstream, this
    function will return the primary keys of the task instances that gave
    the best result on a per-observation basis.
    """

    # Get the PKs from upstream.
    pks = []
    log.debug(f"Upstream tasks: {task.upstream_list}")
    for upstream_task in task.upstream_list:
        pks.append(ti.xcom_pull(task_ids=upstream_task.task_id))

    pks = flatten(pks)
    log.debug(f"Getting best initial guess among primary keys {pks}")

    # Need to uniquely identify observations.
    param_bit_mask = bitmask.ParamBitMask()
    bad_grid_edge = (param_bit_mask.get_value("GRIDEDGE_WARN") | param_bit_mask.get_value("GRIDEDGE_BAD"))

    trees = {}
    best_tasks = {}
    for i, pk in enumerate(pks):
        q = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk==pk)
        instance = q.one_or_none()

        if instance.output is None:
            log.warning(f"No output found for task instance {instance}")
            continue

        p = instance.parameters

        # Check that the telescope is the same as what we expect from this task ID.
        # This is a bit of a hack. Let us explain.

        # The "BA" grid does not have a telescope/fiber model, so you can run LCO and APO
        # data through the initial-BA grid. And those outputs go to the "get_best_results"
        # for each of the APO and LCO tasks (e.g., this function).
        # If there is only APO data, then the LCO "get_best_result" will only have one
        # input: the BA results. Then it will erroneously think that's the best result
        # for that source.

        # It's hacky to put this logic in here. It should be in the DAG instead. Same
        # thing for parsing 'telescope' name in the DAG (eg 'APO') from 'apo25m'.
        this_telescope_short_name = p["telescope"][:3].upper()
        expected_telescope_short_name = task.task_id.split(".")[1]
        log.info(f"For instance {instance} we have {this_telescope_short_name} and {expected_telescope_short_name}")
        if this_telescope_short_name != expected_telescope_short_name:
            continue

        try:
            tree = trees[p["release"]]                
        except KeyError:
            tree = trees[p["release"]] = SDSSPath(release=p["release"])
        
        key = "_".join([
            p['release'],
            p['filetype'],
            *[p[k] for k in tree.lookup_keys(p['filetype'])]
        ])
        
        best_tasks.setdefault(key, (np.inf, None))
        
        # TODO: Confirm that this is base10 log. This should also be 'log_reduced_chisq_fit',
        #       according to the documentation.
        log_chisq_fit, *_ = instance.output.log_chisq_fit
        previous_teff, *_ = instance.output.teff
        bitmask_flag, *_ = instance.output.bitmask_flag
        
        log.debug(f"Result {instance} {instance.output} with log_chisq_fit = {log_chisq_fit} and {previous_teff} and {bitmask_flag}")
        
        # Note: If FERRE totally fails then it will assign -999 values to the log_chisq_fit. So we have to
        #       check that the log_chisq_fit is actually sensible!
        #       (Or we should only query task instances where the output is sensible!)
        if log_chisq_fit < 0: # TODO: This is a fucking hack.
            log.debug(f"Skipping result for {instance} {instance.output} as log_chisq_fit = {log_chisq_fit}")
            continue
            
        parsed_header = utils.parse_header_path(p["header_path"])
        
        # Penalise chi-sq in the same way they did for DR17.
        # See github.com/sdss/apogee/python/apogee/aspcap/aspcap.py#L658
        if parsed_header["spectral_type"] == "GK" and previous_teff < 3900:
            log.debug(f"Increasing \chisq because spectral type GK")
            log_chisq_fit += np.log10(10)

        bitmask_flag_logg, bitmask_flag_teff = bitmask_flag[-2:]
        if bitmask_flag_logg & bad_grid_edge:
            log.debug(f"Increasing \chisq because logg flag is bad edge")
            log_chisq_fit += np.log10(5)
            
        if bitmask_flag_teff & bad_grid_edge:
            log.debug(f"Increasing \chisq because teff flag is bad edge")
            log_chisq_fit += np.log10(5)
        
        # Is this the best so far?
        if log_chisq_fit < best_tasks[key][0]:
            log.debug(f"Assigning this output to best task as {log_chisq_fit} < {best_tasks[key][0]}: {pk}")
            best_tasks[key] = (log_chisq_fit, pk)
    
    for key, (log_chisq_fit, pk) in best_tasks.items():
        if pk is None:
            log.warning(f"No good task found for key {key}: ({log_chisq_fit}, {pk})")
        else:
            log.info(f"Best task for key {key} with \chi^2 of {log_chisq_fit:.2f} is primary key {pk}")

    if best_tasks:
        return [pk for (log_chisq_fit, pk) in best_tasks.values() if pk is not None]
    else:
        raise AirflowSkipException(f"no task outputs found from {len(pks)} primary keys")



class AspcapOperator(ApStarOperator):
    """
    Get initial estimates of stellar parameters for all new ApStar observations, and distribute
    task instances across FERRE grids given the header paths supplied.
    """

    def __init__(
        self,
        header_paths,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.header_paths = header_paths


    def common_task_parameters(self, **kwargs):
        kwds = kwargs.copy()
        kwds["ignore"] = (
            "self", "parameters", "args", "kwargs", "slurm_kwargs",
            # Exclude header_paths from the parameters that are given to each task.
            "header_paths", 
            # We will not accept any spectrum callbacks for this operator!
            "spectrum_callback", "spectrum_callback_kwargs"
        )
        return super(AspcapOperator, self).common_task_parameters(**kwds)


    def pre_execute(self, context):
        """ Overriding default behaviour for pre_execute() functionality. """
        return None


    def execute(self, context):
        """
        Create task instances for all the data model identifiers, which could include
        multiple task instances for each data model identifier set.

        :param context:
            The Airflow context dictionary.
        """

        # Get header information.
        grid_info = utils.parse_grid_information(self.header_paths)

        args = (context["dag"].dag_id, context["task"].task_id, context["run_id"])

        # Get parameters from the parent class initialisation that should also be stored.
        common_task_parameters = self.common_task_parameters()

        pks = []
        trees = {}
        
        for data_model_identifiers in self.data_model_identifiers(context):

            parameters = { **common_task_parameters, **data_model_identifiers }

            release = parameters["release"]
            tree = trees.get(release, None)
            if tree is None:
                trees[release] = tree = SDSSPath(release=release)

            path = tree.full(**parameters)
            
            # Generate initial guess(es).
            initial_guesses = []

            # From headers
            try:
                header = getheader(path)

                teff = safe_read_header(header, ("RV_TEFF", "RVTEFF"))
                logg = safe_read_header(header, ("RV_LOGG", "RVLOGG"))
                fe_h = safe_read_header(header, ("RV_FEH", "RVFEH"))

                # Get information relevant for matching initial guess and grids.
                initial_guesses.append(dict(
                    telescope=parameters["telescope"], # important for LSF information
                    mean_fiber=header["MEANFIB"], # important for LSF information
                    teff=teff,
                    logg=logg,
                    metals=fe_h,
                ))

            except:
                log.exception(f"Unable to load relevant headers from path {path}")
                continue
            
            # Add any other initial guesses? From Gaia? etc?
            for initial_guess in initial_guesses:
                for header_path, _ in utils.yield_suitable_grids(grid_info, **initial_guess):
                    parameters.update(
                        header_path=header_path,
                        initial_teff=np.round(initial_guess["teff"], 0),
                        initial_logg=np.round(initial_guess["logg"], 3),
                        initial_metals=np.round(initial_guess["metals"], 3),
                        initial_log10vdop=np.round(utils.approximate_log10_microturbulence(initial_guess["logg"]), 3),
                        initial_o_mg_si_s_ca_ti=0.0,
                        initial_lgvsini=1.0,  # :eyes:
                        initial_c=0.0,
                        initial_n=0.0,
                    )
                    instance = create_task_instance(*args, parameters)
                    pks.append(instance.pk)
                    
                    log.debug(f"Created {instance} with parameters {parameters}")

        if not pks:
            raise AirflowSkipException("No data model identifiers found for this time period.")

        return pks
    

def write_database_outputs(
        task, 
        ti, 
        run_id, 
        element_from_task_id_callable=None,
        **kwargs
    ):
    """
    Collate outputs from upstream FERRE executions and write them to an ASPCAP database table.
    
    :param task:
        This task, as given by the Airflow context dictionary.
    
    :param ti:
        This task instance, as given by the Airflow context dictionary.
    
    :param run_id:
        This run ID, as given by the Airflow context dictionary.
    
    :param element_from_task_id_callable: [optional]
        A Python callable that returns the chemical element, given a task ID.
    """

    
    log.debug(f"Writing ASPCAP database outputs")

    pks = []
    for upstream_task in task.upstream_list:
        pks.append(ti.xcom_pull(task_ids=upstream_task.task_id))

    log.debug(f"Upstream primary keys: {pks}")

    # Group them together by source.
    instance_pks = []
    for source_pks in list(zip(*pks)):

        # The one with the lowest primary key will be the stellar parameters.
        sp_pk, *abundance_pks = sorted(source_pks)
        
        sp_instance = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk == sp_pk).one_or_none()
        abundance_instances = session.query(astradb.TaskInstance).filter(astradb.TaskInstance.pk.in_(abundance_pks)).all()

        # Get parameters that are in common to all instances.
        keep = {}
        for key, value in sp_instance.parameters.items():
            for instance in abundance_instances:
                if instance.parameters[key] != value:
                    break
            else:
                keep[key] = value

        # Create a task instance.
        instance = create_task_instance(
            dag_id=task.dag_id, 
            task_id=task.task_id, 
            run_id=run_id,
            parameters=keep
        )

        # Create a partial results table.
        keys = ["snr"]
        label_names = ("teff", "logg", "metals", "log10vdop", "o_mg_si_s_ca_ti", "lgvsini", "c", "n")
        for key in label_names:
            keys.extend([key, f"u_{key}"])
        
        results = dict([(key, getattr(sp_instance.output, key)) for key in keys])

        # Now update with elemental abundance instances.
        for el_instance in abundance_instances:
            
            if element_from_task_id_callable is not None:
                element = element_from_task_id_callable(el_instance.task_id).lower()
            else:
                element = el_instance.task_id.split(".")[-1].lower()
            
            # Check what is not frozen.
            thawed_label_names = []
            ignore = ("lgvsini", ) # Ignore situations where lgvsini was missing from grid and it screws up the task
            for key in label_names:
                if key not in ignore and not getattr(el_instance.output, f"frozen_{key}"):
                    thawed_label_names.append(key)

            if len(thawed_label_names) > 1:
                log.warning(f"Multiple thawed label names for {element} {el_instance}: {thawed_label_names}")

            values = np.hstack([getattr(el_instance.output, ln) for ln in thawed_label_names]).tolist()
            u_values = np.hstack([getattr(el_instance.output, f"u_{ln}") for ln in thawed_label_names]).tolist()

            results.update({
                f"{element}_h": values,
                f"u_{element}_h": u_values,
            })

        # Include associated primary keys so we can reference back to original parameters, etc.
        results["associated_ti_pks"] = [sp_pk, *abundance_pks]

        log.debug(f"Results entry: {results}")

        # Create an entry in the output interface table.
        # (We will update this later with any elemental abundance results).
        # TODO: Should we link back to the original FERRE primary keys?
        output = create_task_output(
            instance,
            astradb.Aspcap,
            **results
        )
        log.debug(f"Created output {output} for instance {instance}")
        instance_pks.append(instance.pk)
        
    return instance_pks