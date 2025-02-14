#!/usr/bin/env python3
import typer
import os
from typing import List, Optional
from typing_extensions import Annotated
from enum import Enum

app = typer.Typer()

in_airflow_context = os.environ.get("AIRFLOW_CTX_TASK_ID", None) is not None

class Product(str, Enum):
    mwmTargets = "mwmTargets"
    mwmAllStar = "mwmAllStar"
    mwmAllVisit = "mwmAllVisit"
    astraAllStarASPCAP = "astraAllStarASPCAP"
    astraAllStarAPOGEENet = "astraAllStarAPOGEENet"
    astraAllVisitAPOGEENet = "astraAllVisitAPOGEENet"

@app.command()
def version():
    """Print the version of Astra."""
    from astra import __version__
    typer.echo(f"Astra version: {__version__}")    

@app.command()
def create(
    products: Annotated[List[Product], typer.Argument(help="The product name(s) to create.")],
    overwrite: Annotated[bool, typer.Option(help="Overwrite the product if it already exists.")] = False,
    limit: Annotated[int, typer.Option(help="Limit the number of rows per product.", min=1)] = None,
):    
    """Create an Astra summary product."""    
    from astra.products.mwm_summary import (
        create_mwm_targets_product,
        create_mwm_all_star_product,
        create_mwm_all_visit_product
    )
    from astra.products.pipeline_summary import create_all_star_product, create_all_visit_product
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar
    mapping = (
        {
            Product.mwmTargets: (create_mwm_targets_product, {}),
            Product.mwmAllVisit: (create_mwm_all_visit_product, {}),
            Product.mwmAllStar: (create_mwm_all_star_product, {}),
            Product.astraAllStarASPCAP: (
                create_all_star_product, 
                {
                    "pipeline_model": "aspcap.ASPCAP",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar
                }
            ),
            Product.astraAllStarAPOGEENet: (
                create_all_star_product,
                {
                    "pipeline_model": "apogeenet.ApogeeNet",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar
                }
            ),
            Product.astraAllVisitAPOGEENet: (
                create_all_visit_product,
                {
                    "pipeline_model": "apogeenet.ApogeeNet",
                    "apogee_spectrum_model": ApogeeVisitSpectrumInApStar
                }
            )
        }
    )    

    for product in products:
        fun, kwargs = mapping[product]
        path = fun(overwrite=overwrite, limit=limit, **kwargs)
        typer.echo(f"Created {product}: {path}")


@app.command()
def srun(
    task: Annotated[str, typer.Argument(help="The task name to run (e.g., `aspcap`, or `astra.pipelines.aspcap.aspcap`).")],
    model: Annotated[str, typer.Argument(
        help=(
            "The input model to use (e.g., `ApogeeCombinedSpectrum`, `BossCombinedSpectrum`). "
        )
        )] = None,
    nodes: Annotated[int, typer.Option(help="The number of nodes to use.", min=1)] = 1,
    procs: Annotated[int, typer.Option(help="The number of `astra` processes to use per node.", min=1)] = 1,
    limit: Annotated[int, typer.Option(help="Limit the number of inputs.", min=1)] = None,
    account: Annotated[str, typer.Option(help="Slurm account")] = "sdss-np",
    partition: Annotated[str, typer.Option(help="Slurm partition")] = None,
    gres: Annotated[str, typer.Option(help="Slurm generic resources")] = None,
    mem: Annotated[str, typer.Option(help="Memory per node")] = None,
    time: Annotated[str, typer.Option(help="Wall-time")] = "24:00:00",
):
    """Distribute an Astra task over many nodes using Slurm."""

    partition = partition or account

    import os
    import sys
    import json
    import numpy as np
    import concurrent.futures
    import subprocess
    import pickle
    from datetime import datetime
    from tempfile import TemporaryDirectory, mkdtemp
    from peewee import JOIN
    from importlib import import_module
    from astra import models, __version__, generate_queries_for_task
    from astra.utils import silenced, expand_path, log, resolve_task, accepts_live_renderable
    from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.logging import RichHandler
    from rich.console import Console
    from logging import FileHandler

    model, q = next(generate_queries_for_task(task, model, limit))

    total = q.count()
    if total == 0:
        log.info(f"No {model.__name__} spectra to process.")
        sys.exit(0)

    workers = nodes * procs
    limit = int(np.ceil(total / workers))
    today = datetime.now().strftime("%Y-%m-%d")

    # Re-direct log handler
    live_renderable = Table.grid()
    console = Console()
    if not accepts_live_renderable(resolve_task(task)):
        overall_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        live_renderable.add_row(Panel(overall_progress, title=task))
    
    with Live(live_renderable, console=console, redirect_stdout=False, redirect_stderr=False) as live:
        log.handlers.clear()
        log.handlers.extend([
            RichHandler(console=live.console, markup=True, rich_tracebacks=True),
        ])

        #with Progress(
        #    SpinnerColumn(),
        #    TextColumn("[progress.description]{task.description}"),
        #    BarColumn(),
        #    transient=not in_airflow_context
        #) as p:
        futures = []
        with concurrent.futures.ProcessPoolExecutor(nodes) as executor:
            # Load a whole bunch of sruns in processes
            td = mkdtemp(dir=expand_path("$PBS"), prefix=f"{task}-{today}-")    
            log.info(f"Working directory: {td}")

            status_path_locks = {}
            items_in_row = []
            for n in range(nodes):
                job_name = f"{task}" + (f"-{n}" if nodes > 1 else "")

                progress = Progress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn()
                )
                items_in_row.append(Panel.fit(progress, title=job_name, padding=(0, 2, 0, 2)))

                if len(items_in_row) == 2 or n == (nodes - 1):
                    live_renderable.add_row(*items_in_row)
                    items_in_row = []
                status_path_locks[progress] = {}

                # TODO: Let's not hard code this here.
                commands = ["export CLUSTER=1"]
                for page in range(n * procs, (n + 1) * procs):
                    status_path = f"{td}/live-{n}-{page}"
                    status_path_locks[progress][status_path] = 0
                    commands.append(f"astra run {task} {model.__name__} --limit {limit} --page {page + 1} --live-renderable-path {status_path} &")
                commands.append("wait")

                script_path = f"{td}/node_{n}.sh"
                with open(script_path, "w") as fp:
                    fp.write("\n".join(commands))

                os.system(f"chmod +x {script_path}")
                executable = [
                    "srun",
                    "--nodes=1",
                    f"--partition={partition}",
                    f"--account={account}",
                    f"--job-name={job_name}",
                    f"--time={time}",
                    f"--output={td}/{n}.out",
                    f"--error={td}/{n}.err",
                ]
                if mem is not None:
                    executable.append(f"--mem={mem}")
                if gres is not None:
                    executable.append(f"--gres={gres}")
                
                executable.extend(["bash", "-c", f"{script_path}"])

                futures.append(
                    executor.submit(
                        subprocess.run,
                        executable,
                        capture_output=True
                    )
                )
                    
            max_returncode, mappings = (0, {})
            while len(futures):
                try:
                    future = next(concurrent.futures.as_completed(futures, timeout=1))
                except TimeoutError:
                    pass
                else:
                    futures.remove(future)
                    max_returncode = max(max_returncode, future.result().returncode)

                for progress, kwds in status_path_locks.items():
                    for path, skip in kwds.items():
                        
                        # copy the contents to a temp file
                        try:
                            with open(path, "r") as fp:
                                for n in range(skip):
                                    next(fp)
                                content = fp.readlines()
                        except FileNotFoundError:
                            continue
                        except:
                            # no content
                            continue
                        
                        kwds[path] += len(content)
                            
                        for line in content:
                            try:
                                command, *state = json.loads(line.rstrip())
                                if command == "add_task":
                                    number, args, kwds = state
                                    mappings[(path, number)] = progress.add_task(*args, **kwds)
                                elif command == "update":
                                    (ref_num, *args), kwds = state
                                    progress.update(mappings[(path, ref_num)], *args, **kwds)
                            except Exception as e:
                                log.exception(f"Failed to parse line: {line} - {e}")
                                continue

    sys.exit(max_returncode)


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The task name to run (e.g., `aspcap`, or `astra.pipelines.aspcap.aspcap`).")],
    spectrum_model: Annotated[str, typer.Argument(
        help=(
            "The spectrum model to use (e.g., `ApogeeCombinedSpectrum`, `BossCombinedSpectrum`). "
            "If `None` is given then all spectrum models accepted by the task will be analyzed."
        )
        )] = None,
    limit: Annotated[int, typer.Option(help="Limit the number of spectra.", min=1)] = None,
    page: Annotated[int, typer.Option(help="Page to start results from (`limit` spectra per `page`).", min=1)] = None,
    live_renderable_path: Annotated[str, typer.Option(hidden=True)] = None
):
    """Run an Astra task on spectra."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, BarColumn, MofNCompleteColumn 
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.logging import RichHandler
    from rich.console import Console
    from logging import FileHandler

    from astra import models, __version__, generate_queries_for_task
    from astra.utils import log, resolve_task, accepts_live_renderable

    fun = resolve_task(task)
    fun_accepts_live_renderable = accepts_live_renderable(fun)
    live_renderable = Table.grid()

    # Re-direct log handler
    console = Console()

    use_local_renderable = (live_renderable_path is None) and not fun_accepts_live_renderable
    if use_local_renderable:
        overall_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        live_renderable.add_row(Panel(overall_progress, title=task))
    
    with Live(live_renderable, console=console, redirect_stdout=False, redirect_stderr=False) as live:
        log.handlers.clear()
        log.handlers.extend([
            RichHandler(console=live.console, markup=True, rich_tracebacks=True),
        ])

        for model, q in generate_queries_for_task(fun, spectrum_model, limit, page=page):            
            if total := q.count():
                if use_local_renderable:
                    task = overall_progress.add_task(model.__name__, total=total)
                for r in fun(q, live_renderable=(live_renderable_path or live_renderable)):
                    if use_local_renderable:
                        overall_progress.update(task, advance=1)
                if use_local_renderable:
                    overall_progress.update(task, completed=True)

    """
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as p:
        t = p.add_task(description="Resolving task", total=None)
        fun = resolve_task(task)
        p.remove_task(t)

    messages = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        transient=not in_airflow_context
    ) as progress:
        
        for model, q in generate_queries_for_task(fun, spectrum_model, limit, page=page):
            t = progress.add_task(description=f"Running {fun.__name__} on {model.__name__}", total=limit)
            total = q.count()
            progress.update(t, total=total)
            if total > 0:
                for n, r in enumerate(fun(q, progress=progress), start=1):
                    progress.update(t, advance=1, refresh=True)
                messages.append(f"Processed {n} {model.__name__} spectra with {fun.__name__}")
            progress.update(t, completed=True)

    list(map(typer.echo, messages))
    """


@app.command()
def migrate(
    apred: Optional[str] = typer.Option(None, help="APOGEE data reduction pipeline version."),
    run2d: Optional[str] = typer.Option(None, help="BOSS data reduction pipeline version."),
):
    """Migrate spectra and auxillary information to the Astra database."""

    import os
    import multiprocessing as mp
    from signal import SIGKILL
    from rich.console import Console
    from rich.progress import Text, Progress, SpinnerColumn, Text, TextColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn as _MofNCompleteColumn

    class MofNCompleteColumn(_MofNCompleteColumn):
        def render(self, task):
            completed = int(task.completed)
            total = f"{int(task.total):,}" if task.total is not None else "?"
            total_width = len(str(total))
            return Text(
                f"{completed:{total_width},d}{self.separator}{total}",
                style="progress.download",
            )

    from astra.migrations.boss import (
        migrate_from_spall_file,
        migrate_specfull_metadata_from_image_headers
    )
    #from astra.migrations.apogee import (
    #    migrate_apvisit_metadata_from_image_headers,
    #)
    from astra.migrations.new_apogee import (
        migrate_apogee_spectra_from_sdss5_apogee_drpdb,
        migrate_dithered_metadata
    )
    from astra.migrations.catalog import (
        migrate_healpix,
        migrate_twomass_photometry,
        migrate_unwise_photometry,
        migrate_glimpse_photometry,
        migrate_tic_v8_identifier,
        migrate_gaia_source_ids,
        migrate_gaia_dr3_astrometry_and_photometry,
        migrate_zhang_stellar_parameters,
        migrate_bailer_jones_distances,
        migrate_gaia_synthetic_photometry
    )
    from astra.migrations.misc import (
        compute_f_night_time_for_boss_visits,
        compute_f_night_time_for_apogee_visits,
        update_visit_spectra_counts,       
        compute_n_neighborhood,
        update_galactic_coordinates,
        compute_w1mag_and_w2mag,
        fix_unsigned_apogee_flags
    )
    from astra.migrations.reddening import update_reddening
    from astra.migrations.targeting import (
        migrate_carton_assignments_to_bigbitfield,
        migrate_targeting_cartons
    )
    from astra.utils import silenced
    import sys
    if in_airflow_context:
        console = Console(file=sys.stdout)
    else:
        console = Console()

    ptq = []
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(),
            transient=not in_airflow_context
        ) as progress:

            def process_task(target, *args, description=None, **kwargs):
                queue = mp.Queue()

                kwds = dict(queue=queue)
                kwds.update(kwargs)
                process = mp.Process(target=target, args=args, kwargs=kwds)
                process.start()

                task = progress.add_task(description=(description or ""), total=None)
                return (process, task, queue)

            
            if apred is not None or run2d is not None:
                if apred is not None:
                    if apred == "dr17":
                        from astra.migrations.apogee import migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb 
                        ptq.append(process_task(migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb, description="Ingesting APOGEE dr17 spectra"))
                    else:
                        ptq.append(process_task(migrate_apogee_spectra_from_sdss5_apogee_drpdb, apred, description=f"Ingesting APOGEE {apred} spectra"))
                if run2d is not None:
                    ptq.append(process_task(migrate_from_spall_file, run2d, description=f"Ingesting BOSS {run2d} spectra"))
                                
                awaiting = set(t for p, t, q in ptq)
                while awaiting:
                    for p, t, q in ptq:
                        try:
                            r = q.get(False)
                            if r is Ellipsis:
                                progress.update(t, completed=True)
                                awaiting.remove(t)
                                p.join()
                                progress.update(t, visible=False)
                            else:
                                progress.update(t, **r)
                                if "completed" in r and r.get("completed", None) == 0:
                                    # reset the task
                                    progress.reset(t)
                        except mp.queues.Empty:
                            pass

            # Now that we have sources and spectra, we can do other things.
            ptq = [
                process_task(migrate_gaia_source_ids, description="Ingesting Gaia DR3 source IDs"),
                process_task(migrate_twomass_photometry, description="Ingesting 2MASS photometry"),
                process_task(migrate_unwise_photometry, description="Ingesting unWISE photometry"),
                process_task(migrate_glimpse_photometry, description="Ingesting GLIMPSE photometry"),
                process_task(migrate_specfull_metadata_from_image_headers, description="Ingesting specFull metadata"),


                process_task(migrate_dithered_metadata, description="Ingesting APOGEE dithered metadata"),
                #process_task(migrate_apvisit_metadata_from_image_headers, description="Ingesting apVisit metadata"),                
                process_task(migrate_healpix, description="Ingesting HEALPix values"),
                process_task(migrate_tic_v8_identifier, description="Ingesting TIC v8 identifiers"),
                process_task(update_galactic_coordinates, description="Computing Galactic coordinates"),
                process_task(fix_unsigned_apogee_flags, description="Fix unsigned APOGEE flags"),
                process_task(migrate_targeting_cartons, description="Ingesting targeting cartons"),
                process_task(compute_f_night_time_for_apogee_visits, description="Computing f_night for APOGEE visits"),                        
                process_task(update_visit_spectra_counts, description="Updating visit spectra counts"),               
            ]
            # reddening needs unwise, 2mass, glimpse, 
            task_gaia, task_twomass, task_unwise, task_glimpse, task_specfull, *_ = [t for p, t, q in ptq]
            reddening_requires = {task_twomass, task_unwise, task_glimpse, task_gaia}
            started_reddening = False
            awaiting = set(t for p, t, q in ptq)
            while awaiting:
                additional_tasks = []
                for p, t, q in ptq:
                    try:
                        r = q.get(False)
                        if r is Ellipsis:
                            progress.update(t, completed=True)
                            awaiting.remove(t)
                            p.join()
                            progress.update(t, visible=False)
                            if t == task_gaia:
                                # Add a bunch more tasks!
                                new_tasks = [
                                    process_task(migrate_gaia_dr3_astrometry_and_photometry, description="Ingesting Gaia DR3 astrometry and photometry"),
                                    process_task(migrate_zhang_stellar_parameters, description="Ingesting Zhang stellar parameters"),
                                    process_task(migrate_bailer_jones_distances, description="Ingesting Bailer-Jones distances"),                 
                                    #process_task(migrate_gaia_synthetic_photometry, description="Ingesting Gaia synthetic photometry"),
                                    process_task(compute_n_neighborhood, description="Computing n_neighborhood"),
                                ]
                                reddening_requires.update({t for p, t, q in new_tasks[:3]}) # reddening needs Gaia astrometry, Zhang parameters, and Bailer-Jones distances
                                additional_tasks.extend(new_tasks)
                            if t == task_specfull:
                                additional_tasks.append(
                                    process_task(compute_f_night_time_for_boss_visits, description="Computing f_night for BOSS visits")
                                )
                            if t == task_unwise:
                                additional_tasks.append(
                                    process_task(compute_w1mag_and_w2mag, description="Computing W1, W2 mags")
                                )
                            if not started_reddening and not (awaiting & reddening_requires):
                                started_reddening = True
                                #additional_tasks.append(
                                #    process_task(update_reddening, description="Computing extinction")
                                #)
                        else:
                            progress.update(t, **r)
                            if "completed" in r and r.get("completed", None) == 0:
                                # reset the task
                                progress.reset(t)

                    except mp.queues.Empty:
                        pass

                ptq.extend(additional_tasks)
                awaiting |= set(t for p, t, q in additional_tasks)

    except KeyboardInterrupt:  
        """
        with silenced():
            import psutil
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
        """
        raise KeyboardInterrupt



@app.command()
def init(
    drop_tables: Optional[bool] = typer.Option(False, help="Drop tables if they exist."), 
    delay: Optional[int] = typer.Option(10, help="Delay in seconds to wait.")
):
    """Initialize the Astra database."""

    from time import sleep
    from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
    from importlib import import_module
    from astra.models.base import (database, BaseModel)
    from astra.models.pipeline import PipelineOutputModel

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeRemainingColumn(),
        transient=not in_airflow_context
    ) as progress:

        init_model_packages = (
            "apogee",
            "boss",
            "bossnet",
            "apogeenet",
            "astronn_dist",
            "astronn",
            "source",
            "spectrum",
        )
        for package in init_model_packages:
            import_module(f"astra.models.{package}")
        
        models = set(BaseModel.__subclasses__()) - {PipelineOutputModel}
        
        if drop_tables:
            tables_to_drop = [m for m in models if m.table_exists()]
            if delay > 0:
                t = progress.add_task(description=f"About to drop {len(tables_to_drop)} tables..", total=delay)
                for i in range(delay):
                    progress.advance(t)
                    sleep(1)
        
            with database.atomic():
                database.drop_tables(tables_to_drop, cascade=True)
            progress.remove_task(t)

        t = progress.add_task(description="Creating tables", total=len(models))
        with database.atomic():
            database.create_tables(models)
        

if __name__ == "__main__":
    app()