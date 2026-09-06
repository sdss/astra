#!/usr/bin/env python3
import typer
import os
from pathlib import Path
from typing import List, Optional, Tuple
from typing_extensions import Annotated
from enum import Enum
import inspect

app = typer.Typer()
config_app = typer.Typer(help="Manage Astra configuration.")
app.add_typer(config_app, name="config")
validate_app = typer.Typer(help="Validate Astra products.")
app.add_typer(validate_app, name="validate")

in_airflow_context = os.environ.get("AIRFLOW_CTX_TASK_ID", None) is not None

# User config file path
USER_CONFIG_DIR = Path.home() / ".config" / "sdss" / "astra"
USER_CONFIG_FILE = USER_CONFIG_DIR / "astra.yml"


def _get_nested_value(d: dict, key: str):
    """Get a value from a nested dict using dot notation (e.g., 'database.host')."""
    keys = key.split(".")
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def _set_nested_value(d: dict, key: str, value):
    """Set a value in a nested dict using dot notation (e.g., 'database.host')."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _format_config(d: dict, indent: int = 0) -> str:
    """Format a config dict for display."""
    lines = []
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_config(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def _load_user_config() -> dict:
    """Load the user config file if it exists."""
    if USER_CONFIG_FILE.exists():
        import yaml
        with open(USER_CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_user_config(config: dict) -> None:
    """Save config to the user config file."""
    import yaml
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


@config_app.command("show")
def config_show():
    """Show all configuration settings."""
    from astra import config
    typer.echo(_format_config(dict(config)))


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="The configuration key (use dot notation for nested keys, e.g., 'database.host').")]
):
    """Get a configuration value."""
    from astra import config
    value = _get_nested_value(dict(config), key)
    if value is None:
        typer.echo(f"Key '{key}' not found.", err=True)
        raise typer.Exit(1)
    if isinstance(value, dict):
        typer.echo(_format_config(value))
    else:
        typer.echo(value)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="The configuration key (use dot notation for nested keys, e.g., 'database.host').")],
    value: Annotated[str, typer.Argument(help="The value to set.")]
):
    """Set a configuration value in the user config file."""
    import yaml

    # Try to parse value as YAML to handle booleans, numbers, etc.
    try:
        parsed_value = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed_value = value

    user_config = _load_user_config()
    _set_nested_value(user_config, key, parsed_value)
    _save_user_config(user_config)

    typer.echo(f"Set {key} = {parsed_value}")
    typer.echo(f"Saved to {USER_CONFIG_FILE}")


@config_app.command("path")
def config_path():
    """Show the path to the user configuration file."""
    typer.echo(f"User config file: {USER_CONFIG_FILE}")
    if USER_CONFIG_FILE.exists():
        typer.echo("(exists)")
    else:
        typer.echo("(does not exist yet)")


@validate_app.command("summary")
def validate_summary(
    paths: Annotated[List[Path], typer.Argument(help="Paths to FITS summary files to validate.")],
    log_file: Annotated[Optional[Path], typer.Option(help="Path to write the log file.")] = None,
    json_file: Annotated[Optional[Path], typer.Option(help="Path to write JSON report.")] = None,
):
    """Check for null columns in Astra summary FITS files."""
    import logging
    import json
    import numpy as np
    from astropy.table import Table
    from astropy.io import fits

    logger = logging.getLogger("astra.validate.summary")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    all_results = {}
    for file_path in paths:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"Checking {file_path.name}")
        results_tbl = {}
        hdu = fits.open(file_path)
        for i in range(1, len(hdu)):
            results_tbl[f"HDU {i}"] = {}
            tbl = Table.read(file_path, hdu=i)

            if len(tbl) == 0:
                continue

            # Find columns where all values are masked/null
            null_cols_all = [
                col for col in tbl.colnames
                if hasattr(tbl[col], "mask") and tbl[col].mask.all()
            ]
            results_tbl[f"HDU {i}"]["all"] = null_cols_all

            if null_cols_all:
                logger.warning(
                    f"HDU = {i}, File: {file_path.name} | "
                    f"All-null columns ({len(null_cols_all)}): {', '.join(null_cols_all)}"
                )

            # Per-release breakdown
            try:
                release = np.unique(tbl["release"])
                for r in release:
                    ev_release = tbl["release"] == r
                    null_cols = [
                        col for col in tbl.colnames
                        if hasattr(tbl[col], "mask") and tbl[col][ev_release].mask.all()
                    ]
                    null_cols = [c for c in null_cols if c not in null_cols_all]
                    results_tbl[f"HDU {i}"][r] = null_cols

                    if null_cols:
                        logger.warning(
                            f"HDU = {i}, File: {file_path.name} | "
                            f"{r} only-null columns ({len(null_cols)}): {', '.join(null_cols)}"
                        )
            except KeyError:
                pass

        hdu.close()
        all_results[file_path.name] = results_tbl

    if json_file is not None:
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=4)
        typer.echo(f"JSON report written to {json_file}")


class Product(str, Enum):
    mwmTargets = "mwmTargets"
    mwmAllStar = "mwmAllStar"
    mwmAllVisit = "mwmAllVisit"
    mwmStar = "mwmStar"
    mwmVisit = "mwmVisit"
    mwmVisit_mwmStar = "mwmVisit/mwmStar"
    astraAllStarASPCAP = "astraAllStarASPCAP"
    astraAllVisitASPCAP = "astraAllVisitASPCAP"
    astraAllStarAPOGEENet = "astraAllStarAPOGEENet"
    astraAllVisitAPOGEENet = "astraAllVisitAPOGEENet"
    astraAllStarBOSSNet = "astraAllStarBOSSNet"
    astraAllVisitBOSSNet = "astraAllVisitBOSSNet"
    astraAllStarLineForest = "astraAllStarLineForest"
    astraAllVisitLineForest = "astraAllVisitLineForest"
    astraAllStarAstroNN = "astraAllStarAstroNN"
    astraAllVisitAstroNN = "astraAllVisitAstroNN"
    astraAllStarAstroNNDist = "astraAllStarAstroNNDist"
    astraAllStarSlam = "astraAllStarSlam"
    astraAllStarMDwarfType = "astraAllStarMDwarfType"
    astraAllVisitMDwarfType = "astraAllVisitMDwarfType"
    astraAllVisitCorv = "astraAllVisitCorv"

    astraAllStarSnowWhite = "astraAllStarSnowWhite"
    astraAllStarThePayne = "astraAllStarThePayne"




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
    from astra.products.mwm import create_mwmVisit_and_mwmStar_products
    from astra.models.apogee import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar
    from astra.models.boss import BossVisitSpectrum
    from astra.models.mwm import (BossCombinedSpectrum, ApogeeCombinedSpectrum, BossRestFrameVisitSpectrum, ApogeeRestFrameVisitSpectrum)
    from astra.models.source import Source

    mwmVisit_mwmStar_args = (
        create_mwmVisit_and_mwmStar_products,
        dict(
            task="astra.products.mwm.create_mwmVisit_and_mwmStar_products",
            sources=Source.select(),
            batch_size=1,
            overwrite=overwrite
        )
    )
    mapping = (
        {
            Product.mwmVisit: mwmVisit_mwmStar_args,
            Product.mwmStar: mwmVisit_mwmStar_args,
            Product.mwmVisit_mwmStar: mwmVisit_mwmStar_args,
            Product.mwmTargets: (create_mwm_targets_product, dict(overwrite=overwrite)),
            Product.mwmAllVisit: (create_mwm_all_visit_product, dict(overwrite=overwrite)),
            Product.mwmAllStar: (create_mwm_all_star_product, dict(overwrite=overwrite)),
            Product.astraAllStarASPCAP: (
                create_all_star_product,
                {
                    "pipeline_model": "aspcap.ASPCAP",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitASPCAP: (
                create_all_visit_product,
                {
                    "pipeline_model": "aspcap.ASPCAP",
                    "apogee_spectrum_model": ApogeeVisitSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarAPOGEENet: (
                create_all_star_product,
                {
                    "pipeline_model": "apogeenet.ApogeeNet",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitAPOGEENet: (
                create_all_visit_product,
                {
                    "pipeline_model": "apogeenet.ApogeeNet",
                    "apogee_spectrum_model": ApogeeVisitSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarBOSSNet: (
                create_all_star_product,
                {
                    "pipeline_model": "bossnet.BossNet",
                    "boss_spectrum_model": BossCombinedSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitBOSSNet: (
                create_all_visit_product,
                {
                    "pipeline_model": "bossnet.BossNet",
                    "boss_spectrum_model": BossVisitSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarLineForest: (
                create_all_star_product,
                {
                    "pipeline_model": "line_forest.LineForest",
                    "boss_spectrum_model": BossCombinedSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitLineForest: (
                create_all_visit_product,
                {
                    "pipeline_model": "line_forest.LineForest",
                    "boss_spectrum_model": BossVisitSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarAstroNN: (
                create_all_star_product,
                {
                    "pipeline_model": "astronn.AstroNN",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitAstroNN: (
                create_all_visit_product,
                {
                    "pipeline_model": "astronn.AstroNN",
                    "apogee_spectrum_model": ApogeeVisitSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarAstroNNDist: (
                create_all_star_product,
                {
                    "pipeline_model": "astronn_dist.AstroNNdist",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarSlam: (
                create_all_star_product,
                {
                    "pipeline_model": "slam.Slam",
                    "boss_spectrum_model": BossCombinedSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarMDwarfType: (
                create_all_star_product,
                {
                    "pipeline_model": "mdwarftype.MDwarfType",
                    "boss_spectrum_model": BossCombinedSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitMDwarfType: (
                create_all_visit_product,
                {
                    "pipeline_model": "mdwarftype.MDwarfType",
                    "boss_spectrum_model": BossVisitSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllVisitCorv: (
                create_all_visit_product,
                {
                    "pipeline_model": "corv.Corv",
                    "boss_spectrum_model": BossVisitSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarSnowWhite: (
                create_all_star_product,
                {
                    "pipeline_model": "snow_white.SnowWhite",
                    "boss_spectrum_model": BossCombinedSpectrum,
                    "overwrite": overwrite
                }
            ),
            Product.astraAllStarThePayne: (
                create_all_star_product,
                {
                    "pipeline_model": "the_payne.ThePayne",
                    "apogee_spectrum_model": ApogeeCoaddedSpectrumInApStar,
                    "overwrite": overwrite
                }
            )
        }
    )

    for product in products:
        fun, kwargs = mapping[product]
        r = fun(limit=limit, **kwargs)
        if inspect.isgenerator(r):
            for _ in r:
                pass
        elif isinstance(r, str):
            typer.echo(f"Created {product}: {r}")
        else:
            pass


@app.command()
def srun(
    task: Annotated[str, typer.Argument(help="The task name to run (e.g., `aspcap`, or `astra.pipelines.aspcap.aspcap`).")],
    model: Annotated[str, typer.Argument(
        help=(
            "The input model to use (e.g., `ApogeeCombinedSpectrum`, `BossCombinedSpectrum`). If no model is given then "
            "only the first model accepted by that task (that has spectra) will be used."
        )
        )] = None,
    sdss_ids: Annotated[List[int], typer.Argument(help="Restrict to some set of SDSS IDs.")] = None,
    nodes: Annotated[int, typer.Option(help="The number of nodes to use.", min=1)] = 1,
    procs: Annotated[int, typer.Option(help="The number of `astra` processes to use per node.", min=1)] = 1,
    limit: Annotated[int, typer.Option(help="Limit the number of inputs.", min=1)] = None,
    account: Annotated[str, typer.Option(help="Slurm account")] = "sdss-np",
    partition: Annotated[str, typer.Option(help="Slurm partition")] = None,
    qos: Annotated[str, typer.Option(help="Slurm QoS")] = None,
    gres: Annotated[str, typer.Option(help="Slurm generic resources")] = None,
    mem: Annotated[str, typer.Option(help="Memory per node")] = 0,
    time: Annotated[str, typer.Option(help="Wall-time")] = "24:00:00",
    exclusive: Annotated[bool, typer.Option(help="Use exclusive node allocation.")] = True,
    only_missing: Annotated[bool, typer.Option("--only-missing", help="Only include rows missing from the output model (skip those with stale results).")] = False,
    pipeline: Annotated[str, typer.Option(help="Pipeline name, for tasks parameterized by pipeline (e.g. `ASPCAP`).")] = None,
    apreds: Annotated[List[str], typer.Option(help="APOGEE reduction version(s), for tasks that accept an `apreds` parameter.")] = None,
    run2ds: Annotated[List[str], typer.Option(help="BOSS reduction version(s), for tasks that accept a `run2ds` parameter.")] = None,
):
    """Distribute an Astra task over many nodes using Slurm."""

    # check that spectrum_model is not an integer
    try:
        model = int(model)
    except:
        None
    else:
        if sdss_ids is None:
            sdss_ids = [model]
        else:
            sdss_ids.append(model)
        model = None

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

    queries = generate_queries_for_task(task, model, sdss_ids=sdss_ids, limit=limit, missing_only=only_missing, pipeline=pipeline)

    considered_models = []
    for model, q in queries:
        total = q.count()
        if total == 0:
            considered_models.append(model)
            continue
        else:
            break
    else:
        log.info(f"No {', or '.join([m.__name__ for m in considered_models])} spectra to process.")
        sys.exit(0)

    workers = nodes * procs
    limit = int(np.ceil(total / workers))
    today = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(expand_path("$PBS"), exist_ok=True)

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

                if sdss_ids is None:
                    sdss_id_str = ""
                else:
                    sdss_id_str = " ".join(map(str, sdss_ids))

                only_missing_flag = " --only-missing" if only_missing else ""
                pipeline_flag = f" --pipeline {pipeline}" if pipeline is not None else ""
                apreds_flag = "".join(f" --apreds {a}" for a in apreds) if apreds else ""
                run2ds_flag = "".join(f" --run2ds {r}" for r in run2ds) if run2ds else ""
                extra_flags = f"{only_missing_flag}{pipeline_flag}{apreds_flag}{run2ds_flag}"

                # TODO: Let's not hard code this here.
                commands = ["export CLUSTER=1"]
                for page in range(n * procs, (n + 1) * procs):
                    status_path = f"{td}/live-{n}-{page}"
                    status_path_locks[progress][status_path] = 0
                    commands.append(f"astra run {task} {model.__name__} {sdss_id_str} --limit {limit} --page {page + 1}{extra_flags} --live-renderable-path {status_path} &")
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
                if exclusive:
                    executable.append(f"--exclusive")
                if qos is not None:
                    executable.append(f"--qos={qos}")
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
                    None
                else:
                    futures.remove(future)
                    max_returncode = max(max_returncode, future.result().returncode)

                for progress, kwds in status_path_locks.items():

                    for path, skip in kwds.items():
                        try:
                            with open(path, "r") as fp:
                                for n in range(skip):
                                    next(fp)
                                content = fp.readlines()
                        except FileNotFoundError:
                            continue
                        except:
                            # no new content
                            continue

                        kwds[path] += len(content)

                        for line in content:
                            try:
                                command, *state = json.loads(line.rstrip())
                                if command == "add_task":
                                    number, args, task_kwds = state
                                    mappings[(path, number)] = progress.add_task(*args, **task_kwds)
                                elif command == "update":
                                    (ref_num, *args), task_kwds = state
                                    progress.update(mappings[(path, ref_num)], *args, **task_kwds)

                            except Exception as e:
                                log.exception(f"Failed to parse line: {line} - {e}")


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
    sdss_ids: Annotated[List[int], typer.Argument(help="Restrict to some set of SDSS IDs.")] = None,
    limit: Annotated[int, typer.Option(help="Limit the number of spectra.", min=1)] = None,
    page: Annotated[int, typer.Option(help="Page to start results from (`limit` spectra per `page`).", min=1)] = None,
    live_renderable_path: Annotated[str, typer.Option(hidden=True)] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the queries that would be run without executing them.")] = False,
    only_missing: Annotated[bool, typer.Option("--only-missing", help="Only include rows missing from the output model (skip those with stale results).")] = False,
    pipeline: Annotated[str, typer.Option(help="Pipeline name, for tasks parameterized by pipeline (e.g. `ASPCAP`).")] = None,
    apreds: Annotated[List[str], typer.Option(help="APOGEE reduction version(s), for tasks that accept an `apreds` parameter.")] = None,
    run2ds: Annotated[List[str], typer.Option(help="BOSS reduction version(s), for tasks that accept a `run2ds` parameter.")] = None,
):
    """Run an Astra task on spectra."""

    # check that spectrum_model is not an integer
    try:
        spectrum_model = int(spectrum_model)
    except:
        None
    else:
        if sdss_ids is None:
            sdss_ids = [spectrum_model]
        else:
            sdss_ids.append(spectrum_model)
        spectrum_model = None

    import os
    import json
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

    class RemoteProgress:
        def __init__(self, path):
            self.path = path
            self.task_counter = 0
            if not os.path.exists(path):
                with open(path, "w"):
                    pass
            return None

        def append(self, data):
            try:
                r = json.dumps(data) + "\n"
                with open(self.path, "a") as fp:
                    fp.write(r)
                return True
            except Exception as e:
                return False

        def update(self, *args, **kwargs):
            return self.append(("update", args, kwargs))

        def add_task(self, *args, **kwargs):
            self.task_counter += 1
            self.append(("add_task", self.task_counter, args, kwargs))
            return self.task_counter

    use_local_renderable = (live_renderable_path is None) and not fun_accepts_live_renderable
    if use_local_renderable:
        overall_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        live_renderable.add_row(Panel(overall_progress, title=task))
    elif live_renderable_path is not None:
        overall_progress = RemoteProgress(live_renderable_path)
    else:
        overall_progress = None

    iterable = generate_queries_for_task(
        fun,
        spectrum_model,
        sdss_ids=sdss_ids,
        limit=limit,
        page=page,
        missing_only=only_missing,
        pipeline=pipeline,
    )
    from time import sleep

    extra_kwargs = {}
    if pipeline is not None:
        extra_kwargs["pipeline"] = pipeline
    if apreds:
        extra_kwargs["apreds"] = apreds
    if run2ds:
        extra_kwargs["run2ds"] = run2ds

    with Live(live_renderable, console=console, redirect_stdout=False, redirect_stderr=False) as live:
        for model, q in iterable:
            if total := q.count():
                worker = fun(q, live=True, live_renderable=(live_renderable_path or live_renderable), **extra_kwargs)

                if use_local_renderable or (overall_progress is not None):
                    task_id = overall_progress.add_task(model.__name__)
                    overall_progress.update(task_id, total=total)
                    if dry_run:
                        from time import sleep
                        for r in range(total):
                            overall_progress.update(task_id, advance=1, refresh=True)
                            sleep(1)
                    else:
                        for r in worker:
                            overall_progress.update(task_id, advance=1, refresh=True)
                    #overall_progress.update(task_id, refresh=True, completed=True)
                else:
                    if not dry_run:
                        for r in worker:
                            pass



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
    limit: Optional[int] = typer.Option(None, help="Limit the number of spectra to migrate."),
    max_mjd: Optional[int] = typer.Option(None, help="Maximum MJD of spectra to migrate."),
    metadata: Optional[bool] = typer.Option(True, help="Migrate metadata (e.g., photometry, astrometry)."),
    incremental: Optional[bool] = typer.Option(True, help="Only attempt to migrate new spectra."),
    extinction: Optional[bool] = typer.Option(True, help="Compute extinction."),
):
    """Migrate spectra and auxiliary information to the Astra database."""

    import sys
    import multiprocessing as mp
    from typing import Dict

    # Set multiprocessing start method
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn as _MofNCompleteColumn, Text
    )
    from rich.live import Live
    from rich.panel import Panel
    from astra.utils import log

    # Import all migration functions
    from astra.migrations.boss import (
        migrate_from_spall_file,
        migrate_specfull_metadata_from_image_headers
    )
    from astra.migrations.apogee import (
        migrate_apogee_spectra_from_sdss5_apogee_drpdb,
        migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb,
        migrate_sdss4_dr17_apogee_coadded_spectra_from_visits,
        migrate_dithered_metadata,
        migrate_apogee_visits_in_apStar_files
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
        migrate_gaia_synthetic_photometry,
        migrate_sdss4_apogee_id
    )
    from astra.migrations.misc import (
        compute_f_night_time_for_boss_visits,
        compute_f_night_time_for_apogee_visits,
        update_visit_spectra_counts,
        compute_n_neighborhood,
        update_galactic_coordinates,
        compute_w1mag_and_w2mag,
        fix_unsigned_apogee_flags,
        update_sdss_id_related_fields
    )
    from astra.migrations.reddening import update_reddening, preload_dust_maps
    from astra.migrations.targeting import migrate_targeting_cartons
    from astra.migrations.source import create_sources_and_link_spectra
    from astra.migrations.scheduler import MigrationTask, MigrationScheduler, get_satisfiable_tasks

    class MofNCompleteColumn(_MofNCompleteColumn):
        def render(self, task):
            completed = int(task.completed)
            # Handle None or 0 total gracefully
            if task.total is None or task.total == 0:
                total = "?"
            else:
                total = f"{int(task.total):,}"
            total_width = len(str(total))
            return Text(
                f"{completed:{total_width},d}{self.separator}{total}",
                style="progress.download",
            )

    # Build the task graph
    tasks: Dict[str, MigrationTask] = {}

    # ==========================================================================
    # Phase 1: Spectrum ingestion (spectrum-level data only, no source creation)
    # ==========================================================================
    if apred is not None:
        if apred == "dr17":
            tasks["apogee_spectra"] = MigrationTask(
                name="apogee_spectra",
                func=migrate_sdss4_dr17_apogee_spectra_from_sdss5_catalogdb,
                kwargs={"limit": limit},
                description="Ingesting APOGEE dr17 spectra",
                writes_to={"apogee_visit_spectrum"}
            )
        else:
            tasks["apogee_spectra"] = MigrationTask(
                name="apogee_spectra",
                func=migrate_apogee_spectra_from_sdss5_apogee_drpdb,
                args=(apred,),
                kwargs={"limit": limit, "incremental": incremental, "max_mjd": max_mjd},
                description=f"Ingesting APOGEE {apred} spectra",
                writes_to={"apogee_visit_spectrum"}
            )
            tasks["apstar_visits"] = MigrationTask(
                name="apstar_visits",
                func=migrate_apogee_visits_in_apStar_files,
                args=(apred, ),
                description="Ingesting ApogeeVisitSpectrumInApStar entries",
                depends_on={"apogee_spectra"},
                writes_to={"apogee_visit_spectrum_in_ap_star"}
            )

    if run2d is not None:
        tasks["boss_spectra"] = MigrationTask(
            name="boss_spectra",
            func=migrate_from_spall_file,
            args=(run2d,),
            kwargs={"limit": limit, "incremental": incremental, "max_mjd": max_mjd},
            description=f"Ingesting BOSS {run2d} spectra",
            writes_to={"boss_visit_spectrum"}
        )

    # ==========================================================================
    # Phase 2: Source creation and linking (after spectra are ingested)
    # ==========================================================================
    spectra_deps = {"apogee_spectra", "boss_spectra"} & tasks.keys()

    # `apstar_visits` must complete first, not merely run without overlapping writes:
    # create_sources links ApogeeVisitSpectrumInApStar rows, so if it wins the race those
    # rows do not exist yet and nothing ever comes back to link them.
    create_source_deps = spectra_deps | ({"apstar_visits"} & tasks.keys())

    tasks["create_sources"] = MigrationTask(
        name="create_sources",
        func=create_sources_and_link_spectra,
        description="Creating sources and linking spectra",
        depends_on=create_source_deps,
        writes_to={
            "source",
            "boss_visit_spectrum",
            "apogee_visit_spectrum",
            "apogee_visit_spectrum_in_ap_star",
            "apogee_coadded_spectrum_in_ap_star",
        }
    )

    # ==========================================================================
    # Phase 3: Metadata (depends on sources being created)
    # ==========================================================================
    if metadata:
        apogee_deps = {"apogee_spectra"} & tasks.keys()
        boss_deps = {"boss_spectra"} & tasks.keys()
        source_deps = {"create_sources"} & tasks.keys()

        # --- Source table updates (run sequentially to avoid write conflicts) ---

        # Update sdss_id-related fields first (needed for downstream operations)
        tasks["sdss_id_fields"] = MigrationTask(
            name="sdss_id_fields",
            func=update_sdss_id_related_fields,
            description="Updating sdss_id related fields",
            depends_on=source_deps or spectra_deps,
            writes_to={"source"}
        )

        # Gaia chain: must run sequentially, each depends on previous
        tasks["gaia_source_ids"] = MigrationTask(
            name="gaia_source_ids",
            func=migrate_gaia_source_ids,
            description="Ingesting Gaia DR3 source IDs",
            depends_on={"sdss_id_fields"} if "sdss_id_fields" in tasks else (source_deps or spectra_deps),
            writes_to={"source"}
        )
        tasks["sdss4_apogee_id"] = MigrationTask(
            name="sdss4_apogee_id",
            func=migrate_sdss4_apogee_id,
            description="Ingesting SDSS4 APOGEE IDs",
            depends_on={"gaia_source_ids"},
            writes_to={"source"}
        )
        if apred == "dr17":
            # Coadds are matched to sources via Source.sdss4_apogee_id, so this can only
            # run after sdss4_apogee_id is populated -- NOT as part of the Phase 1
            # apogee_spectra task, which runs before Source even exists.
            tasks["dr17_coadds"] = MigrationTask(
                name="dr17_coadds",
                func=migrate_sdss4_dr17_apogee_coadded_spectra_from_visits,
                description="Deriving APOGEE DR17 coadded spectra",
                depends_on={"sdss4_apogee_id", "apogee_spectra"},
                writes_to={"apogee_coadded_spectrum_in_ap_star"}
            )
        tasks["gaia_astrometry"] = MigrationTask(
            name="gaia_astrometry",
            func=migrate_gaia_dr3_astrometry_and_photometry,
            description="Ingesting Gaia DR3 astrometry and photometry",
            depends_on={"sdss4_apogee_id"},
            writes_to={"source"}
        )
        tasks["zhang_params"] = MigrationTask(
            name="zhang_params",
            func=migrate_zhang_stellar_parameters,
            description="Ingesting Zhang stellar parameters",
            depends_on={"gaia_astrometry"},
            writes_to={"source"}
        )
        tasks["bailer_jones"] = MigrationTask(
            name="bailer_jones",
            func=migrate_bailer_jones_distances,
            description="Ingesting Bailer-Jones distances",
            depends_on={"zhang_params"},
            writes_to={"source"}
        )
        tasks["gaia_synth_phot"] = MigrationTask(
            name="gaia_synth_phot",
            func=migrate_gaia_synthetic_photometry,
            description="Ingesting Gaia synthetic photometry",
            depends_on={"bailer_jones"},
            writes_to={"source"}
        )
        """
        tasks["n_neighborhood"] = MigrationTask(
            name="n_neighborhood",
            func=compute_n_neighborhood,
            description="Computing n_neighborhood",
            depends_on={"gaia_synth_phot"},
            writes_to={"source"}
        )
        """

        # Photometry tasks: can run after Gaia chain to avoid source conflicts
        tasks["twomass"] = MigrationTask(
            name="twomass",
            func=migrate_twomass_photometry,
            description="Ingesting 2MASS photometry",
            depends_on={"gaia_synth_phot"},
            writes_to={"source"}
        )
        tasks["unwise"] = MigrationTask(
            name="unwise",
            func=migrate_unwise_photometry,
            description="Ingesting unWISE photometry",
            depends_on={"twomass"},
            writes_to={"source"}
        )
        tasks["glimpse"] = MigrationTask(
            name="glimpse",
            func=migrate_glimpse_photometry,
            description="Ingesting GLIMPSE photometry",
            depends_on={"unwise"},
            writes_to={"source"}
        )

        # Other source updates: chain after photometry
        tasks["healpix"] = MigrationTask(
            name="healpix",
            func=migrate_healpix,
            description="Ingesting HEALPix values",
            depends_on={"glimpse"},
            writes_to={"source"}
        )
        tasks["tic_v8"] = MigrationTask(
            name="tic_v8",
            func=migrate_tic_v8_identifier,
            description="Ingesting TIC v8 identifiers",
            depends_on={"healpix"},
            writes_to={"source"}
        )
        tasks["galactic_coords"] = MigrationTask(
            name="galactic_coords",
            func=update_galactic_coordinates,
            description="Computing Galactic coordinates",
            depends_on={"tic_v8"},
            writes_to={"source"}
        )
        tasks["targeting_cartons"] = MigrationTask(
            name="targeting_cartons",
            func=migrate_targeting_cartons,
            description="Ingesting targeting cartons",
            depends_on={"galactic_coords"},
            writes_to={"source"}
        )
        tasks["visit_counts"] = MigrationTask(
            name="visit_counts",
            func=update_visit_spectra_counts,
            description="Updating visit spectra counts",
            depends_on={"targeting_cartons"},  # Only needs sources to exist
            writes_to={"source"}
        )
        tasks["w1w2_mags"] = MigrationTask(
            name="w1w2_mags",
            func=compute_w1mag_and_w2mag,
            description="Computing W1, W2 mags",
            depends_on={"unwise"},  # Needs unWISE photometry
            writes_to={"source"}
        )

        # --- BOSS spectrum updates ---
        # Always include these tasks when --metadata is set; they'll no-op if no spectra exist
        # Only depend on source creation if we're actually ingesting new spectra
        boss_task_deps = (source_deps or boss_deps) if boss_deps else set()
        tasks["specfull_metadata"] = MigrationTask(
            name="specfull_metadata",
            func=migrate_specfull_metadata_from_image_headers,
            description="Ingesting specFull metadata",
            depends_on=boss_task_deps,
            writes_to={"boss_visit_spectrum"},
            exclusive=True  # Uses internal process pool
        )
        tasks["f_night_boss"] = MigrationTask(
            name="f_night_boss",
            func=compute_f_night_time_for_boss_visits,
            description="Computing f_night for BOSS visits",
            depends_on={"specfull_metadata"},
            writes_to={"boss_visit_spectrum"}
        )

        # --- APOGEE spectrum updates ---
        # Always include these tasks when --metadata is set; they'll no-op if no spectra exist
        # Only depend on source creation if we're actually ingesting new spectra
        apogee_task_deps = (source_deps or apogee_deps) if apogee_deps else set()
        tasks["dithered_metadata"] = MigrationTask(
            name="dithered_metadata",
            func=migrate_dithered_metadata,
            description="Ingesting APOGEE dithered metadata",
            depends_on=apogee_task_deps,
            writes_to={"apogee_visit_spectrum"}
        )
        tasks["fix_apogee_flags"] = MigrationTask(
            name="fix_apogee_flags",
            func=fix_unsigned_apogee_flags,
            description="Fix unsigned APOGEE flags",
            depends_on=apogee_task_deps,
            writes_to={"apogee_visit_spectrum"}
        )
        tasks["f_night_apogee"] = MigrationTask(
            name="f_night_apogee",
            func=compute_f_night_time_for_apogee_visits,
            description="Computing f_night for APOGEE visits",
            depends_on=apogee_task_deps,
            writes_to={"apogee_visit_spectrum"}
        )

        # --- Extinction (depends on photometry and distances) ---
        if extinction:
            # Start preloading dust maps early (no writes, so can run in parallel with photometry)
            # This populates OS file cache so the actual reddening computation loads maps faster
            '''
            tasks["preload_dust_maps"] = MigrationTask(
                name="preload_dust_maps",
                func=preload_dust_maps,
                description="Preloading dust maps",
                depends_on=source_deps or spectra_deps,
                writes_to=set()  # No writes, allows parallel execution with other source tasks
            )
            '''
            # Actual reddening computation depends on dust map preload AND photometry
            tasks["reddening"] = MigrationTask(
                name="reddening",
                func=update_reddening,
                description="Computing extinction",
                #depends_on={"preload_dust_maps", "twomass", "unwise", "glimpse", "bailer_jones"} & (tasks.keys() | {"preload_dust_maps"}),
                depends_on={"targeting_cartons", "w1w2_mags", "twomass", "unwise", "glimpse", "bailer_jones"} & (tasks.keys()),
                writes_to={"source"}
            )

    # Remove tasks with unsatisfied dependencies (from disabled features)
    tasks = get_satisfiable_tasks(tasks)

    if not tasks:
        typer.echo("No migration tasks to run.")
        return

    # Set up progress display
    console = Console(file=sys.stdout) if in_airflow_context else Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    # Run the scheduler
    try:
        with Live(
            Panel(progress, title="Migration Progress", border_style="blue"),
            console=console,
            redirect_stdout=False,
            redirect_stderr=False,
            refresh_per_second=4,
            transient=True  # Remove display when complete
        ):
            log.handlers.clear()
            scheduler = MigrationScheduler(tasks, progress)
            scheduler.run()

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    typer.echo(f"Migration complete. {len(tasks)} tasks executed.")



@app.command()
def init(
    drop_tables: Optional[bool] = typer.Option(False, help="Drop tables if they exist."),
    delay: Optional[int] = typer.Option(10, help="Delay in seconds to wait.")
):
    """Initialize the Astra database."""

    from time import sleep
    from importlib import import_module
    from astra.models.base import (database, BaseModel)

    init_model_packages = (
        "apogee",
        "aspcap",
        "boss",
        "bossnet",
        "apogeenet",
        "astronn_dist",
        "astronn",
        "source",
        "spectrum",
        "line_forest",
        "mwm",
        "snow_white",
        "corv",
        "slam",
        "mdwarftype",
        "the_payne"
    )
    for package in init_model_packages:
        import_module(f"astra.models.{package}")

    schema = BaseModel._meta.schema
    database.execute_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    typer.echo(f"Initializing Astra database schema '{schema}'...")

    # For Mixin models, only create sub-classes of them
    models = []
    for model in BaseModel.__subclasses__():
        if model.__name__.endswith("Mixin"):
            models.extend(model.__subclasses__())
        else:
            models.append(model)

    if drop_tables:
        tables_to_drop = [m for m in models if m.table_exists()]
        if delay > 0:
            typer.echo(f"About to drop {len(tables_to_drop)} tables in {delay} seconds...")
            sleep(delay)

        with database.atomic():
            database.drop_tables(tables_to_drop, cascade=True)
        typer.echo(f"Dropped {len(tables_to_drop)} tables.")

    with database.atomic():
        database.create_tables(models)

    typer.echo(f"Created {len(models)} tables in the Astra database:")
    for m in models:
        typer.echo(f" - {m._meta.schema}.{m._meta.table_name}")

    database.execute_sql(
        f"grant all privileges on schema {schema} to group sdss;"
        f"grant all privileges on all tables in schema {schema} to sdss;"
    )
    typer.echo(f"Granted all privileges on schema '{schema}' to group 'sdss'.")
    admin_uids = ["u6033276"]
    for uid in admin_uids:
        database.execute_sql(
            f"GRANT ALL PRIVILEGES ON SCHEMA {schema} TO {uid};"
            f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} TO {uid};"
            f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {schema} TO {uid};"
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
            f"GRANT ALL PRIVILEGES ON TABLES TO {uid};"
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
            f"GRANT ALL PRIVILEGES ON SEQUENCES TO {uid};"
        )
        typer.echo(f"Granted all privileges on schema '{schema}' to admin user '{uid}'.")

if __name__ == "__main__":
    app()
