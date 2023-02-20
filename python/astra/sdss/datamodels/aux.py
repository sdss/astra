import os
from astra.database.targetdb import Target, CartonToTarget, Carton
from astra.database.astradb import Source
from tqdm import tqdm
from astra.utils import flatten, expand_path
from astra import __version__

from astra.database.astradb import DataProduct, Source
from sdss_access import SDSSPath
from tqdm import tqdm
from collections import OrderedDict

from astropy.table import MaskedColumn, Table

### TODO: These should be called 'allMWMStar files, allMWMVisit files, etc.

def all_star_table(
    release: str = "sdss5",
    apred: str = "1.0",
    run2d: str = "v6_0_9",
    upper_case_columns: bool = False,
    default_priority: int = 100_000,
    include_dr17: bool = False,
    **kwargs
) -> Table:
    """ Create a table of all stars that match the given release. """

    # TODO: Include photometry / astrometry in this query?
    q = (
        Carton
        .select(
            Target.catalogid, 
            Source.ra,
            Source.dec,
            Carton.carton, 
            CartonToTarget.priority
        )
        .join(CartonToTarget)
        .join(Target)
        .join(Source, on=(Target.catalogid == Source.catalogid))
        .join(DataProduct)
        .where(
            (DataProduct.release == release)
            &   (
                ((DataProduct.filetype == "apVisit") & (DataProduct.kwargs["apred"] == apred))
            |   ((DataProduct.filetype == "specFull") & (DataProduct.kwargs["run2d"] == run2d))
            )      
        )
        .tuples()
    )

    # Build up the cartons that each source is in, and their priorities.
    cartons = {}
    positions = {}
    for catalogid, ra, dec, carton, priority in tqdm(q, desc="Retrieving.."):
        cartons.setdefault(catalogid, {})
        positions.setdefault(catalogid, [ra, dec])
        cartons[catalogid][carton] = priority or default_priority

    # Create a table where each carton is a column.
    # We need the unique carton names.
    unique_carton_names = sorted(set(flatten([ea.keys() for ea in cartons.values()])))

    rows = []
    for catalogid, source_cartons in tqdm(cartons.items(), desc="Sorting.."):
        # get highest priority.
        carton_with_lowest_priority_value, *_ = sorted(source_cartons, key=lambda x: x[1])
        ra, dec = positions[catalogid]
        row = [
            catalogid,
            ra,
            dec,
            carton_with_lowest_priority_value,
            source_cartons[carton_with_lowest_priority_value]
        ]
        for carton in unique_carton_names:
            row.append(carton in source_cartons)
        rows.append(row)

    if include_dr17:
        q = (
            Source
            .select(
                Source.catalogid,
                Source.ra,
                Source.dec
            )
            .distinct(Source)
            .join(DataProduct)
            .where(
                (DataProduct.release == "dr17")
            )        
            .tuples()
        )
        for catalogid, ra, dec in q:
            if catalogid in positions:
                # This star is targeted in SDSS-V, and a row already exists.
                continue
            row = [catalogid, ra, dec, "", default_priority]
            row.extend([False] * len(unique_carton_names))
            rows.append(row)

    names = ["catalogid", "ra", "dec", "carton_0", "priority_0"] + unique_carton_names
    return Table(rows=rows, names=names)



def all_visit_table(
    release: str = "sdss5", 
    apred: str = "1.0", 
    run2d: str = "v6_0_9", 
    upper_case_columns: bool = False, 
    **kwargs
) -> Table:
    """ Create a table of all visits (e.g., apVisit or specFull files) that match the given release. """

    # TODO: include DR17 apVisit 

    q = (
        DataProduct
        .select()
        .where(
            (DataProduct.release == release)
        &   (
            ((DataProduct.filetype == "apVisit") & (DataProduct.kwargs["apred"] == apred))
        |   ((DataProduct.filetype == "specFull") & (DataProduct.kwargs["run2d"] == run2d))
            )
        )
    )

    all_keys = ["catalogid", "release", "filetype", "data_product_id"]
    translations = dict(
        catalogid="source",
        data_product_id="id"
    )
    default_values = dict(
        release="",
        filetype="",
        fiber=-1,
        mjd=-1,
        telescope="",
        field="",
        plate="",
        apred="",
        run2d="",
    )

    for filetype in ["apVisit", "specFull"]:
        new_keys = SDSSPath(release).lookup_keys(filetype)
        all_keys.extend([k for k in new_keys if k not in all_keys])

    data = OrderedDict([(translations.get(k, k), []) for k in all_keys])
    dtypes = {}
    mask = { translations.get(k, k): [] for k in all_keys }
    for result in tqdm(q):
        result.update(result.pop("kwargs"))
        for k in all_keys:
            key = translations.get(k, k)
            value = result.get(key, default_values.get(key, None))
            masked = not (key in result)

            data[key].append(value)
            mask[key].append(masked)

    translations = dict(data_product_id="id")

    columns = []
    for name, values in data.items():
        if upper_case_columns:
            name = name.upper()
        columns.append(
            MaskedColumn(
                values, 
                name=name, 
                mask=mask[key], 
                dtype=type(default_values.get(key, values[0]))
            )
        )
    all_visit_table = Table(columns)
    return all_visit_table



def all_carton_table(upper_case_columns: bool = False, default_priority: int = 100_000, **kwargs) -> Table:
    """
    Create a table that has one MWM star per row, with columns corresponding to all possible carton assignments.

    :param upper_case_columns: 
        If True, then the column names will be upper case.
    """

    q = (
        Carton
        .select(
            Target.catalogid, 
            Carton.carton, 
            CartonToTarget.priority
        )
        .join(CartonToTarget)
        .join(Target)
        .join(Source, on=(Target.catalogid == Source.catalogid))
        .tuples()
    )

    # Build up the cartons that each source is in, and their priorities.
    cartons = {}
    for catalogid, carton, priority in tqdm(q, desc="Retrieving.."):
        cartons.setdefault(catalogid, {})
        cartons[catalogid][carton] = priority or default_priority

    # Create a table where each carton is a column.
    # We need the unique carton names.
    unique_carton_names = sorted(set(flatten([ea.keys() for ea in cartons.values()])))

    rows = []
    for catalogid, source_cartons in tqdm(cartons.items(), desc="Sorting.."):
        # get highest priority.
        carton_with_lowest_priority_value, *_ = sorted(source_cartons, key=lambda x: x[1])
        row = [
            catalogid,
            carton_with_lowest_priority_value,
            source_cartons[carton_with_lowest_priority_value]
        ]
        for carton in unique_carton_names:
            row.append(carton in source_cartons)
        rows.append(row)

    names = ["catalogid", "carton_0", "priority_0"] + unique_carton_names
    return Table(rows=rows, names=names)


def write_all_carton_table(**kwargs):
    path = expand_path(f"$MWM_ASTRA/{__version__}/allCartons.fits")
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    table = all_carton_table(**kwargs)
    table.write(path, overwrite=True)


def write_all_visit_table(release: str = "sdss5", apred: str = "1.0", run2d: str = "v6_0_9", **kwargs):
    path = expand_path(f"$MWM_ASTRA/{__version__}/{run2d}-{apred}/aux/allVisits-{run2d}-{apred}.fits")
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    table = all_visit_table(
        release=release,
        apred=apred,
        run2d=run2d,
        **kwargs
    )
    table.write(path, overwrite=True)


def write_all_star_table(release: str = "sdss5", apred: str = "1.0", run2d: str = "v6_0_9", **kwargs):
    path = expand_path(f"$MWM_ASTRA/{__version__}/{run2d}-{apred}/aux/allStars-{run2d}-{apred}.fits")
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    table = all_star_table(
        release=release,
        apred=apred,
        run2d=run2d,
        **kwargs
    )
    table.write(path, overwrite=True)


