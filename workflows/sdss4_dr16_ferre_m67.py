
import astra
import astra.contrib.ferre.tasks.sdss4_refactor as aspcap
from astra.contrib.ferre.tasks.aspcap import doppler_estimate_in_bounds_factory
from astra.database import session, automap_table
from astra.utils import batcher
from tqdm import tqdm

# We want to load the SDSS DR16 APOGEE star table from the catalogdb.
release = "dr16"
apogeestar = automap_table("catalogdb", f"sdss_{release}_apogeestar")

# To analyse a SDSS-IV ApStar object we need:
#   > from astra.tasks.io.sdss4 import ApStarFile
#   > print(ApStarFile.batch_param_names())
#   > ['apstar', 'telescope', 'prefix', 'apred', 'field', 'obj']

# Select M67 sources.
q = session.query(*(
        apogeestar.c.apstar_version,    # apstar
        apogeestar.c.telescope,
        apogeestar.c.field,
        apogeestar.c.apogee_id,         # obj
        apogeestar.c.file,              # For inferring `prefix` and `apred`
    )).filter(apogeestar.c.field.like("%M67%"))

stars = []
for (apstar, telescope, field, obj, file) in q.all():

    prefix, apred = (file.strip()[:2], file.split("-")[1])

    stars.append({
        "apred": apred,
        "apstar": apstar,
        "prefix": prefix,
        "telescope": telescope,
        "field": field,
        "obj": obj,
    })

in_bounds = doppler_estimate_in_bounds_factory(release=release, public=True, mirror=False)

# We will only run things that have an initial estimate of stellar parameters
# (from cross-correlation) that is within the bounds of our grid.
stars = list(filter(in_bounds, tqdm(stars, desc="Checking initial guess")))

# Batch together and remove any duplicates.
stars = batcher(stars, unique=True)

# Let's do the stellar parameters in one run, and then abundances.
task_sp = aspcap.EstimateStellarParametersGivenSDSS4ApStarFile(release=release, **stars)

astra.build(
    [task_sp],
    workers=4,
    local_scheduler=True
)