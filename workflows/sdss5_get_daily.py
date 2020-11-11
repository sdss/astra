
import astra
from astra.utils import batcher
from astra.tasks.io import ApStarFile, ApVisitFile
from astra.tasks.daily import (get_visits, get_stars)


mjd = 59163

get_stars_task = ApStarFile(**batcher(get_stars(mjd=mjd)))
get_visits_task = ApVisitFile(**batcher(get_visits(mjd=mjd)))

"""
astra.build(
    [get_stars_task, get_visits_task],
    local_scheduler=True
)
"""

tasks = [
    ("stars", get_stars_task),
    ("visits", get_visits_task)
]

missing = {}

for description, task in tasks:

    missing[description] = []

    for (sub_task, output) in zip(task.get_batch_tasks(), task.output()):
        if not output.exists():
            missing[description].append([
                sub_task.remote_path,
                { k: getattr(sub_task, k) for k in sub_task.batch_param_names() }
            ])


print(f"Report for MJD {mjd}")
for description, items in missing.items():
    print(f"There are {len(items)} missing {description}")
    if len(items) > 0:
        for remote_path, kwds in items:
            print(f"\t{kwds}\n\t\t{remote_path}")