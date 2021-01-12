

import astra
from astra.utils import batcher, get_default
from astra.tasks.io import ApStarFile, ApVisitFile
from astra.tasks.daily import (GetDailyApStarFiles, get_visits, get_visits_given_star, get_stars)
import astra.contrib.apogeenet.tasks as apogeenet
import astra.contrib.ferre.tasks.aspcap as aspcap
import astra.contrib.classifier.tasks.test as classifier
# TODO: revise
import astra.contrib.thecannon.tasks.sdss5 as thecannon
import astra.contrib.thepayne.tasks as thepayne
import astra.contrib.wd.tasks as wd

from astra.tasks.targets import DatabaseTarget
from sqlalchemy import Boolean, Column
from tqdm import tqdm


mjds = [
  59146,
  59159,
  59163,
  59164,
  59165,
  59166,
  59167,
  59168,
  59169,
  59186,
  59187,
  59188,
  59189,
  59190,
  59191,
  59192,
  59193,
  59195,
  59198,
  59199,
  59200,
  59201,
  59202,
  59203,
  59204,
  59205,
  59206,
  59207,
  59208,
  59209,
  59210,
  59211,
  59212,
  59214,
  59215,
  59216,
  59127,
]

missing_star_paths = []
missing_visit_paths = []

for mjd in mjds:
    print(f"Checking MJD {mjd}:")
    for star_kwds in tqdm(list(get_stars(mjd=mjd))):
        star = ApStarFile(**star_kwds)

        if not star.complete():
            print(f"{star} does not exist at {star.local_path}")
            missing_star_paths.append(star.local_path)

        missing_visits = []
        for visit_kwds in get_visits_given_star(star_kwds["obj"], star_kwds["apred"]):
            visit = ApVisitFile(**visit_kwds)
            if not visit.complete():
                missing_visits.append(visit)
                missing_visit_paths.append(missing_visit_paths.local_path)
        
        if missing_visits:
            print(f"Missing these visits for star {star}:")
            print("\n".join([f"\t{visit} does not exist at {visit.local_path}" for visit in missing_visits]))
        