from tqdm import tqdm
from astra import __version__
from astra.models.base import database
from astra.models import ASPCAP, ApogeeCoaddedSpectrumInApStar
from peewee import chunked

# Get all DR17 results for v_astra = 0.5.0
q = (
    ASPCAP
    .select()
    .distinct(ASPCAP.spectrum_pk)
    .join(
        ApogeeCoaddedSpectrumInApStar, 
        on=(ApogeeCoaddedSpectrumInApStar.spectrum_pk == ASPCAP.spectrum_pk)
    )
    .where(
        (ASPCAP.v_astra == "0.5.0")
    &   (ApogeeCoaddedSpectrumInApStar.release == "dr17")
    )
    .dicts()
)

rows = []
for r in tqdm(q):
    r.pop("task_pk")
    r["v_astra"] = __version__
    # Remove any calibrations.
    for k, v in r.items():
        if k.startswith("raw_"):
            r[k[4:]] = v
    rows.append(r)

spectrum_pks = [e["spectrum_pk"] for e in rows]

# Check to make sure we are not duplicating...
q = (
    ASPCAP
    .select()
    .where(
        (ASPCAP.v_astra == __version__)
    &   ASPCAP.spectrum_pk.in_(spectrum_pks)
    )
    .first()
)
assert q is None

# Bulk insert
batch_size = 1000
with database.atomic():
    with tqdm(desc="Inserting", total=len(rows)) as pb:
        for chunk in chunked(rows, batch_size):
            (
                ASPCAP
                .insert_many(chunk)
                .execute()
            )
            pb.update(min(batch_size, len(chunk)))
            pb.refresh()