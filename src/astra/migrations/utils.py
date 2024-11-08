from peewee import chunked
from astra.utils import flatten
from astra.models.base import database
from astra.models.spectrum import Spectrum

class NoQueue:
    def put(self, kwargs):
        pass

def generate_new_spectrum_pks(N, batch_size=100):
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Assigning spectrum identifiers", unit="spectra", total=N) as pb:
            for chunk in chunked([{"spectrum_flags": 0}] * N, batch_size):                
                yield from flatten(
                    Spectrum
                    .insert_many(chunk)
                    .returning(Spectrum.pk)
                    .tuples()
                    .execute()
                )
                pb.update(min(batch_size, len(chunk)))
                pb.refresh()


def enumerate_new_spectrum_pks(iter, batch_size=100):
    N = len(iter)

    with database.atomic():
        for chunk, batch in zip(chunked(iter, batch_size), chunked([{"spectrum_flags": 0}] * len(iter), batch_size)):
            spectrum_pks = flatten(
                Spectrum
                .insert_many(batch)
                .returning(Spectrum.pk)
                .tuples()
                .execute()
            )
            for spectrum_pk, item in zip(spectrum_pks, chunk):
                yield (spectrum_pk, item)


def upsert_many(model, returning, data, batch_size, queue, description):
    returned = []
    with database.atomic():
        queue.put(dict(description=description, total=len(data), completed=0))
        for chunk in chunked(data, batch_size):
            returned.extend(
                flatten(
                    model
                    .insert_many(chunk)
                    .on_conflict_ignore()
                    .returning(returning)
                    .tuples()
                    .execute()
                )
            )
            n = min(batch_size, len(chunk))
            queue.put(dict(advance=n))

    return tuple(returned)