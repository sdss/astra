from peewee import chunked
from astra.utils import flatten
from astra.models.base import database
from astra.models.spectrum import Spectrum
from tqdm import tqdm

def generate_new_spectrum_pks(N, batch_size=100):
    with database.atomic():
        # Need to chunk this to avoid SQLite limits.
        with tqdm(desc="Assigning spectrum identifiers", unit="spectra", total=N) as pb:
            for chunk in chunked([{"spectrum_type_flags": 0}] * N, batch_size):                
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
        for chunk, batch in zip(chunked(iter, batch_size), chunked([{"spectrum_type_flags": 0}] * len(iter), batch_size)):
            spectrum_pks = flatten(
                Spectrum
                .insert_many(batch)
                .returning(Spectrum.pk)
                .tuples()
                .execute()
            )
            for spectrum_pk, item in zip(spectrum_pks, chunk):
                yield (spectrum_pk, item)
