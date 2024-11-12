import pytest
import peewee
import os
os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

def test_task_query_builder_with_group_by():
    from astra import task, generate_queries_for_task
    from astra.models.base import database
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.pipeline import PipelineOutputModel
    from astra.fields import BooleanField
    from datetime import datetime
    from typing import Iterable

    class ApogeeVisitState(PipelineOutputModel):
        used = BooleanField()

    models = (Source, Spectrum, ApogeeVisitSpectrum, ApogeeVisitState)
    database.create_tables(models)

    source_pks = [Source.create().pk for n in range(3)]
    spectrum_pks = [Spectrum.create().pk for n in range(4)]


    @task(group_by=("source_pk", "telescope"))
    def make_stack(spectra: Iterable[ApogeeVisitSpectrum]) -> Iterable[ApogeeVisitState]:
        for given in spectra:
            q = (
                ApogeeVisitSpectrum
                .select()
                .where(
                    (ApogeeVisitSpectrum.telescope == given.telescope)
                &   (ApogeeVisitSpectrum.source_pk == given.source_pk)
                )
            )
            for s in q:
                yield ApogeeVisitState.from_spectrum(s, used=True)


    ApogeeVisitSpectrum.delete().execute()
    
    s1 = ApogeeVisitSpectrum.create(spectrum_pk=spectrum_pks[1-1], source_pk=source_pks[1-1], release="test", apred="apred", plate="plate", telescope="apo", fiber=0, mjd=0, field="field", prefix="ap")
    s2 = ApogeeVisitSpectrum.create(spectrum_pk=spectrum_pks[2-1], source_pk=source_pks[1-1], release="test", apred="apred", plate="plate", telescope="apo", fiber=1, mjd=0, field="field", prefix="ap")
    s3 = ApogeeVisitSpectrum.create(spectrum_pk=spectrum_pks[3-1], source_pk=source_pks[2-1], release="test", apred="apred", plate="plate", telescope="lco", fiber=1, mjd=0, field="field", prefix="ap")


    _, q = next(generate_queries_for_task(make_stack))
    assert q.count() == 2
    assert s3 in list(q)

    # Run everything
    list(make_stack(q))

    _, q = next(generate_queries_for_task(make_stack))
    assert q.count() == 0

    # Now modify one of the original spectra
    s2.modified = datetime.now()
    s2.save()

    _, q = next(generate_queries_for_task(make_stack))
    assert q.count() == 1
    assert q.first() == s2

    s1.modified = datetime.now()
    s1.save()

    s4 = ApogeeVisitSpectrum.create(spectrum_pk=spectrum_pks[4-1], source_pk=source_pks[3-1], release="test", apred="apred", plate="plate", telescope="apo", fiber=12, mjd=1, field="field", prefix="ap")
    _, q = next(generate_queries_for_task(make_stack))
    assert q.count() == 2
    q = list(q)
    assert s4 in q
    assert s1 in q





