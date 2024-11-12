import pytest
import peewee
import os
os.environ["ASTRA_DATABASE_PATH"] = ":memory:"

def test_task_query_builder_with_group_by():
    from astra import task, generate_queries_for_task
    from astra.models.source import Source
    from astra.models.spectrum import Spectrum
    from astra.models.apogee import ApogeeVisitSpectrum
    from astra.models.pipeline import PipelineOutputModel
    from astra.fields import BooleanField
    from datetime import datetime
    from typing import Iterable



    class ApogeeVisitState(PipelineOutputModel):
        used = BooleanField()

    for model in (ApogeeVisitState, Source, Spectrum, ApogeeVisitSpectrum):
        model.create_table()

    for n in range(3):
        Source.create()
    for n in range(4):
        Spectrum.create()


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



    s1 = ApogeeVisitSpectrum.create(spectrum_pk=1, source_pk=1, release="test", apred="apred", plate="plate", telescope="apo", fiber=0, mjd=0, field="field", prefix="ap")
    s2 = ApogeeVisitSpectrum.create(spectrum_pk=2, source_pk=1, release="test", apred="apred", plate="plate", telescope="apo", fiber=1, mjd=0, field="field", prefix="ap")
    s3 = ApogeeVisitSpectrum.create(spectrum_pk=3, source_pk=2, release="test", apred="apred", plate="plate", telescope="lco", fiber=1, mjd=0, field="field", prefix="ap")


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

    s4 = ApogeeVisitSpectrum.create(spectrum_pk=4, source_pk=3, release="test", apred="apred", plate="plate", telescope="apo", fiber=12, mjd=1, field="field", prefix="ap")
    _, q = next(generate_queries_for_task(make_stack))
    assert q.count() == 2
    q = list(q)
    assert s4 in q
    assert s1 in q





