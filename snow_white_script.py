from astra import __version__
from astra.pipelines.snow_white import snow_white
from tqdm import tqdm
from astra.models import BossCombinedSpectrum, Source, SnowWhite, BossRestFrameVisitSpectrum
from peewee import JOIN

q = (
    BossCombinedSpectrum
    .select()
    .join(Source)
    .switch(BossCombinedSpectrum)
    .join(
        SnowWhite, 
        JOIN.LEFT_OUTER, 
        on=(
            (SnowWhite.spectrum_pk == BossCombinedSpectrum.spectrum_pk)
        &   (SnowWhite.v_astra == __version__)        
        )
    )
    .where(
        Source.assigned_to_program("mwm_wd")
    &   SnowWhite.spectrum_pk.is_null()
    )
)

for item in tqdm(snow_white(q), total=1):
    None

q = (
    BossRestFrameVisitSpectrum 
    .select()
    .join(Source)
    .switch(BossRestFrameVisitSpectrum)
    .join(
        SnowWhite, 
        JOIN.LEFT_OUTER, 
        on=(
            (SnowWhite.spectrum_pk == BossRestFrameVisitSpectrum.spectrum_pk)
        &   (SnowWhite.v_astra == __version__)
        )
    )
    .where(
        Source.assigned_to_program("mwm_wd")
    &   SnowWhite.spectrum_pk.is_null()
    )
)

for item in tqdm(snow_white(q), total=1):
    None
