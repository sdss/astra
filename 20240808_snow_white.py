from astra import __version__
from astra.models.boss import BossVisitSpectrum
from astra.models.mwm import BossCombinedSpectrum, BossRestFrameVisitSpectrum
from astra.pipelines.snow_white import snow_white
from astra.models import Source, SnowWhite
from peewee import JOIN
from tqdm import tqdm

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
    &   (BossCombinedSpectrum.run2d == "v6_1_3")
    )        
)

for item in tqdm(snow_white(q), total=1):
    None

q = (
    BossVisitSpectrum
    .select()
    .join(Source)
    .switch(BossVisitSpectrum)
    .join(
        SnowWhite, 
        JOIN.LEFT_OUTER, 
        on=(
            (SnowWhite.spectrum_pk == BossVisitSpectrum.spectrum_pk)
        &   (SnowWhite.v_astra == __version__)
        )
    )
    .where(
        Source.assigned_to_program("mwm_wd")
    &   SnowWhite.spectrum_pk.is_null()
    &   (BossVisitSpectrum.run2d == "v6_1_3")
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
    &   (BossRestFrameVisitSpectrum.run2d == "v6_1_3")
    )        
)
for item in tqdm(snow_white(q), total=1):
    None

