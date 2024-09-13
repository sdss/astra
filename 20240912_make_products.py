from astra.models import (ASPCAP, ApogeeNet, AstroNN, AstroNNdist, BossNet, Corv, LineForest, MDwarfType, Slam, SnowWhite, TheCannon, ThePayne)
from astra.models.mwm import BossVisitSpectrum, BossCombinedSpectrum
from astra.products.mwm_summary import (create_mwm_targets_product, create_mwm_all_star_product, create_mwm_all_visit_product)
from astra.utils import log


log.info("mwmTargets")
create_mwm_targets_product(overwrite=True)
log.info("mwmAllStar")
create_mwm_all_star_product(overwrite=True)
log.info("mwmAllVisit")
create_mwm_all_visit_product(overwrite=True)


star_level_models = (
    (ASPCAP, {}),
    (ApogeeNet, {}),
    (AstroNN, {}),
    (AstroNNdist, {}),
    (TheCannon, {}),
    (ThePayne, {}),
    (LineForest, dict(boss_spectrum_model=BossCombinedSpectrum)),
    (MDwarfType, dict(boss_spectrum_model=BossCombinedSpectrum)),
    (SnowWhite, dict(boss_spectrum_model=BossCombinedSpectrum)),
    (Corv, dict(boss_spectrum_model=BossCombinedSpectrum)),
    #(Slam, dict(boss_spectrum_model=BossCombinedSpectrum)),
)
visit_level_models = (
    (AstroNN, {}),
    (BossNet, {}),
    (Corv, dict(boss_spectrum_model=BossVisitSpectrum)),
    (LineForest, dict(boss_spectrum_model=BossVisitSpectrum)),
    (MDwarfType, dict(boss_spectrum_model=BossVisitSpectrum)),
    (SnowWhite, dict(boss_spectrum_model=BossVisitSpectrum)),
    (ThePayne, {}),
    #(Slam, dict(boss_spectrum_model=BossVisitSpectrum)),
)

for model, kwargs in star_level_models:
    log.info(f"{model.__name__} star level")
    create_mwm_all_star_product(pipeline_model=model, overwrite=True, **kwargs)
    raise a


for model, kwargs in visit_level_models:
    log.info(f"{model.__name__} visit level")
    create_mwm_all_visit_product(pipeline_model=model, overwrite=True, **kwargs)
