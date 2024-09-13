from astra.models import (ASPCAP, ApogeeNet, AstroNN, AstroNNdist, BossNet, Corv, LineForest, MDwarfType, Slam, SnowWhite, TheCannon, ThePayne)
from astra.models.mwm import BossVisitSpectrum, BossCombinedSpectrum
from astra.products.mwm_summary import (create_mwm_targets_product, create_mwm_all_star_product, create_mwm_all_visit_product)
from astra.products.pipeline_summary import (create_astra_all_star_product, create_astra_all_visit_product)
from astra.utils import log

from astra.models import ApogeeCoaddedSpectrumInApStar, ApogeeVisitSpectrumInApStar, ApogeeVisitSpectrum


log.info("mwmTargets")
create_mwm_targets_product(overwrite=True)

log.info("mwmAllStar")
create_mwm_all_star_product(
    overwrite=True,
    apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")),
    boss_where=(BossCombinedSpectrum.run2d == "v6_1_3"),
)

log.info("mwmAllVisit")
create_mwm_all_visit_product(
    overwrite=True,
    apogee_where=ApogeeVisitSpectrum.apred.in_(("dr17", "1.3")),
    boss_where=(BossVisitSpectrum.run2d == "v6_1_3"),
)


star_level_models = (
    (ASPCAP, dict(apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (ApogeeNet, dict(apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (AstroNN, dict(apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (AstroNNdist, dict(apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (BossNet, dict(boss_spectrum_model=BossCombinedSpectrum, boss_where=(BossCombinedSpectrum.run2d == "v6_1_3"))),
    (TheCannon, dict(apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (ThePayne, dict(apogee_where=ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (LineForest, dict(boss_spectrum_model=BossCombinedSpectrum, boss_where=(BossCombinedSpectrum.run2d == "v6_1_3"))),
    (MDwarfType, dict(boss_spectrum_model=BossCombinedSpectrum, boss_where=(BossCombinedSpectrum.run2d == "v6_1_3"))),
    (SnowWhite, dict(boss_spectrum_model=BossCombinedSpectrum, boss_where=(BossCombinedSpectrum.run2d == "v6_1_3"))),
    (Corv, dict(boss_spectrum_model=BossCombinedSpectrum, boss_where=(BossCombinedSpectrum.run2d == "v6_1_3"))),
    (Slam, dict(boss_spectrum_model=BossCombinedSpectrum)),
)


for model, kwargs in star_level_models:
    log.info(f"{model.__name__} star level")
    create_astra_all_star_product(model, overwrite=True, **kwargs)


visit_level_models = (
    (AstroNN, dict(apogee_spectrum_model=ApogeeVisitSpectrumInApStar,apogee_where=ApogeeVisitSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (BossNet, dict(boss_spectrum_model=BossVisitSpectrum, boss_where=(BossVisitSpectrum.run2d == "v6_1_3"))),
    (Corv, dict(boss_spectrum_model=BossVisitSpectrum, boss_where=(BossVisitSpectrum.run2d == "v6_1_3"))),
    (LineForest, dict(boss_spectrum_model=BossVisitSpectrum, boss_where=(BossVisitSpectrum.run2d == "v6_1_3"))),
    (ThePayne, dict(apogee_spectrum_model=ApogeeVisitSpectrumInApStar, apogee_where=ApogeeVisitSpectrumInApStar.apred.in_(("dr17", "1.3")))),
    (SnowWhite, dict(boss_spectrum_model=BossVisitSpectrum, boss_where=(BossVisitSpectrum.run2d == "v6_1_3"))),    
)

for model, kwargs in visit_level_models:
    log.info(f"{model.__name__} visit level")
    create_astra_all_visit_product(model, overwrite=True, **kwargs)
