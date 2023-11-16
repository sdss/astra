from astra.models.apogeenet import ApogeeNet
from astra.models.apogee import (
    ApogeeCoaddedSpectrumInApStar,
    ApogeeVisitSpectrum,
    ApogeeVisitSpectrumInApStar,
)
from astra.models.apogeenet_v2 import ApogeeNetV2
from astra.models.aspcap import ASPCAP
from astra.models.astronn import AstroNN
from astra.models.base import BaseModel
from astra.models.bossnet import BossNet
from astra.models.boss import BossVisitSpectrum
from astra.models.classifier import SpectrumClassification
from astra.models.corv import Corv
from astra.models.ferre import (
    FerreChemicalAbundances,
    FerreCoarse,
    FerreStellarParameters,
)
from astra.models.line_forest import LineForest
from astra.models.mdwarftype import MDwarfType
from astra.models.slam import Slam
from astra.models.snow_white import SnowWhite
from astra.models.source import Source
from astra.models.spectrum import Spectrum, SpectrumMixin
from astra.models.the_payne import ThePayne
from astra.models.the_cannon import TheCannon
from astra.models.hot_payne import HotPayne
from astra.models.madgics import ApogeeMADGICSVisitSpectrum
from astra.models.mwm import (
    BossCombinedSpectrum,
    BossRestFrameVisitSpectrum,
    ApogeeCombinedSpectrum,
    ApogeeRestFrameVisitSpectrum,
)
