"""Registry of which `astraStar`/`astraVisit` product levels each pipeline supports."""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class PipelineSpec:
    #: Dotted pipeline model path (e.g. `"aspcap.ASPCAP"`), resolved via `astra.products.utils.resolve_model`.
    model: str
    #: Whether this pipeline produces an `astraStar<PIPELINE>` (co-added/star-level) product.
    create_star: bool = True
    #: Whether this pipeline produces an `astraVisit<PIPELINE>` (visit-level) product.
    create_visit: bool = True
    #: Optional callable `where(pipeline_model) -> peewee expression` restricting which
    #: pipeline result rows count as a usable result for this pipeline.
    where: Optional[Callable] = None


def _aspcap_where(model):
    # Sources with no FERRE result raise inside `_create_pipeline_product` if included.
    return model.ferre_index.is_null(False)


PIPELINES = {
    "ASPCAP": PipelineSpec("aspcap.ASPCAP", where=_aspcap_where, create_visit=False),
    "ApogeeNet": PipelineSpec("apogeenet.ApogeeNet"),
    "BossNet": PipelineSpec("bossnet.BossNet"),
    "LineForest": PipelineSpec("line_forest.LineForest"),
    "AstroNN": PipelineSpec("astronn.AstroNN"),
    "AstroNNdist": PipelineSpec("astronn_dist.AstroNNdist", create_visit=False),
    "Slam": PipelineSpec("slam.Slam", create_visit=False),
    "MDwarfType": PipelineSpec("mdwarftype.MDwarfType"),
    "Corv": PipelineSpec("corv.Corv", create_star=False),
    "SnowWhite": PipelineSpec("snow_white.SnowWhite", create_visit=False),
    "ThePayne": PipelineSpec("the_payne.ThePayne", create_visit=False),
}
