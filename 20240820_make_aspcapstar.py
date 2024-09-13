from astra.models import ASPCAP, ApogeeCoaddedSpectrumInApStar
from astra.products.pipeline import create_star_pipeline_products_for_all_sources

if __name__ == "__main__":
    import sys
    create_star_pipeline_products_for_all_sources(
        ASPCAP,
        apogee_where=(
            ApogeeCoaddedSpectrumInApStar.apred.in_(("dr17", "1.3"))
        ),
        page=int(sys.argv[1]),
        limit=int(sys.argv[2])
    )
