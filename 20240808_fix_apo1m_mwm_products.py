from astra.products.mwm import create_mwmVisit_and_mwmStar_products
from astra.models import ApogeeCoaddedSpectrumInApStar, Source

from tqdm import tqdm

s = (
    Source
    .select()
    .join(ApogeeCoaddedSpectrumInApStar, on=(ApogeeCoaddedSpectrumInApStar.source_pk == Source.pk))
    .where(ApogeeCoaddedSpectrumInApStar.telescope == "apo1m")
)

for source in tqdm(s):
    create_mwmVisit_and_mwmStar_products(source, apreds=("dr17", "1.3"), run2ds=("v6_1_3", ), overwrite=True) 

