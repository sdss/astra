from astra.pipelines.corv import corv
from tqdm import tqdm

for item in tqdm(corv(max_workers=32), total=0):
    None
