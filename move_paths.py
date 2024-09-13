import os
from glob import glob
from tqdm import tqdm
from shutil import rmtree
k = 100
base_dir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/astra/0.2.5/v6_0_9-daily/spectra/"

for prefix in ("star", "visit"):
    for old_path in tqdm(glob(f"{base_dir}{prefix}/*/*/*.fits"), desc=prefix):
        catalogid = int(old_path.split("-")[-1].split(".")[0])
        new_path = f"{(catalogid // k) % k:0>2.0f}/{catalogid % k:0>2.0f}/{os.path.basename(old_path)}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(old_path, new_path)

# remove empty directories
for prefix in ("star", "visit"):
    single_col = glob(f"{base_dir}{prefix}/?/*/*.fits")
    assert len(single_col) == 0
    for empty_dir in glob(f"{base_dir}{prefix}/?"):
        print(f"removing {empty_dir}")
        rmtree(empty_dir)
    triple_col = glob(f"{base_dir}{prefix}/???/*/*.fits")
    assert len(triple_col) == 0
    for empty_dir in glob(f"{base_dir}{prefix}/???"):
        print(f"removing {empty_dir}")
        rmtree(empty_dir)