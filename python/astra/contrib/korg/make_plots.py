import numpy as np

seconds = np.loadtxt("seconds")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(seconds, bins=50)
ax.set_xlabel("time elapsed [s]")
fig.tight_layout()
fig.savefig("seconds.png", dpi=300)


with open("results", "r") as fp:
    lines = fp.readlines()

p0 = np.nan * np.ones((5000, 5))
multilocal = np.nan * np.ones((5000, 5))
global_x = np.nan * np.ones((5000, 5))

k = 0
j = -1
for i, line in enumerate(lines):
    if not line.strip("-\n"): continue
    if "multilocal" in line or "Progress" in line:
        k = 0
        j += 1
        continue

    words = line.split()
    p0_ = float(words[3])
    multilocal_ = float(words[4])
    global_x_ = float(words[5].rstrip("windows:"))
    p0[j, k] = p0_
    multilocal[j, k] = multilocal_
    global_x[j, k] = global_x_

    k += 1

from astropy.table import Table
label_names = "teff logg fe_h alpha_fe c_fe".split()

data = dict(zip([f"p0_{ln}" for ln in label_names], p0.T))
data.update(
    dict(
        zip(
            [f"multilocal_{ln}" for ln in label_names],
            multilocal.T
        )
    )
)
data.update(
    dict(
        zip(
            [f"global_{ln}" for ln in label_names],
            global_x.T
        )
    )
)
t = Table(data=data)

keep = p0[:, 0] >= 4000

#t.write("results.csv")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat[:5]):
    multilocal_diff = multilocal[:, i] - p0[:, i]
    global_diff = global_x[:, i] - p0[:, i]
    #ax.scatter(p0[:, i], multilocal[:, i], c="tab:red", s=5, alpha=0.25, label="multilocal")
    ax.scatter(p0[keep, i], global_x[keep, i], c="tab:blue", s=5, alpha=0.25, label="global")
    mu_multilocal = np.nanmean(multilocal_diff[keep])
    std_multilocal = np.nanstd(multilocal_diff[keep])
    mu_global = np.nanmean(global_diff[keep])
    std_global = np.nanstd(global_diff[keep])
    ax.set_title(f"{mu_multilocal:.2f} +/- {std_multilocal:.2f} (multilocal) vs {mu_global:.2f} +/- {std_global:.2f} (global)")

all_limits = dict(
    teff=(4000, 6000),
    alpha_fe=(-0.5, 0.5)
)
for i, ax in enumerate(axes.flat[:5]):
    limits = all_limits.get(label_names[i], None)
    if limits is None:        
        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
        limits = limits.flatten()
        limits = (np.min(limits), np.max(limits))
    
    ax.plot(limits, limits, c="#666666", ls=":", zorder=-1, lw=0.5)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(f"p0 {label_names[i]}")
    
for value in np.arange(0, 4, 0.5):
    axes.flat[1].axhline(value, c="#666666", ls=":", zorder=-1)

axes.flat[0].legend()
fig.tight_layout()
fig.savefig("p0_global_restricted.png", dpi=300)