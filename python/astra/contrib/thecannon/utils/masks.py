import numpy as np


def create_mask(dispersion, regions):
    """
    Return a boolean mask given a structured list of (start, end) regions.

    :param dispersion:
        An array of dispersion values.

    :param regions:
        A list of two-length tuples containing the `(start, end)` points of a region.

    :returns:
        A boolean mask indicating if the pixels in the `dispersion` array are within the `regions`.
    """

    mask = np.zeros(dispersion.size, dtype=bool)

    if isinstance(regions[0], (int, float)):
        regions = [regions]

    for start, end in regions:
        start, end = (start or -np.inf, end or +np.inf)
        mask[(end >= dispersion) * (dispersion >= start)] = True

    return mask
