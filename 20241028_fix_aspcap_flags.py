import numpy as np
from astra.models import ASPCAP

# There is a higher than expected density of the following two flags:
#flag_v_sini_grid_edge_warn = result_flags.flag(2**11, help_text="v_sini is within one step from the highest grid edge")
#flag_v_sini_grid_edge_bad = result_flags.flag(2**12, help_text="v_sini is within 1/8th of a step from the highest grid edge")


# So, this will return lots of things:
(
    ASPCAP
    .select()
    .where(
        ASPCAP.flag_v_sini_grid_edge_bad
    &   (ASPCAP.v_sini == np.nan)
    )
    .count()
)

# But this will return 0:
(
    ASPCAP
    .select()
    .where(
        ASPCAP.flag_v_sini_grid_edge_bad
    &   (ASPCAP.v_sini.is_null())
    )
    .count()
)

# So, let's clear the flags where vsini is NaN
(
    ASPCAP
    .update(result_flags=ASPCAP.flag_v_sini_grid_edge_warn.clear())
    .where(ASPCAP.v_sini == np.nan)
    .execute()
)

(
    ASPCAP
    .update(result_flags=ASPCAP.flag_v_sini_grid_edge_bad.clear())
    .where(ASPCAP.v_sini == np.nan)
    .execute()
)