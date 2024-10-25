#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

#%%%% imports
# %pylab inline
# %load_ext autoreload
# %autoreload 2
#%reload_ext autoreload
%pylab
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from laspec import mpl
mpl.set_cham(15)

#%%%% 
from astropy import table
m9 = table.Table.read("/Users/cham/projects/sb2/sb2dc/data/pub/m9v1short.fits")


from laspec.lamost_kits import PubKit
code = PubKit.compress_table(m9, tbl_name="m9", reserved=("bjd", "ra", "dec"))
        

#%%

PubKit.modify_column(m9, colname="id", name="id", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="obsid", name="obsid", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="spid", name="spid", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="fiberid", name="fiberid", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="lmjm", name="lmjm", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="planid", name="planid", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="lmjd", name="lmjd", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="ra", name="ra", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=True, )
PubKit.modify_column(m9, colname="dec", name="dec", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=True, )
PubKit.modify_column(m9, colname="ra_obs", name="ra_obs", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=True, )
PubKit.modify_column(m9, colname="dec_obs", name="dec_obs", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=True, )
PubKit.modify_column(m9, colname="band_B", name="band_B", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="snr_B", name="snr_B", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="band_R", name="band_R", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="snr_R", name="snr_R", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="fibermask", name="fibermask", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="obsdate", name="obsdate", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="GroupID", name="GroupID", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )
PubKit.modify_column(m9, colname="GroupSize", name="GroupSize", description="", remove_mask=False, fill_value=None, remove_directly=False, reserved=False, )

