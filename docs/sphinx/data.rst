
.. _data:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Data flow

:tocdepth: 2

.. rubric:: :header_no_toc:`Data flow`

It is assumed that 1D extracted spectra are available 'upstream' and those data
are transferred to a location where Astra is run (e.g., Utah). Astra can be 
installed and run locally, but you will need access to SDSS data in order to do 
anything useful. 

Astra will monitor the specified folders for new data files, and will maintain
a database that records the data files present in each directory in order to
identify new or updated files. When new data are detected, Astra will schedule
all *active* components to be executed.

If a new component is added to Astra it can be set to either run on all
available data, only on new data, or some named subset of the data. Named subsets
will be documented in more detail later, but in short having named subsets 
will allow for multiple components to run on something like 'calibration sets'
(or 'training sets', etc).


Once new observations have been processed by all active components, they will
only be re-processed once a new component is added.
