
.. _data:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Data

:tocdepth: 2

.. rubric:: :header_no_toc:`Data`

It is assumed that 1D extracted spectra are made available upstream and those 
reduced data products are transferred to a location where Astra is run. Astra 
can be installed and run locally, but you will need access to SDSS data in order 
to do anything useful. 

Watching folders
================

Astra will 'watch' specified folders for new data files, and will maintain
a database that records the data files present in each directory in order to
identify new or updated files. When new or updated data are detected, Astra will
schedule all *active* components to be executed. Once new observations have been 
processed by all active components, they will only be re-processed once a new 
component is added. To watch a folder you can use the ``astra`` shell 
utility::

  astra data watch -r /path/to/SDSS/data --interval 3600

[TBD: link to full docs for ``astra`` CLI]

Unwatching folders
==================

You can use the ``astra`` shell utility to stop watching a folder::

  astra data unwatch /path/to/SDSS/data


Force refresh
=============

If you want to forcibly refresh all watched folders instead of waiting until
the next interval, you can do so with::

  astra data refresh


Subsets
=======

If a new component is added to Astra it can be set to either run on all
available data, only on new data, or some named subset of the data. Named subsets
allow for multiple components to run on something like 'calibration sets'
(or 'science verification set', etc). Named subsets are available to any 
component through the functions in  ``astra.utils``.

Use the ``astra`` utility to create a named subset::

  astra subset create dr20.science-verification -i paths.txt

Data products can be part of many subsets (data products have a many-to-many 
relationship with subsets), and subsets can be created to automatically add 
new data products where the path matches a given regular expression pattern.

[TBD: example of updating/deleting a subset]

[TBD: example of creating a subset with just a regular expression (e.g., APOGEE 1M targets)]
