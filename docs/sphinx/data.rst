
.. _data:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Data

:tocdepth: 2

.. rubric:: :header_no_toc:`Data`

Astra assumes that 1D extracted spectra are made available upstream, and those 
reduced data products are transferred to a location where Astra is run. Astra
can be installed and run locally, but depending on the `components <components>`_
you want to use, you may require a local copy of SDSS data (or model grids)
to do anything useful.


If you just want to execute a component in Astra then you should read the
documentation on `components <components>`_. This section describes how Astra
monitors folders for new reduced data products as part of the `continuous data
analysis <#>`_ mode.


Monitoring folders
==================

Watch a folder
^^^^^^^^^^^^^^

Astra can mointor specified folders for new data files. When new files appear
Astra records the path and checksum of those files in a database. The type of
data (e.g., the data model) is identified from the path, and Astra will check
to see which active components are capable of analysing those data. If there
are more than one components that are capable of analysing the data then Astra
will schedule those components in accordance with their `execution order
<components>`_.

To watch a folder you can use the ``astra`` command line tool::

    ~$ astra folder watch --help
    Usage: astra folder watch [OPTIONS] PATH

      Start monitoring a  folder for new data.

    Options:
      -r, --recursive              Monitor recursively (default: `False`).
      --interval INTEGER           Number of seconds to wait between checking for
                                   new data (default: `3600`).
      --regex-match-pattern TEXT   A regular expression pattern that, if given,
                                   only files that match this pattern will be
                                   acted upon.
      --regex-ignore-pattern TEXT  A regular expression pattern that, when
                                   matched, will ignore files in the watched
                                   folder. If both --regex-ignore-pattern and
                                   --regex-match-pattern are given, then any paths
                                   that match *both* will be ignored.
      --help                       Show this message and exit.


For example, to recursively check a folder for new data products every hour::

    astra folder watch /path/to/sdss/data -r --interval 3600


Refresh a folder
^^^^^^^^^^^^^^^^

If you don't want to wait the ``--interval`` number of seconds before checks,
you can force a refresh of a single folder by doing::

    astra folder refresh /path/to/sdss/data

And for all folders with::

    astra folder refresh


Stop watching a folder
^^^^^^^^^^^^^^^^^^^^^^

You can use the ``astra`` command line tool to stop watching a folder::

  astra folder unwatch /path/to/SDSS/data




Subsets
=======

When a component is added to Astra it can be configured to run on all available
data, only on new data, or on some subset of the data. For example, it is
convenient to be able to run **all** components on the same subset of the data
(e.g., like a calibration set). For this reason, Astra allows you to create
named subsets of the data which can be specified either by:

- a list of data paths, and/or
- a regular expression pattern that matches against all existing data, and all future data.

A single data product can be part of many subsets; data products have a many-to-many
relationship with subsets. To create a new subset you can use the 
``astra subset create`` function::

    ~$ astra subset create --help
    Usage: astra subset create [OPTIONS]

      Create a subset from the available data products. A subset can be created
      from data paths,  and/or by providing a regular expression pattern to
      match against data paths.

    Options:
      --data-paths TEXT           Supply data paths that will form this subset.
      --regex-match-pattern TEXT  Supply a regular expression pattern to match
                                  against all data paths.
      --name TEXT                 Provide a name for this subset.
      --visible                   Make this subset visible when users search for
                                  subsets.
      --auto-update               Automatically update this subset when new data
                                  are available. This is only relevant for subsets
                                  that have a regular expression pattern to match
                                  against.
      --raise-on-unrecognised     Raise an exception if a data path is given that
                                  is not recognised in the database. If this flag
                                  is not given then the unrecognised data paths
                                  will be added to the database.
      --help                      Show this message and exit.


For example, to create a new subset called "calibration-set" based on all the 
FITS files in the current working directory::

    astra subset create calibration-set --data-paths *.fits


