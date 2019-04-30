.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Astra

:tocdepth: 2

.. rubric:: :header_no_toc:`Astra`

Astra is the analysis framework for the Sloan Digital Sky Survey (SDSS V) Milky
Way Mapper. The purpose of Astra is to manage the analysis of reduced data
products from SDSS V and to streamline data releases.


Motivation
==========

Milky Way Mapper will produce data products that differ from previous generations
of SDSS. High-resolution infrared spectra will continue to be acquired using the
APOGEE spectrographs, but this will be complemented with low-resolution 
optical spectra. A wide range of scientific targets (e.g., red giants, main-sequence
exoplanet host stars, white dwarfs) will be observed by both instruments.
This change in strategy requires a framework to manage the analysis of targets, 
to encourage improvements to existing analysis approaches, and to streamline 
the data release process. 


Overview
========

Astra monitors specific folders for new reduced data products, schedules 
analysis tasks to be executed using those data products, and stores outputs from
those tasks in a database. This allows the collaboration to easily evaluate results 
in the context of previous sets of results, and ultimately to streamline the
data release process. All book-keeping (e.g., version control, data and input 
file provenance) is managed by Astra. 


Installation
============

Using ``sdss_install``
^^^^^^^^^^^^^^^^^^^^^^

Astra can be run locally, but in production mode it is intended to be run on SDSS clusters at the
University of Utah. Astra is already available to all users on Utah. If you want to install a new
version of Astra at Utah, you can do so using `sdss_install`::

  sdss_install -G astra

.. note::
    Once installed, you can make Astra available by simply running the ``module load astra``
    command next time you log in.

If you are running locally and you have the ``sdss_install`` tool from the `SDSS organization on GitHub <https://github.com/sdss/sdss_install>`_
then you can use that to install Astra.


Development version
^^^^^^^^^^^^^^^^^^^

Alternatively, if you want to be on the bleeding edge and install the development version then you
can do so using::

  git clone git@github.com:sdss/astra.git
  cd astra/
  python setup.py install


Configuration
=============

The configuration for Astra is specified by the ``python/astra/etc/astra.yml`` file. The most
relevant terms in this file are those that relate to the database. For example, the following
settings in ``python/astra/etc/astra.yml`` would be suitable for running Astra locally, if you have
a PostgreSQL server running::

  database_config:
    host: localhost
    database: astra

If you don't have a local PostgreSQL server, then you can explicitly specify a database "connection
string" for any kind of database. For example, a SQLite database is lightweight and suitable for 
local testing. The configuration for a local SQLite database might look something like::

  database_config:
    connection_string: sqlite:///astra.db


Setup
=====

Once you have installed and configured Astra, you should be able to immediately start using the
``astra`` command line tool. However, the first time you use this tool should be to run::

  astra setup

Which sets up the database and folder structure. You only need to do this once. 


Usage
=====

When testing Astra locally, the typical use case for adding a component might be something like::

  astra component add [MY_COMPONENT_NAME]

and if you want to run that component on some data, you can do so by running::

  astra execute [MY_COMPONENT_NAME] [INPUT_DATA]

which, by default, will execute the component in the current working directory. In the background
there is a little more going on. Specifically, in the first command Astra has done the following:

  - Fetched your component from a remote GitHub repository
  - Checked that there are no requirement conflicts with that component repository
  - Built your component and made the executables available on your `$PATH`
  - Installed your component using Modules

And in the second line:

  - Added the `[INPUT_PATHS]` to a database so that results between components can be tracked and
    compared.
  - Created a Data Subset using those `[INPUT_PATHS]` so that the same subset can be executed using
    other components in the future.
  - Made your component available using Modules
  - Executed the specified component
  - [In future]: Parsed the outputs from the component and recoreded them in a database, where all
    results are defined by having declrarative data models
  - Un-loaded your component

This abstraction has the following advantages:

  - Track results from the same component between different versions
  - Compare results from different components analysing the same data
  - Re-use subsets of data for calibration or scientific verification purposes
  - etc



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. [#] Astra can be run locally, but you would need a local mirror of SDSS
       reduced data products to do anything useful.
