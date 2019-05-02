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

Astra can be installed using the following terminal commands::

  git clone git@github.com:sdss/astra.git
  cd astra/
  python setup.py install

Alternatively, if you have the ``sdss_install`` package then you can use the one-liner::

  sdss_install -G astra

If you are running Astra on Utah then you only have to install Astra once, and you can make it
available by using the ``module load astra`` command next time you log in. If you have problems
running ``sdss_install`` then you may want to visit the `frequently encountered problems <#>`_
page to check that you have all the requisite environment variables set.



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


You can immediately start using the ``astra`` command line tool once you have installed and
configured Astra. The first time you run this tool should be to set up the database and folder
structure. You can do this using::

  astra setup

Now you're good to go! Next you may want to read about `components <components>`_ or check out the
`getting started guides <guides>`_.
