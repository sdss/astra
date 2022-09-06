.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Astra

:tocdepth: 0

.. rubric:: :header_no_toc:`Astra`

Astra is the analysis framework for the Sloan Digital Sky Survey (SDSS V) Milky
Way Mapper. The purpose of Astra is to manage the analysis of reduced data
products from SDSS V and to streamline data releases.


Milky Way Mapper will produce data products that differ from previous generations
of SDSS. High-resolution infrared spectra will continue to be acquired using the
APOGEE spectrographs, but this will be complemented with low-resolution
optical spectra. A wide range of scientific targets (e.g., red giants, main-sequence
exoplanet host stars, white dwarfs) will be observed by both instruments.
This change in strategy requires a framework to manage the analysis of targets,
to encourage improvements to existing analysis approaches, and to streamline
the data release process.


How it works
------------

Astra uses a directed acyclic graph to figure out which analysis tasks need to be run
on what observations. Astra may initiate many different analysis tasks, depending on
the kind of source (e.g., white dwarf, FGK-type star).
Since these analysis routines will likely improve with time, Astra stores all version information
and data provenance. This allows results to be reproducible, tracked, and compared
to previous sets of results.
