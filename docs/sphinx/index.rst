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


