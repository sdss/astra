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

Milky Way Mapper will acquire data products that differ from previous generations
of SDSS. High-resolution infrared spectra will continue to be acquired using the
APOGEE spectrographs, but this will be complemented with low-resolution BOSS-like
optical spectra. A wide range of scientific targets (e.g., red giants, main-sequence
exoplanet host stars, white dwarfs) are expected to be observed in both wavelengths.
This change in strategy requires an analysis framework to manage the analysis of
targets, to encourage improvements to existing analysis approaches, and to
streamline the data release process. 


Conceptual overview
===================

* :ref:`Components <components>`
* :ref:`Data flow <data>`
* :ref:`Database design <database>`
* :ref:`Astra setup (for local/Utah/etc)`
* :ref:`Astra command line tools`
* :ref:`Astra utilities`


Roadmap and priorities
======================

1. Documentation describing the overview of Astra, conceptual pieces, and how
   everything is expected to flow together

2. Technical documentation on how to add a component (e.g. pipeline) to Astra

3. Distribute technical and overview documentation to interested parties
   (MWM email list, Astra email list SMAUG list, SDSS V software group, etc)
   and invite comments

4. Create an initial version of Astra framework that can do the following from
   command line interfaces: 
    
  - Setup database structure for components (e.g., pipelines), data paths,
    schedule analysis jobs, etc.

  - Create, replace, update, delete (CRUD) functionality for components
    (e.g., pipelines)

  - CRUD functionality for triggering analysis jobs

5. Wrap up ASPCAP as a component

6. Wrap up The Cannon as a component

7. Wrap up hot star code as a component

8. Wrap up The Payne as a component

9. Wrap up IN-SYNC as a component

10. Wrap up WD code as a component? (awaiting code)

11. Wrap up The Chat code as a component? (awaiting code)

12. Create a web interface for monitoring data flow, CRUDding components, and
    watching analysis job status, etc.

13. CRUD functionality for "experiments" to compare results between versions, etc

14. Enable "experiments" functionality through Astra website.

15. Modularize common components:

    - Continuum normalisation procedures

    - Spectrum synthesis generation (by various methods)





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
