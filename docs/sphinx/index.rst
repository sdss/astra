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

A *component* in Astra describes an analysis tool that can run on some subset
(or all!) of SDSS data. Once components are added, Astra will monitor select data
folders for new SDSS data products and schedule all relevant active components
to be executed. Certain outputs from those components will be stored and tracked
by Astra so the collaboration can evaluate changes in results with time, and 
make comparisons to previous sets of results.

The concept of component-driven design implies that no one component can govern
how another component behaves. All data could, in principle, be processed by all
active components. Each component must come with an associated utility that
returns whether or not it *should* be able to process a SDSS data model file.
In the simpler case of SDSS-IV, this utility might simply return ``True`` if the
given data file followed the SDSS data model format for APOGEE spectra, and
``False`` otherwise. In Astra, the decision about whether a component *should*
process some observation could depend on the specified data model (e.g., APOGEE
or BOSS), inputs from other components (e.g., a suite of classifiers), some
targeting information or other external data, or it could depend on the values
in the data array itself (e.g., Are there any finite data values? is the estimated
S/N value above some threshold?). For these reasons, each component makes the
decision about what it *should* be able to process, and Astra's role is to
maintain version control, streamline data processing and task allocation, and
to manage book-keeping of all component results.

It is important to note that just because multiple components may analyse the
same observation does not mean that all results will form part of the data 
release candidate! As an example, Astra would keep the results from one
component that has been improved over time (with many tagged versions), and
each time that component has been run over a subset of the data. Those 
earlier results will not form part of a data release: they are merely to 
track and compare results over time. It will be the responsibility of the
data release coordinators to decide what components (and versions) will 
contribute the results to a data release candidate.

For more details see:

* :ref:`Data flow <data>`
* :ref:`Components <components>`
* :ref:`Database design <database>`
* :ref:`[TBD: Astra setup (for local/Utah/etc)]`
* :ref:`Astra command line tools <astra-tools>`
* :ref:`Astra utilities <astra-utilities>`


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

- Setup database structure for components (e.g., pipelines), data paths, schedule analysis jobs, etc.

- Create, replace, update, delete (CRUD) functionality for components (e.g., pipelines)

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
