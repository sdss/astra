.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Road map and priorities

:tocdepth: 2


Proritized tasks
================

Completed
^^^^^^^^^

*1. Documentation describing the overview of Astra, conceptual pieces, and how everything is expected to flow together

*2. Technical documentation on how to add a component (e.g. pipeline) to Astra

*3. Distribute technical and overview documentation to interested parties (MWM email list, Astra email list, SMAUG list, SDSS V software group, etc) and invite comments

*4. Create an initial version of Astra framework that can do the following from command line interfaces: 

- Setup database structure for components (e.g., pipelines), data paths, schedule analysis jobs, etc.

- Create, replace, update, delete (CRUD) functionality for components (e.g., pipelines)

- CRUD functionality for triggering analysis jobs

*5. Deploy initial version of Astra to Utah.

*6. Wrap up ASPCAP as a component

*7. Wrap up The Cannon as a component

*8. Wrap up The Payne as a component

*9. Wrap up IN-SYNC as a component


Upcoming
^^^^^^^^

*10. Wrap up hot star code as a component

*11. Wrap up WD code as a component? (awaiting code)

*12. Create a web interface for monitoring data flow, CRUDding components, and watching analysis job status, etc.

*13. CRUD functionality for "experiments" to compare results between versions, etc

*14. Enable "experiments" functionality through Astra website.

*15. Modularize common components (e.g., continuum normalisation, spectrum generation)


Road map
========

- **Component-driven execution of tasks / Conductor-driven execution of tasks**.
  Currently each *component* decides whether it would be able to analyze a
  given data file. This ensures that no component can govern whether another
  component should run or not. For example, if a classifier was deciding whether
  a source should be analyzed by a certain pipeline, then any time that classifer
  was wrong it would cause problems downstream. This is resolved if each
  component has a utility that decides whether it should be able to analyze this
  spectrum or not. However, having these utilities does incur unnecessary
  overhead in Astra. Once the components have been iterated upon and it is
  well-established what kinds of spectra they should process, it is worth 
  considering moving to a conductor-driven execution where a top-level actor
  decides which components should be executed.

- **Declarative data models for input and output data formats.**

- **How to manage** ``output_dir`` **when** ``-i`` **flag is used for components?**
  The output files should all exist in some declarative folder path like
  ``astra/results/{task_id}/...`` but for multiple objects in a single job we 
  will need sensible sub-folders to be created, which follow the SDSS data model
  principles. I know we can do this, but I am not certain that the SDSS V data
  models are specified well enough for us to declare this now.

- **Web interface with user-level control over tasks.**

- **Allow for experiments/hypotheses**. Users can set up an experiment (e.g., an
  alteration to an existing pipeline component) and it can be scheduled to run
  on a defined subset of data, and then Astra can generate specific plot
  comparisons to evaluate the impact of those component changes.

