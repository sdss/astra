
.. _astra-tools:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Astra command line tools

:tocdepth: 2

.. rubric:: :header_no_toc:`Tools`


``sdss-astra``
========

The ``sdss-astra`` command line tool provides most of the functionality needed
to manage data flows, analysis components, scheduling and monitoring of analysis 
tasks, and bookkeeping the outputs from those analysis tasks.

This documentation page will be automatically generated using the help strings
of each function, but for now you can see a list of commands that are expected
to be available in the ``sdss-astra`` tool (e.g., ``sdss-astra data watch``):

=====================  =============
   Command              Description
=====================  =============
``setup``              Run the setup procedure to initialise Astra
``data watch``         Watch a folder for new SDSS data products
``data unwatch``       Stop watching a folder for new SDSS data products
``data refresh``       Check for new data in all watched folders
``subset create``      Name a subset of the data that is recognizable by all components
``subset update``      Update the paths that form part of an existing named subset
``subset delete``      Delete an existing named subset
``component create``   Add a component to Astra
``component refresh``  Check all components for new tagged versions
``component update``   Update attributes of an existing component
``component delete``   Delete an existing component
``query-execute``      Query which components will run on the given data path
``execute``            Execute a component (or all components) on some given data path
``schedule``           Schedule the execution of component(s) on some given data path
=====================  =============


Setup
-----

Needs to only be run once. 

[TBD: auto-fill arguments from help string]


Data
----

[TBD: auto-fill arguments from help string]

Subsets
-------

[TBD: auto-fill arguments from help string]

[TBD: subsets can follow regular expression patterns]

Components
----------

[TBD: auto-fill arguments from help string]

Query execution
---------------

[TBD: auto-fill arguments from help string]
[TBD: better name for ``query-execute``?]

Execute
-------

[TBD: auto-fill arguments from help string]

Schedule execution
------------------

[TBD: auto-fill arguments from help string]

Experiments
-----------

[TBD: ``experiment create/update/delete``]

