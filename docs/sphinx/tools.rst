
.. _astra-tools:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Astra command line tools

:tocdepth: 3

.. rubric:: :header_no_toc:`Tools`


``astra``
========

The ``astra`` command line tool provides most of the functionality needed
to manage data flows, analysis components, scheduling and monitoring of analysis 
tasks, and bookkeeping the outputs from those analysis tasks.

This documentation page will be automatically generated using the help strings
of each function, but for now you can see a list of commands that are expected
to be available in the ``astra`` tool (e.g., ``astra data watch``):

=====================  =============
   Command              Description
=====================  =============
``setup``              Run the setup procedure to initialise Astra
``data watch``         Watch a folder for new SDSS data products
``data unwatch``       Stop watching a folder for new SDSS data products
``data refresh``       Check for new data in all watched folders
``component create``   Add a component to Astra
``component refresh``  Check all components for new tagged versions
``component update``   Update attributes of an existing component
``component delete``   Delete an existing component
``subset create``      Name a subset of the data that is recognizable by all components
``subset update``      Update the paths that form part of an existing named subset
``subset delete``      Delete an existing named subset
``query-execute``      Query which components will run on the given data path
``execute``            Execute a component (or all components) on some given data path
``schedule``           Schedule the execution of component(s) on some given data path
=====================  =============

Setup
-----

The ``setup`` command will set up Astra according to the settings given in the
current configuration file. If an Astra database already exists and components
have been installed already, these will be removed.

.. click:: astra.tools.sdss_astra.setup:setup
   :prog: astra setup


Data
----

.. click:: astra.tools.sdss_astra.data:watch
   :prog: astra data watch

.. click:: astra.tools.sdss_astra.data:unwatch
   :prog: astra data unwatch

.. click:: astra.tools.sdss_astra.data:refresh
   :prog: astra data refresh

Components
----------

.. click:: astra.tools.sdss_astra.component:create
   :prog: astra component create

.. click:: astra.tools.sdss_astra.component:refresh
   :prog: astra component refresh

.. click:: astra.tools.sdss_astra.component:update
   :prog: astra component update

.. click:: astra.tools.sdss_astra.component:delete
   :prog: astra component delete


Subsets
-------

[TBD: auto-fill arguments from help string]

[TBD: subsets can follow regular expression patterns]

[TBD: auto-fill arguments from help string]

Query execution
---------------

[TBD: auto-fill arguments from help string]
[TBD: better name for ``query-execute``?]

Execute
-------

.. click:: astra.tools.sdss_astra.execute:execute
   :prog: astra execute


Schedule execution
------------------

[TBD: auto-fill arguments from help string]

Experiments
-----------

[TBD: ``experiment create/update/delete``]

