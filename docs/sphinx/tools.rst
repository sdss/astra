
.. _astra-tools:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Astra command line tools

:tocdepth: 3

.. rubric:: :header_no_toc:`Tools`

**(This documentation will become out-of-date as Astra moves to using Luigi Tasks instead of command-line tools).**


``astra``
========

The ``astra`` command line tool provides most of the functionality needed
to manage data flows, analysis components, scheduling and monitoring of analysis 
tasks, and bookkeeping the outputs from those analysis tasks.

Here is an overview of the commands available through ``astra``, and more detailed
documentation is listed below:

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

.. click:: astra.tools.parsers.setup:setup
   :prog: astra setup


Components
----------

.. click:: astra.tools.parsers.component:add
   :prog: astra component add

.. click:: astra.tools.parsers.component:update
   :prog: astra component update

.. click:: astra.tools.parsers.component:list
   :prog: astra component list

.. click:: astra.tools.parsers.component:delete
   :prog: astra component delete


Execute
-------

.. click:: astra.tools.parsers.execute:execute
   :prog: astra execute


Data
----

.. click:: astra.tools.parsers.folder:watch
   :prog: astra folder watch

.. click:: astra.tools.parsers.folder:unwatch
   :prog: astra folder unwatch

.. click:: astra.tools.parsers.folder:refresh
   :prog: astra folder refresh


Subsets
-------

.. click:: astra.tools.parsers.subset:create
   :prog: astra subset create

.. click:: astra.tools.parsers.subset:list
   :prog: astra subset list

.. click:: astra.tools.parsers.subset:refresh
   :prog: astra subset refresh

.. click:: astra.tools.parsers.subset:update
   :prog: astra subset update


Tasks
-----

.. click:: astra.tools.parsers.task:create
   :prog: astra task create

.. click:: astra.tools.parsers.task:update
   :prog: astra task update

.. click:: astra.tools.parsers.task:delete
   :prog: astra task delete



Experiments
-----------

In the `roadmap <roadmap>`_ it is envisaged that you will be able to set up 
"experiments" in Astra such that multiple components (or versions of components)
can be executed on the same data set, and have custom quality control figures
produced in real time while that experiment is taking place.

[TBD: ``experiment create/update/delete``]

