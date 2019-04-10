
.. _components:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Components

:tocdepth: 2

.. rubric:: :header_no_toc:`Components`

The purpose of a *component* is to provide a command line utility that takes
in a path pointing to a SDSS V data product, and to output a data product. The
output data product might describe the result of some analysis, or it might be
a data model (e.g., a data model file with continuum normalisation applied).

Components overview
===================

An astronomer might find the term *component* synonymous with 'pipeline'.
The difference is that a pipeline might be a series of sequential steps to execute
in order to deliver a final answer (e.g., astrophysical parameters), whereas a
*component* is only expected to perform *at least* one task.
That task might be continuum normalisation, or it might be providing a 
classification for a type of object. However, a component *can* do more than one
task. For the purpose of incorporating existing 'pipelines' into the initial
version of Astra, we will describe a large pipeline as a *component* and seek
to modularize common components in the future. In other words, for the purposes
of Astra, ASPCAP could be described as a *component*, just as an object classifier
could be considered a pipeline component.


Components can be executed sequentially in Astra so that the outputs of one
component are accessible to other components. All Astra components will have 
access to the ``astra`` Python module, which includes utility functions to 
access targeting information about a source, retrieve outputs that other
components have produced about this source, and to access external databases
(e.g., Gaia photometry and astrometry).


What makes a component?
=======================

A valid Astra component must meet the following requirements:

1. It must be stored in a public ``git`` repository on GitHub_, preferably in
   the `SDSS organization GitHub <http://github.com/sdss>`_.

2. A component must be `tagged <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_. 
   All components have full version control through ``git``, but only tagged 
   versions of components will be considered as a 'component update'.

3. A component must have a command line utility called ``sdss-astra-accepts`` (name TBD)
   that takes the path to a SDSS data model file as an input, and returns 
   *whether or not* it can provide an analysis of that source/file. Given the
   input observation file, this utility will be able to access all relevant
   information about targetting, etc, through the functions in ``astra.utils``.

4. A component must have at least one command line utility that takes as an 
   argument the path to a SDSS data model, and produces an output file that
   is a valid data model [#]_.

Components must be self-contained Python packages that fully describe all of the
required dependencies, and can be installed using ``python setup.py``. If there 
are deep dependency requirements on operating system, or software that
can not -- or should not -- be installed from ``setup.py``, then components can
also have a Docker container file that describes the environment. If no Docker
container is provided then the component will be run in the `standard Astra environment <#>`_.


Adding a component to Astra
===========================

Eventually you will be able to add components to Astra through a website.
For now you should use the ``sdss-astra`` command line utility. To add a
component you will to specify the relevant information in the following format::

  short_name: Continuum normalization
  github_repo_slug: sdss/my-continuum-normalization
  owner: Andy Casey
  owner_email_address: andrew.casey@monash.edu
  execution_order: 10
  component_cli: continuum-normalize 

Component execution order
^^^^^^^^^^^^^^^^^^^^^^^^^

Most keywords in the above example are self-explanatory. The ``execution_order`` 
key **only** matters for components that rely on the output of other components. 
If your component does not rely on the output of any other components (and does 
not provide outputs that will reasonably be used by other components) then you 
can set ``execution_order: 0``.

If there are five components that are to run on a given observation, then those
components will be executed in order of ascending non-negative execution order 
(``1`` indicates the first execution order). If your component in some part 
relies on the outputs of other components, then you should set your 
``execution_order`` to be higher than those other components, otherwise you
will not be able to access the outputs of those components.


Astra periodically checks for new tagged versions to existing components, and
will update itself automatically [#]_.

TBD: how to add a component using the ``astra`` command line utility.

TBD: how to edit aspects of components using the ``astra`` command line utility


TBD: Resources and utiltiies that each component has access to (e.g. ``astra.utils``)


Component command line arguments
--------------------------------

An Astra command line tool must accept the following arguments::

  --path: [single data file to run on]

  --f: [read paths from a file provided]

  --others common to all?


TBD: [An example python file that takes in the arguments and does something]

TBD: [How to deal with ``-f`` flag for just deciding whether it can analyse it or not?]

TBD: [Input/output data file paths...]




.. _GitHub: http://www.github.com/

.. [#] What constitutes a 'valid data model' for output is still to be determined,
       but it could look something like either a FITS data model file, or a
       YAML-like output file.

.. [#] When there is a live version of Astra running continuously this will make
       use of GitHub_ webhooks to be notified of version changes.