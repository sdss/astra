
.. _components:

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Components

:tocdepth: 2

.. rubric:: :header_no_toc:`Components`

A *component* in Astra describes an analysis tool that can run on some subset
(or all!) of SDSS data. Each component should have one command line tool
that takes in input paths pointing to SDSS V data products, and to output some
data products. The output data proucts might describe the result of some analysis, 
or they might be an "intermediate" data product (e.g., a data model file with continuum 
normalisation applied).


Definitions
===========

An astronomer might find the term *component* synonymous with 'pipeline'. 

The difference here is that a pipeline might be a series of sequential steps to 
execute in order to deliver a final answer (e.g., astrophysical parameters),
whereas a *component* is only expected to perform *at least* one task [#]_. 
That task might be continuum normalisation, or it might be providing a 
classification for a type of object. However, a component *can* do more than 
one task. 

For the purpose of incorporating existing 'pipelines' into the initial version 
of Astra, we will describe a large pipeline as a *component* and seek to 
modularize common components in the future. In other words, for the purposes of 
Astra, *ASPCAP* could be described as a *component*, just as an object classifier 
could be considered a pipeline component.


What makes a component?
=======================


Components must be self-contained Python packages that fully describe all of the
required dependencies. If you intend to write your own component, please see
the `writing your own component <#>`_ guide. 

A valid Astra component must meet the following requirements:

1. It must be stored in a public ``git`` repository on GitHub, preferably in
   the `SDSS organization GitHub <http://github.com/sdss>`_.

2. A component must have at lease one `release <https://help.github.com/en/articles/creating-releases>`_.
   All components have full version control through ``git``, but only new
   releases of components will be considered a sufficiently substantive change
   to trigger (and differentiate between) analysis tasks.

3. A component must have a function called ``saqe`` (short for 'SDSS Astra query 
   execute') that takes the path to a SDSS data product and returns **whether or
   not** it can provide an analysis of that source. 

4. A component must have at least one command line utility that takes as an 
   argument the path to a SDSS data model, and produces an output file that
   has a valid SDSS data model.


Requirement #3 above implies that no one component can govern how another
component behaves. All data could, in principle, be processed by all components. 
In the simpler case of SDSS-IV/APOGEE, the equivalent ``saqe`` utility might 
simply return ``True`` if the given data file is an APOGEE spectrum, and ``False``
otherwise. In Astra the decision about whether a component *can* process some 
observation could depend on:

- the specified data model (e.g., APOGEE or BOSS), 
- targeting information,
- photometry and astrometry from external catalogues,
- inputs from other components (e.g., a suite of classifiers), or
- the flux array values themselves (e.g., Are there any finite data values? Is the S/N sufficient?)

For these reasons, each component makes the decision about what it *should* be 
able to process, and Astra's role is to maintain version control, streamline 
data processing and task allocation, and to manage book-keeping of all component 
results.

In the future this ``saqe`` requirement may no longer exist if we move to a 
conductor-driven execution approach (see `roadmap <roadmap.htm#road-mapl>`_), 
but the ``saqe`` function will be necessary until components have iterated 
into a steady-state mode.


.. attention::
    Just because multiple components might analyse the same observation does not
    mean that all results will form part of the data release candidate! As an 
    example, Astra would keep the results from one component that has been 
    improved over time (with many tagged versions), and each time that component 
    has been run over a subset of the data. Those earlier results will not form 
    part of a data release: they are merely to track and compare results over 
    time. It will be the responsibility of the data release coordinators to 
    decide what components (and versions) will contribute the results to a data 
    release candidate.

    Keeping all relevant results between component versions in Astra will allow 
    collaborators to iterate and improve their components, whilst automating
    much of the requisite scientific verification that comes with making those
    component changes.



Adding a component to Astra
===========================

Once you have run ``astra setup`` you will be able to add components to Astra.
If you are running Astra locally then you must add components using the ``astra``
command line tool. If you want to add components to Astra on the SDSS systems at
Utah then you can do so through a SSH terminal, or eventually through an
internal collaboration website that will be available in early 2020.

To add a component to Astra you will need the name of a GitHub repository. If
you type ``astra component add --help`` in a terminal then this is the output
you can expect::

    ~$ astra component add --help
    Usage: astra component add [OPTIONS] PRODUCT

      Add a new component in Astra from an existing GitHub repository
      (`product`) and a  command line tool in that repository (`command`).

    Options:
      --version TEXT             The version of this product to use. If no version
                                 is given then this will default to the last
                                 release made available on GitHub.
      --owner TEXT               The owner of the repository on GitHub (default:
                                 sdss).
      --execution-order INTEGER  Set the execution order for the component
                                 (default: 0).
      --command TEXT             Specify the name of the command line utility to
                                 execute from that component. This is only
                                 required if there are more than one executable
                                 components in the bin/ directory of that
                                 repository.
      --description TEXT         A short description for this component. If no
                                 description is given then this will default to
                                 the description that exists on GitHub.
      -a, --alt-module TEXT      Specify an alternate module name for this
                                 component.
      --default-args TEXT        Default arguments to supply to the command.
      -t, --test                 Test mode. Do not actually install anything.
      --help                     Show this message and exit.


If your component's GitHub repository does not fall under the `SDSS organization GitHub <http://github.com/sdss>`_
then you will need to specify the ``--owner`` flag. The ``--version`` flag
indicates the release tag on GitHub. If no ``--version`` is given then Astra
will find the most recent version on GitHub.

Here are some components that you might be interested in adding to Astra:

- `FERRE <https://github.com/sdss/astra_ferre>`_ (`Allende-Prieto et al. <https://ui.adsabs.harvard.edu/abs/2015AAS...22542207A/abstract>`_; `website <http://www.as.utexas.edu/~hebe/ferre/>`_/`user guide <http://www.as.utexas.edu/~hebe/ferre/ferre.pdf>`_) interpolates between a grid of synthetic spectra and compares the interpolated spectra with observations.
- `The Cannon <https://github.com/sdss/astra_thecannon>`_ (`Ness et al. <https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N/abstract>`_) for building a data-driven model of stellar spectra.
- `INSYNC <https://github.com/sdss/astra_insync>`_ (Cottaar; `original repository <https://bitbucket.org/mcottaar/apogee/src/master>`_) estimates stellar parameters and veiling for young star spectra.
- `The Payne <https://github.com/sdss/astra_thepayne>`_ (`Ting et al. <https://ui.adsabs.harvard.edu/abs/2018arXiv180401530T/abstract>`_) trains a single layer fully connected neural network on synthetic spectra.
- `GSSP <https://github.com/sdss/astra_gssp>`_ (`Tkachenko <https://ui.adsabs.harvard.edu/abs/2015A%26A...581A.129T/abstract>`_ ; `website <https://fys.kuleuven.be/ster/meetings/binary-2015/gssp-software-package>`) performs a grid search in stellar parameters and is typically used to analyse hot star spectra.

If you want all of these components then you can use the commands::

  astra component add astra_ferre
  astra component add astra_thecannon
  astra component add astra_insync
  astra component add astra_thepayne
  astra component add astra_gssp


Astra will fetch and install all of these components and make them accessible
through `modules <https://github.com/cea-hpc/modules>`_. 

.. note:: 
    If you are adding a new component to the SDSS systems at Utah and your
    component has dependencies that do not exist at Utah, then you will need
    to `submit a request <#>`_ to have your dependencies installed.


Component execution order
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``--execution-order`` option **only** matters for components that rely on the 
output of other components, and if you are running in `continuous data analysis mode <#>`_. 
If your component does not rely on the output of any other components -- and 
does not provide outputs that will reasonably be used by other components -- 
then you can leave the default value of zero.

If there are five components that are to run on a given observation, then those
components will be executed in order of ascending non-negative execution order 
(``1`` indicates the first execution order). If your component in some part 
relies on the outputs of other components, then you should set your 
``--execution-order`` to be higher than those other components, otherwise you
will not be able to access the outputs of those components.


Component command line interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``command`` describes the command line utility in your component that
is to be executed by Astra. Ideally this should be specified as a ``script``
keyword to ``setup()`` in your ``setup.py`` file. Every command line tool that
describes a component in Astra **must** accept and follow the following
arguments (specified by the :func:`astra.tools.parsers.common.component`
function).

======================  =============================================
 Argument               Description
======================  =============================================
``input_path``          the path to the input data model file
``output_dir``          the directory for output products produced by the component
``-i``/``--from-file``  read the input paths from a local file
``-v``                  verbose output
======================  =============================================
  

.. note::
    If you are are writing a component to add to Astra, then you should look at
    the [guide to writing your own component].


Executing components
====================

Components can be executed directly using the ``astra`` command line tool. If
you are running in `continuous data analysis mode <#>`_ then Astra will manage
the scheduling and execution of components for new data products as they
appear.


You can execute components manually using the ``astra`` command line tool::

    ~$ astra execute --help
    Usage: astra execute [OPTIONS] COMPONENT INPUT_PATH OUTPUT_DIR [ARGS]...

      Execute a component on a data product.

    Options:
      -i, --from-file    specifies that the INPUT_PATH is a text file that
                         contains a list of input paths that are separated by new
                         lines
      --timeout INTEGER  Time out in seconds before killing the task.
      --help             Show this message and exit.



Examples
========

The FERRE component in Astra has a debug mode that you can use to test that
things are being executed. To access this mode, use the following commands::

  astra component add astra_ferre
  astra execute astra_ferre . . --debug


There are Getting Started guides available for all existing Astra components:

- `Getting started with Astra and FERRE <#>`_
- `Getting started with Astra and INSYNC <#>`_
- `Getting started with Astra and The Payne <#>`_
- `Getting started with Astra and The Cannon <#>`_

