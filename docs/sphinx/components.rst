
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

Components can be executed sequentially in Astra so that the outputs of one
component are accessible to other components. The execution order of those
components is managed by Astra. All Astra components will have access to the 
`Astra Python module <#>`_ (and other SDSS-related modules), which includes 
utility functions to access targeting information about a source, retrieve 
outputs that other components have produced about this source, and to access 
external databases (e.g., Gaia photometry and astrometry). A valid Astra 
component must meet the following requirements:

1. It must be stored in a public ``git`` repository on GitHub_, preferably in
   the `SDSS organization GitHub <http://github.com/sdss>`_.

2. A component must have at lease one `release <https://help.github.com/en/articles/creating-releases>`.
   All components have full version control through ``git``, but only new
   releases of components will be considered a sufficiently substantive change
   to trigger (and differentiate between) analysis tasks.

3. A component must have a function called ``saqe`` (short for 'SDSS Astra query 
   execute') that takes the path to a SDSS data product and returns *whether or
   not** it can provide an analysis of that source. Given the data product path,
   this utility will be able to access all relevant information in order to make
   a decision. This could include targeting information, photometry and
   astrometry from external catalogues (e.g., Gaia), and results from other
   Astra components that have run on that data product. For example, the results
   of a classifier.

4. A component must have at least one command line utility that takes as an 
   argument the path to a SDSS data model, and produces an output file that
   is a valid data model [#]_.


.. note::
    In the future the ``saqe`` requirement may no longer exist if we move to a 
    conductor-driven execution approach (see `roadmap <roadmap.htm#road-mapl>`_), 
    but the ``saqe`` function will be necessary until components have iterated 
    into a steady-state mode.


Components must be self-contained Python packages that fully describe all of the
required dependencies, and can be installed using ``python setup.py``. If your
component has dependencies that are not available on the SDSS systems at Utah, 
then Astra will identify those discrepancies [TODO: get link to CPHC systems to
request a python component]. Any module dependencies that are required on Utah
can be specified in your component.


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


If your component's GitHub repository does not fall under the SDSS organization,
then you will need to specify the ``--owner`` flag. The ``--version`` flag
indicates the release tag on GitHub. If no ``--version`` is given then Astra
will find the most recent version on GitHub.

Here are some components that you might be interested in adding to Astra:

  - FERRE ()
  - The Cannon ()
  - INSYNC ()
  - The Payne ()
  - GSSP ()

If you want all of these components then you can use the commands::

  astra component add astra_ferre
  astra component add astra_thecannon
  astra component add astra_insync
  astra component add astra_thepayne
  astra component add astra_gssp



Component execution order
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``execution_order`` key **only** matters for components that rely on the 
output of other components. If your component does not rely on the output of any
other components (and does not provide outputs that will reasonably be used by 
other components) then you can set ``execution_order: 0``.

If there are five components that are to run on a given observation, then those
components will be executed in order of ascending non-negative execution order 
(``1`` indicates the first execution order). If your component in some part 
relies on the outputs of other components, then you should set your 
``execution_order`` to be higher than those other components, otherwise you
will not be able to access the outputs of those components.



Component command line interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``command`` describes the command line utility in your component that
is to be executed by Astra. Ideally this should be installed as a 
``console_scripts`` entry point in your ``setup.py`` file. Every command line 
tool that describes a component in Astra **must** accept and follow the following 
arguments:

=================  =============================================
 Argument           Description
=================  =============================================
``input_path``     the path to the input data model file
``output_dir``     the directory for output products produced by the component
``-i``             read the input paths from a local file
``-v``             verbose output
=================  =============================================
  

In our example component described in ``my-component.yml`` the typical use case 
for a single observation would be::

  continuum-normalize -v {input_path} {output_dir}

and the outputs would be written to the ``output_dir`` directory. Here is an 
example Python script that can be executed as a shell utility::

  from __future__ import (absolute_import, division, print_function, unicode_literals)  

  import click
  from numpy.random import choice  

  @click.command()
  @click.argument("input_path")
  @click.argument("output_dir")
  @click.option("-i", "read_from_path", default=False, is_flag=True,
                help="read input data paths from the given input path")
  @click.option("-v", "verbose", default=False, is_flag=True,
                help="verbose mode")
  def is_executable(input_path, output_dir, read_from_path, verbose):
      if verbose:
          click.echo(f"{input_path} > {output_dir} / {read_from_path} / {verbose}")
      decision = choice([True, False])
      click.echo(decision)
      return decision  

  if __name__ == "__main__":
      is_executable()


You are not required to use ``click``; you can use the built-in ``argparse``
module (or anything similar) if you want. You just need to specify these
dependencies in your ``setup.py`` file.

[TBD: how to manage ``output_dir`` products when the ``-i`` flag is used]


Updating components
===================

All attributes relating to a component can be updated **except** the
``github_repo_slug``. Attribuets can be updated using the ``astra`` tool::

  astra component update {github_repo_slug} --active true

[TBD: more examples of things to alter]

[TBD: one repo for training and one for testing data-driven models? or update 
based on ``component_id``? only require ``component_id`` when there is some
ambiguity?]

Deleting components
===================

You will rarely need to delete components because you can just mark them as
inactive and they will no longer be run on any observations. If you do need
to delete a component you can do so using::

  astra component delete {github_repo_slug}

It will ask you if you are sure. You can use the ``-y`` flag to indicate yes and
skip this question.

Executing components
====================

You can directly execute a component using the ``astra`` utility. For example::

  astra execute the-cannon -i training-paths.txt -o tmp/ --train --data-release 16

will train a Cannon model using the data files listed in the text file 
(``training-paths.txt``) and use Data Release 16 labels for those 
observations. The output model would be written to the ``tmp/`` directory.

In production mode Astra will schedule the execution of relevant components when
new data products are found in a watched folder. For each reduced data product,
Astra will query each component (using ``saqe``) to see whether that component
would analyze the given data file. This will be described as component-driven
design, in contrast to something like a conductor-driven design where one actor
decides which components should be executed for a given observation.

The concept of component-driven design implies that no one component can govern
how another component behaves. All data could, in principle, be processed by all
active components. In the simpler case of SDSS-IV/APOGEE, the equivalent ``saqe`` 
utility might simply return ``True`` if the given data file followed the SDSS 
data model format for APOGEE spectra, and ``False`` otherwise. In Astra, the 
decision about whether a component *should* process some observation could 
depend on:

- the specified data model (e.g., APOGEE or BOSS), 
- inputs from other components (e.g., a suite of classifiers), 
- some targeting information 
- or other external data (e.g., Gaia), 
- or it could depend on the values in the data array itself (e.g., Are there any finite data values? is the estimated S/N value above some threshold?). 

For these reasons, each component makes the decision about what it *should* be 
able to process, and Astra's role is to maintain version control, streamline 
data processing and task allocation, and to manage book-keeping of all component 
results.

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


Registering data models
=======================

Select outputs from registered data models will be stored in the Astra database
for book-keeping, cross-reference, comparisons, and to be accessible to other
components.

[TBD: this is a hard one. Inputs are easier than outputs. There will be some
declarative way to describe the data model of your components' outputs, and 
ths will need to be stored in the component's GitHub repository somewhere.
See the `roadmap <roadmap.html#roadmap>`_]


Examples
========

Physics-driven model component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[TBD: give example repository showing how to package model data files]

Data-driven model component
^^^^^^^^^^^^^^^^^^^^^^^^^^^

[TBD: give example repository showing how to create a component that trains a model based on 
existing SDSS data, and then uses that model for inference on new data]


.. _GitHub: http://www.github.com/

.. [#] Preferably only one task.

.. [#] What constitutes a 'valid data model' for output is still to be determined,
       but it could look something like either a FITS data model file, or a
       YAML-like output file.

.. [#] When there is a live version of Astra running continuously this will make
       use of GitHub_ webhooks to be notified of version changes.