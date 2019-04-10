.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Components

:tocdepth: 2

.. rubric:: :header_no_toc:`Components`

The purpose of a **component** is to provide a command line utility that takes
in a path pointing to a SDSS V data product, and output a data product (e.g., a
data product that describes analysis results). 

Overview
--------

An astronomer might find the term **component** synonymous with 'pipeline'.
The qualitative difference here is that a pipeline might be a series of steps in
sequence that is together expected to deliver a final answer (e.g., astrophysical 
parameters), whereas a component is only expected to perform *at least* one task. 
That task might be continuum normalisation, or it might be a classification on 
the type of object. However, a component can do more than one task: for the 
purpose of incorporating existing 'pipelines' in the initial version of Astra, 
we will describe a large pipeline as a **component** and seek to modularize 
common components in the future. In other words, for the time being ASPCAP could
be described as a **component**, just as an object classifier could be considered
a pipeline component.


What makes a component?
-----------------------

A valid Astra component must meet the following requirements:

- It must be stored in a git respository that is accessible to Astra (e.g., GitHub). 

- New versions of the component must be tagged in git. A freshly tagged version
  indicates that Astra should treat this as an update to the existing component.

- It must have a command line utility that takes in an SDSS V data product and
  outputs *whether or not* it will analyze that data product. 

- It must have a command line utility that takes in an SDSS V data product and
  produces an output.

- It must have a docker container that describes the environment that the
  component can run in. 



Creating a component
--------------------

- GitHub repository

- Version tagging

- A docker container

- A command line utility

- Adding the component to Astra

- Specifying what data should it run on

- Astra will automagically check for new tagged versions and run those components

- 



