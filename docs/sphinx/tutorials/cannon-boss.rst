.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Running The Cannon on BOSS spectra

This document will guide you on how to create a training set for using `The Cannon <>`_,
show you how to train a data-driven model, and then use that model on new spectra.

The spectra that we will use in this tutorial is archival BOSS spectra.

Context
-------

It is expected that you have already `installed Astra <installation>`_ and run the
``astra setup`` command successfully.

If you are running this on Utah then you will need to make sure you have run
``module load astra`` after signing in.

If you are running this tutorial locally then you will need the BOSS archival spectra.


Create a training set
---------------------

A good training set should include high signal-to-noise spectra of sources (e.g., stars)
with high fidelity labels. In other words, it should include high quality stellar spectra
where we know the astrophysical parameters very well. The training set should cover the
range of stellar parameters that we are interested in testing.

Here we will use BOSS spectra with APOGEE labels. That means our first step is to
cross-match the BOSS and APOGEE data sets.
