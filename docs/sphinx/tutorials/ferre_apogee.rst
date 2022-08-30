.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Running FERRE on APOGEE spectra

This document will guide you on how to execute FERRE on APOGEE spectra (from SDSS-IV).

Context
-------

It is expected that you have already `installed Astra <installation>`_ and run the
``astra setup`` command successfully.

If you dont already have it, you will need to install the FERRE component in Astra::

    astra component add astra_ferre --install-args --install-ferre

The ``astra_ferre`` part of this statement is the Python code that wraps FERRE and works with Astra. The optional arguments ``--install-args --install-ferre`` will download FERRE 4.6 and install it for you.
