.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Installation

:tocdepth: 2

.. rubric:: :header_no_toc:`Installation`


Preparing a Conda environment for testing
=========================================

If you already use Conda then we recommend creating a temporary environment for testing
Astra. You can create a file called `astra-test-environment.yml` that contains the following
(or download it from [here](LINK))::

  name: astra-test
  channels:
  dependencies:
    - python=3.6.5
    - requests
    - pip:
      - "--editable=git+https://github.com/sdss/sdss_install.git@master#egg=sdss_install"

Then you can create a Conda environment using the following terminal command::

  conda create -f astra-test-environment.yml

Now activate it using the command::

  source activate astra-test

Once you're done and you want to go back to your normal Conda environment, you can de-activate 
this environment using::

  source deactivate


Installing Astra using ``sdss_install``
=======================================

If you created the Conda environment as described above, or you already have ``sdss_install``,
then you can install Astra using the following terminal command::

  sdss_install -G astra


Installing Astra from source
============================

You can install Astra from source using the following terminal commands::

  git clone git@github.com:sdss/astra.git
  cd astra/
  python setup.py install




If you are running Astra on Utah then you only have to install Astra once, and you can make it
available by using the ``module load astra`` command next time you log in. If you have problems
running ``sdss_install`` then you may want to visit the `frequently encountered problems <#>`_
page to check that you have all the requisite environment variables set.




Configuration
=============

The configuration for Astra is specified by the ``python/astra/etc/astra.yml`` file. The most
relevant terms in this file are those that relate to the database. If you need to find where your
configuration file lives, use the following command::

  python -c "import astra, os; print(os.path.join(os.path.dirname(astra.__file__), 'etc/astra.yml'))"

By default, Astra will create and use a SQLite directory in your current working directory. If you
want to use a PostGreSQL server then you should update the ``etc/astra.yml`` file::

  database_config:
    host: localhost
    database: astra

Alternatively you can specify a database "connection string" for any kind of database. 

You can immediately start using the ``astra`` command line tool once you have installed and
configured Astra. The first time you run this tool should be to set up the database and folder
structure. You can do this using::

  astra setup

Now you're good to go! Next you may want to read about `components <components>`_ or check out the
`getting started guides <guides>`_.