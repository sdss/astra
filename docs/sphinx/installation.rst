.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Installation

:tocdepth: 2

.. rubric:: :header_no_toc:`Installation`

This document will show you how to install and configure Astra in a self-contained environment.


Preparing an environment
------------------------

If you already use `Conda <http://docs.conda.io/>`_ then we recommend creating a temporary environment
for testing Astra.

You can create a file called `astra-test-environment.yml` that contains the following
(or `download it from here <https://github.com/sdss/astra/raw/master/etc/astra-test-environment.yml>`_)::

  name: astra-test
  channels:
  dependencies:
    - python=3.6.5
    - requests
    - pip:
      - "--editable=git+https://github.com/sdss/sdss_install.git@master#egg=sdss_install"

Now create a Conda environment using the following terminal command::

  conda create -f astra-test-environment.yml

And activate it using the command::

  source activate astra-test

Now you're ready to install Astra using the ``sdss_install`` tool (see below), and start using
Astra. Once you're done and you want to go back to your normal Conda environment, you can de-activate 
this environment using::

  source deactivate


Install
-------

Install Astra using ``sdss_install``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you created the Conda environment as described above, or you already have ``sdss_install``,
then you can install Astra using the following terminal command::

  sdss_install -G astra

If Astra has installed correctly then you should be able to run the following command::

  (astra-test) arc:~/$ astra --help
  Usage: astra [OPTIONS] COMMAND [ARGS]...

  Options:
    -v      verbose mode
    --help  Show this message and exit.

  Commands:
    component  Add, update, and delete components.
    execute    Execute a component on a data product.
    folder     Manage monitoring of data folders.
    setup      Setup Astra using the current configuration.
    subset     Create, update, and delete data subsets.
    task       Create, update, and delete tasks.


Install Astra from source
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can install Astra from source using the following terminal commands::

  git clone git@github.com:sdss/astra.git
  cd astra/
  python setup.py install

If Astra has installed correctly then you should be able to run the following command::

  (astra-test) arc:~/$ astra --help
  Usage: astra [OPTIONS] COMMAND [ARGS]...

  Options:
    -v      verbose mode
    --help  Show this message and exit.

  Commands:
    component  Add, update, and delete components.
    execute    Execute a component on a data product.
    folder     Manage monitoring of data folders.
    setup      Setup Astra using the current configuration.
    subset     Create, update, and delete data subsets.
    task       Create, update, and delete tasks.


Installing Astra at Utah
~~~~~~~~~~~~~~~~~~~~~~~~

You can install Astra on Utah using the ``sdss_install`` tool. Astra only needs to be installed
once, and then you can make it available in future sessions by using the command::

  module load astra

next time you log in.


Setup
-----

You will need to run the setup function once Astra is installed. You can do this using the following
terminal command::

  astra setup

And you should see a message telling you Astra is ready. If not, please `raise an issue on GitHub <https://github.com/sdss/astra/issues/new>`_.


Configuration
-------------

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