.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Installation

:tocdepth: 2

.. rubric:: :header_no_toc:`Installation`

Astra needs to be installable in two different ways:
  - Locally (on your computer or cluster) for testing and development
  - On the SDSS infrastructure, largely hosted at Utah

For most users, Astra will run the same in either installation style. The differences are subtle and will depend on whether modules are installed or not. **TODO**: Andy write more about this.

Install
-------

Local installation
~~~~~~~~~~~~~~~~~~

With a local installation we use `Conda <http://docs.conda.io/>`_ to manage a _single_ environment for Astra and all of its components to run from. Run the following commands to create an environment and install Astra:
  
  wget -O environment.yml https://raw.githubusercontent.com/sdss/astra/master/etc/environment.yml
  conda env create -f environment.yml

Activate the environment to confirm that everything installed correctly, and then run the `astra` command line tool:

  io: conda activate astra
  (astra) io: astra --help
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

That looks good. Now run the setup routine for Astra to initialise databases, etc.

  (astra) io: astra setup

Now you're ready to start adding components and processing data. If you ever need to de-activate the Astra environment you can do so with:

  conda deactivate




Local installation to your existing environment 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a Conda environment that you'd like to install Astra to (instead of having Astra in it's own environment) then you can install it from source:

  git clone git@github.com:sdss/astra.git
  cd astra/
  python setup.py install

Once installed, remember to run the setup routine

  astra setup

which will intialise the database and other things.





Installing Astra at Utah
~~~~~~~~~~~~~~~~~~~~~~~~

You can install Astra on Utah using the ``sdss_install`` tool. Astra only needs to be installed
once, and then you can make it available in future sessions by using the command::

  module load astra

next time you log in.




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
