.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Installation

:tocdepth: 2

.. rubric:: :header_no_toc:`Installation`

Astra can be installed in two different ways:

1. Using modules to manage component versions and their dependencies; or
2. Using `Conda <http://docs.conda.io/>`_ with **only** the most up-to-date component versions. 

The first method is how Astra is installed on SDSS inrastructure. If you want to compare results between versions of different components with time, you probably want to use this method. If you only want to test and/or develop a component locally, then it should be okay to use the second method.  

.. _using-modules:

Using `modules`
---------------

If you already use `TCLSH modules <http://modules.sourceforge.net/>`_ or `LUA modules <http://lmod.sourceforge.net/>`_  then you can install Astra in such a way that you can manage multiple different versions of components, and their dependencies. This allows you to track changes in survey results as they change with time.

Alternatively, you can install Astra using a Conda environment (see :ref:`using-conda`).

Install Astra
~~~~~~~~~~~~~

The following instructions will install Astra using the `sdss_install <https://github.com/sdss/sdss_install>`_ tool. The first thing you will need to do is make sure that you have either `TCLSH modules <http://modules.sourceforge.net/>`_ or `LUA modules <http://lmod.sourceforge.net/>`_ installed, then follow the steps below. These instructions are modified from Benjamin Murphy's `guide <https://wiki.sdss.org/display/knowledge/sdss_install+bootstrap+installation+instructions>`_ for installing `sdss_install`.

1.  Create a new directory under which all of your SDSS-related software will be built, and associated module files. 
For example::
    mkdir -p ~/software/sdss/github/modulefiles
2.  Set the ``SDSS_INSTALL_PRODUCT_ROOT`` environment variable to the directory you created in the previous step, 
following that example::
    export SDSS_INSTALL_PRODUCT_ROOT=~/software/sdss 
3.  Add the ``sdss_install`` modulefiles directories to your module path, following the example from the 
previous 2 steps::
    module use ~/software/sdss/github/modulefiles
    module use ~/software/sdss/svn/modulefiles
4.  Clone ``sdss_install`` from 
SDSS GitHub::
    git clone https://github.com/sdss/sdss_install.git github/sdss_install/master
5.  Generate a GitHub `Personal Access Token <https://github.com/settings/tokens>`_ (see `this guide <https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line>`_) with read-only permissions and
set this token as an environment variable named ``SDSS_GITHUB_KEY``::
    export SDSS_GITHUB_KEY=abcdef123456
6.  Run the bootstrap 
installer::
    ./github/sdss_install/master/bin/sdss_install_bootstrap
7.  Now put ``sdss_install`` on 
your path::
    module load sdss_install
8.  Now you can install Astra
using the following command::
    sdss_install astra
9.  Now put Astra on 
your path::
    module load astra
10. Lastly, run the setup 
command for Astra (this only needs to be run once)::
    astra setup

Steps 2, 5, 7, and 9 may need to be added to your ``.bashrc`` or ``.tcshrc`` file for convenience.



Using Astra on SDSS infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Astra is already installed on SDSS infrastructure at Utah. To make it available in your current session you can use the
command::

  module load astra



.. _using-conda:


Using Conda
-----------

If you don't want to install modules locally then you can install Astra into a Conda environment. Installing Astra in this way means that you will *only* have one version of Astra components: you will not be able to easily compare results between different versions of components. If you only want to test, execute, or develop Astra then this kind of installation is fine.

If instead you want to be able to compare results between versions of Astra components, you should install Astra using modules (see :ref:`using-modules`).


Into a new environment
~~~~~~~~~~~~~~~~~~~~~~

With a local installation we use `Conda <http://docs.conda.io/>`_ to manage a *single* environment for Astra and all of its components to run from. Run the following commands to create an environment and install Astra::
  
  wget -O environment.yml https://raw.githubusercontent.com/sdss/astra/master/etc/environment.yml
  conda env create -f environment.yml

Activate the environment to confirm that everything installed correctly, and then run the `astra` command line tool::

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

That looks good. Now run the setup routine for Astra to initialise the database (this only needs to be run once)::

  (astra) io: astra setup

Now in order to add components (that are all hosted on GitHub) you will need an environment variable so that Astra can programatically access certain parts of GitHub. Follow `this guide <https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line>`_ to create a Personal Access Token. It's best to store your Personal Access Token as an environment variable. Adding something like this to your ``~/.bash_profile`` will do it::

  export SDSS_GITHUB_KEY=abcdef123456

And to enable that environment variable you may need to reload your ``~/.bash_profile``::

  source ~/.bash_profile

Now you're ready to start adding components and processing data!

If you ever need to de-activate the Astra environment then you can do so with::

  conda deactivate







To an existing environment 
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a Conda environment that you'd like to install Astra to (instead of having Astra in it's own environment) then you can install it from source::

  git clone git@github.com:sdss/astra.git
  cd astra/
  python setup.py install

Once installed, remember to run the setup routine (this only needs to be run once)::

  astra setup

which will intialise the database and other things.









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
structure. You can do this using (this only needs to be run once)::

  astra setup

Now you're good to go! 


Next you may want to read about `components <components>`_ or check out the
`getting started guides <guides>`_.
