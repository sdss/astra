.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Installation

:tocdepth: 2

.. rubric:: :header_no_toc:`Installation`

Astra can be installed in two different ways:

1. Using a `Conda <http://docs.conda.io/>`_ environment, or
2. Using modules to manage component versions and their dependencies.

If you don't know what you want, you probably want to use a `Conda <http://docs.conda.io>`_ environment. This will be suitable for testing and/or developing some part of Astra locally.
The second method is how Astra is installed on SDSS inrastructure. You can expect the same results regardless of how you installed `astra`. 



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

Now, activate the Conda environment::

  conda activate astra

To install the bleeding-edge version of Astra, use::

  git clone https://github.com/sdss/astra.git 
  cd astra/
  python setup.py install





To an existing environment 
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a Conda environment that you'd like to install Astra to (instead of having Astra in it's own environment) then you can install it from source::

  git clone https://github.com/sdss/astra.git 
  cd astra/
  python setup.py install




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







Next you may want to read about `components <components>`_ or check out the
`getting started guides <guides>`_.
