.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Installation

:tocdepth: 2

.. rubric:: :header_no_toc:`Installation`

|astra| can be installed in two different ways:

1. Using a Conda_ environment, or
2. Using modules to manage component versions and their dependencies.

If you don't know what you want, you probably want to use a Conda_ environment.
This will be suitable for testing and/or developing some part of |astra| locally. 
The second method is how |astra| is installed on SDSS inrastructure. 
You can expect the same results regardless of how you installed |astra|. 



.. _using-conda:


Using Conda
-----------

If you don't want to install modules locally then you can install |astra| into a Conda_ environment. 
If you only want to test, execute, or develop |astra| then this kind of installation is fine.


Into a new environment
~~~~~~~~~~~~~~~~~~~~~~

With a local installation we use Conda_ to manage a *single* environment for |astra| and all of its
components to run from. Run the following commands to create an environment and install |astra|::
  
  wget -O environment.yml https://raw.githubusercontent.com/sdss/astra/master/etc/environment.yml
  conda env create -f environment.yml

Now, activate the Conda environment::

  conda activate astra

To install the bleeding-edge version of |astra|, use::

  git clone https://github.com/sdss/astra.git 
  cd astra/
  python setup.py install





To an existing environment 
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a Conda_ environment that you'd like to install |astra| to (instead of having 
|astra| in it's own environment) then you can install it from source::

  git clone https://github.com/sdss/astra.git 
  cd astra/
  python setup.py install




.. _using-modules:

Using `modules`
---------------

If you already use `TCLSH modules <http://modules.sourceforge.net/>`_ or 
`LUA modules <http://lmod.sourceforge.net/>`_  then you can install |astra| in such a way that you 
can manage multiple different versions of components, and their dependencies. 
This allows you to track changes in survey results as they change with time.

Alternatively, you can install |astra| using a Conda_ environment (see :ref:`using-conda`).

Install |astra|
~~~~~~~~~~~~~~~


The following instructions will install |astra| using the |sdss_install|_ tool. 
The first thing you will need to do is make sure that you have either 
`TCLSH modules <http://modules.sourceforge.net/>`_ or `LUA modules <http://lmod.sourceforge.net/>`_ installed,
then follow the steps below.
These instructions are modified from Benjamin Murphy's `guide <https://wiki.sdss.org/display/knowledge/sdss_install+bootstrap+installation+instructions>`_ for installing |sdss_install|_.

#. Create a new directory under which all of your SDSS-related software will be built, and associated module files. 
   For example::

     mkdir -p ~/software/sdss/github/modulefiles

#. Set the ``SDSS_INSTALL_PRODUCT_ROOT`` environment variable to the directory you created. 
   For example:: 

     export SDSS_INSTALL_PRODUCT_ROOT=~/software/sdss 

#. Add the |sdss_install|_ modulefiles directories to your module path.
   Following the example from the previous 2 steps::

     module use ~/software/sdss/github/modulefiles
     module use ~/software/sdss/svn/modulefiles

#. Clone |sdss_install|_ from GitHub::

     git clone https://github.com/sdss/sdss_install.git github/sdss_install/master

#. Generate a GitHub `Personal Access Token <https://github.com/settings/tokens>`_ 
   (see `this guide <https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line>`_) 
   with read-only permissions and set this token as an environment variable named ``SDSS_GITHUB_KEY``::

     export SDSS_GITHUB_KEY=abcdef123456

#. Run the bootstrap installer::

     ./github/sdss_install/master/bin/sdss_install_bootstrap

#. Now put |sdss_install| on your path::

     module load sdss_install

#. Now you can install |astra| using the following command::

     sdss_install astra

#. Now put |astra| on your path::

     module load astra

Steps 2, 5, 7, and 9 can be added to your ``.bashrc`` or ``.tcshrc`` file so that you don't have to execute them
again every time you load a new terminal.



Using |astra| on SDSS infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|astra| is already installed on SDSS infrastructure at Utah. 
To make it available in your current session you can use the command::

  module load astra



..
  TODO:
  Next you may want to read about `components <components>`_ or check out the
  `getting started guides <guides>`_.

..
  Comment: 
  We aren't able to do nested inline markup, so we use these hacks, which are not recommended.
  https://docutils.sourceforge.io/FAQ.html#is-nested-inline-markup-possible

.. |astra| replace:: `astra`

.. |sdss_install| replace:: ``sdss_install``
.. _sdss_install: https://github.com/sdss/sdss_install

.. _Conda: http://docs.conda.io
.. _sdss_install: https://github.com/sdss/sdss_install