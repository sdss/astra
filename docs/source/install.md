# Install

```{warning}
Version 0.3 of this package is unstable and under active development. If you want to test this version before it is released then you should install directly from source using the ``v0.3`` branch on GitHub. 
```

This warning message should be removed when version 0.3 is released. To install version v0.3 from source code using the ``v0.3`` branch on [GitHub](https://github.com/sdss/astra):

```bash
git clone -b v0.3 https://github.com/sdss/astra.git
cd astra
python setup.py install
```

`astra` can be installed in two different ways:

1. Using a [Conda](http://docs.conda.io) environment, or
2. Using modules to manage component versions and their dependencies.

If you don't know what you want, you probably want to use a [Conda](http://docs.conda.io) environment.
This will be suitable for testing and/or developing some part of `astra` locally. 
The second method is how `astra` is installed on SDSS infrastructure. 
You can expect the same results regardless of how you installed `astra`. 


## Using Conda

If you don't want to install modules locally then you can install `astra` into a [Conda](http://docs.conda.io) environment. 
If you only want to test, execute, or develop `astra` then this kind of installation is fine.


### Into a new environment

With a local installation we use [Conda](http://docs.conda.io) to manage a *single* environment for `astra` and all of its
components to run from. Run the following commands to create an environment and install `astra`:
```bash
wget -O environment.yml https://raw.githubusercontent.com/sdss/astra/master/etc/environment.yml
conda env create -f environment.yml
```

Now, activate the Conda environment:
```bash
conda activate astra
```

To install the bleeding-edge version of `astra`, use:
```bash
git clone https://github.com/sdss/astra.git 
cd astra/
python setup.py install
```


### To an existing environment 

If you already have a [Conda](http://docs.conda.io) environment that you'd like to install `astra` to (instead of having 
`astra` in it's own environment) then you can install it from source::

```bash
git clone https://github.com/sdss/astra.git 
cd astra/
python setup.py install
```

## Using modules

If you already use [TCLSH modules](http://modules.sourceforge.net/) or 
[LUA modules](http://lmod.sourceforge.net/)  then you can install `astra` in such a way that you 
can manage multiple different versions of components, and their dependencies. 
This allows you to track changes in survey results as they change with time.

Alternatively, you can install `astra` using a [Conda](http://docs.conda.io) environment.

### Install `astra`

The following instructions will install `astra` using the [`sdss_install`](https://github.com/sdss/sdss_install) tool. 
The first thing you will need to do is make sure that you have either 
[TCLSH modules](http://modules.sourceforge.net/) or [LUA modules](http://lmod.sourceforge.net/) installed,
then follow the steps below.
These instructions are modified from [Benjamin Murphy's guide](https://wiki.sdss.org/display/knowledge/sdss_install+bootstrap+installation+instructions) for installing [`sdss_install`](https://github.com/sdss/sdss_install).

1. Create a new directory under which all of your SDSS-related software will be built, and associated module files. 
   For example:
   ```bash
     mkdir -p ~/software/sdss/github/modulefiles
   ```

2. Set the ``SDSS_INSTALL_PRODUCT_ROOT`` environment variable to the directory you created. 
   For example:
   ```bash
   export SDSS_INSTALL_PRODUCT_ROOT=~/software/sdss 
   ```

3. Add the [`sdss_install`](https://github.com/sdss/sdss_install) modulefiles directories to your module path.
   Following the example from the previous 2 steps:
   ```bash
   module use ~/software/sdss/github/modulefiles
   module use ~/software/sdss/svn/modulefiles
   ```

4. Clone [`sdss_install`](https://github.com/sdss/sdss_install) from GitHub::
   ```bash
   git clone https://github.com/sdss/sdss_install.git github/sdss_install/master
   ```

5. Generate a GitHub [Personal Access Token](https://github.com/settings/tokens) 
   (see [this guide](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line)) 
   with read-only permissions and set this token as an environment variable named ``SDSS_GITHUB_KEY``:
   ```bash
   export SDSS_GITHUB_KEY=abcdef123456
   ```

6. Run the bootstrap installer:
   ```bash
   ./github/sdss_install/master/bin/sdss_install_bootstrap
   ```

7. Now put `sdss_install` on your path:
   ```bash
   module load sdss_install
   ```

8. Now you can install `astra` using the following command::
   ```bash
   sdss_install astra
   ```

9. Now put `astra` on your path::
   ```bash
   module load astra
   ```

Steps 2, 5, 7, and 9 can be added to your ``.bashrc`` or ``.tcshrc`` file so that you don't have to execute them
again every time you load a new terminal.



### Using `astra` on SDSS infrastructure

`astra` is already installed on SDSS infrastructure at Utah. 
To make it available in your current session you can use the command:
```bash
module load astra
```

```{warning}
Using `module load astra` on SDSS infrastructure will give you an old version of Astra. Until a stable v1 release is available, you should install from source using the v0.3 branch.
```

## Setting `astra` up for the first time

Once you've installed `astra` you will need to run a script to setup the local database:
```bash
> astra initdb
[INFO]: Connecting to database to create tables.
[INFO]: Tables (12): Source, DataProduct, SourceDataProduct, Output, Status, Task, TaskOutput, Bundle, TaskBundle, TaskInputDataProducts, TaskOutputDataProducts, AstraOutputBaseModel
[INFO]: Inserting Status rows
>
```